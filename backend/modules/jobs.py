"""Independent road-damage and road/lane jobs. Optional model stacks load only inside their job workers."""
from __future__ import annotations
import hashlib, json, os, sqlite3, threading, time, uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, Response
from .road_damage.schemas import DamageSettings, DamageReview
from .road_lanes.schemas import LaneSettings

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(os.getenv('ROADGUARD_DATA', ROOT/'data'))
DB=DATA/'roadguard.sqlite'
router=APIRouter(prefix='/api')
pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='infrastructure-model')
_model_guard=threading.Lock()
MAX_BYTES=200*1024*1024
IMAGES={'.jpg','.jpeg','.png','.webp','.bmp'}
VIDEOS={'.mp4','.mov','.avi','.mkv','.webm'}

def db():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row
    c.execute('CREATE TABLE IF NOT EXISTS module_jobs(id TEXT PRIMARY KEY,kind TEXT NOT NULL,name TEXT NOT NULL,status TEXT NOT NULL,progress REAL NOT NULL,result TEXT,error TEXT,created REAL NOT NULL,input_hash TEXT NOT NULL,settings TEXT NOT NULL)')
    c.execute('CREATE TABLE IF NOT EXISTS module_reviews(job_id TEXT NOT NULL,observation_id TEXT NOT NULL,status TEXT NOT NULL,note TEXT NOT NULL DEFAULT \'\',updated REAL NOT NULL,PRIMARY KEY(job_id,observation_id))')
    c.execute('CREATE TABLE IF NOT EXISTS module_model_status(model_key TEXT PRIMARY KEY,status TEXT NOT NULL,reason TEXT,revision TEXT,checksum TEXT,updated REAL NOT NULL)')
    return c

with db() as _conn:
    _conn.execute("UPDATE module_jobs SET status='failed',error='Processing interrupted by service restart; resubmit to retry' WHERE status IN ('queued','processing')")

def update(j,**fields):
    with db() as c: c.execute('UPDATE module_jobs SET '+','.join(f'{k}=?' for k in fields)+' WHERE id=?',[*fields.values(),j])

def getjob(j):
    with db() as c: row=c.execute('SELECT * FROM module_jobs WHERE id=?',(j,)).fetchone()
    if not row: raise HTTPException(404,'Module job not found')
    d=dict(row); d['settings']=json.loads(d['settings']); d['result']=json.loads(d['result']) if d['result'] else None
    if d['result']:
        with db() as c: reviews={r['observation_id']:dict(r) for r in c.execute('SELECT * FROM module_reviews WHERE job_id=?',(j,))}
        for o in d['result'].get('observations',[]): o['review']=reviews.get(o['id'],{}).get('status','pending'); o['review_note']=reviews.get(o['id'],{}).get('note','')
    return d

def _write_pdf(data,path):
    # Minimal self-contained one-page text PDF; no optional report package required.
    lines=[f"RoadGuard {data['kind']} report",f"Input: {data['name']}",f"Status: {data['status']}",f"Model revision: {data.get('model_revision','unknown')}",f"Device: {data.get('device','unknown')} | Runtime: {data.get('runtime_s','unknown')} seconds",f"Input SHA-256: {data.get('input_hash','unknown')}",f"Weights SHA-256: {data.get('weights_sha256','unknown')}",f"Settings: {json.dumps(data.get('processing_settings',{}),separators=(',',':'))[:120]}",f"Observations: {len(data.get('observations',[]))}"]
    lines += [f"{x.get('timestamp_s',0):.2f}s {x.get('class_name',x.get('type','mask'))} conf={x.get('confidence','n/a')} review={x.get('review','pending')}" for x in data.get('observations',[])[:30]]
    esc=lambda s:str(s).encode('ascii','replace').decode().replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
    content='BT /F1 10 Tf 48 790 Td 14 TL '+' '.join(f'({esc(x)}) Tj T*' for x in lines)+' ET'
    objs=[b'<< /Type /Catalog /Pages 2 0 R >>',b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',b'<< /Length '+str(len(content.encode())).encode()+b' >>\nstream\n'+content.encode()+b'\nendstream']
    buf=bytearray(b'%PDF-1.4\n'); offs=[0]
    for i,o in enumerate(objs,1): offs.append(len(buf)); buf+=f'{i} 0 obj\n'.encode()+o+b'\nendobj\n'
    xref=len(buf); buf+=f'xref\n0 {len(offs)}\n0000000000 65535 f \n'.encode()+b''.join(f'{n:010d} 00000 n \n'.encode() for n in offs[1:]); buf+=f'trailer << /Size {len(offs)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF'.encode(); Path(path).write_bytes(buf)

def _process(j):
    import cv2
    row=getjob(j); folder=DATA/j; src=folder/('input'+Path(row['name']).suffix.lower()); settings=row['settings']; is_video=src.suffix.lower() in VIDEOS
    start=time.monotonic(); update(j,status='processing',progress=0)
    try:
        cap=cv2.VideoCapture(str(src)) if is_video else None
        if is_video and not cap.isOpened(): raise ValueError('Could not decode uploaded video')
        fps=cap.get(cv2.CAP_PROP_FPS) if cap else None; total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap else 1
        if cap and (not fps or fps<1 or fps>240): raise ValueError('Source video has invalid frame timing')
        duration=total/fps if cap else 0
        if cap and duration>300: raise ValueError('Video exceeds the 5 minute analysis limit')
        kind=row['kind']; observations=[]; sampled=[]; writer=None; writer_count=0; previous_vis=None; previous_output_stamp=None; frame_index=0; last_sample=-1e9
        device='cuda:0' if __import__('torch').cuda.is_available() else 'cpu'
        with _model_guard:
            if kind=='road_damage':
                from .road_damage.adapter import load_model,predict,WEIGHTS,REVISION
                model=load_model(); names=model.names; revision=REVISION
            else:
                from .road_lanes.adapter import load_model,infer,overlay,WEIGHTS,REPO_REVISION
                model=load_model(device); revision=REPO_REVISION
            model_version=revision
            weight_hash=hashlib.sha256()
            with WEIGHTS.open('rb') as weight_file:
                for chunk in iter(lambda:weight_file.read(4*1024*1024),b''): weight_hash.update(chunk)
            checksum=weight_hash.hexdigest()
            with db() as c: c.execute('INSERT OR REPLACE INTO module_model_status VALUES(?,?,?,?,?,?)',(kind,'loaded successfully',None,model_version,checksum,time.time()))
            while True:
                if cap:
                    ok,frame=cap.read()
                    if not ok: break
                    stamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    if frame_index and stamp<=0: stamp=frame_index/fps
                    frame_index+=1
                    if stamp-last_sample+1e-6<float(settings['sample_interval_s']):
                        continue
                else:
                    frame=cv2.imread(str(src))
                    if frame is None: raise ValueError('Unsupported or corrupt image')
                    stamp=0.0
                if max(frame.shape[:2])>1920:
                    scale=1920/max(frame.shape[:2]); frame=cv2.resize(frame,(int(frame.shape[1]*scale),int(frame.shape[0]*scale)))
                last_sample=stamp; sampled.append(stamp)
                if kind=='road_damage':
                    imgsz = int(settings.get('imgsz') or 640)
                    tiled = bool(settings.get('tiled_inference', False))
                    per_class_conf = settings.get('per_class_confidence')
                    raw_result = predict(model, frame, confidence=settings['confidence'], iou=settings['iou'], device=device, classes=settings.get('classes'), imgsz=imgsz, tiled=tiled, per_class_conf=per_class_conf)
                    # Support both list of dicts (from adapter.predict) and Ultralytics/mock result objects
                    detections = []
                    if isinstance(raw_result, list):
                        for d in raw_result:
                            cid = d['class_id']
                            detections.append({
                                'id': uuid.uuid4().hex[:10],
                                'timestamp_s': round(stamp, 3),
                                'class_id': cid,
                                'class_name': str(names.get(cid, cid)),
                                'confidence': round(float(d['confidence']), 5),
                                'bbox': [round(float(v), 2) for v in d['bbox']],
                                '_shape': frame.shape[:2]
                            })
                    elif hasattr(raw_result, 'boxes') and raw_result.boxes is not None and len(raw_result.boxes):
                        boxes = raw_result.boxes
                        xy = boxes.xyxy.cpu().tolist() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                        cls = boxes.cls.cpu().tolist() if hasattr(boxes.cls, 'cpu') else boxes.cls
                        conf = boxes.conf.cpu().tolist() if hasattr(boxes.conf, 'cpu') else boxes.conf
                        for b, c, s in zip(xy, cls, conf):
                            cid = int(c)
                            detections.append({
                                'id': uuid.uuid4().hex[:10],
                                'timestamp_s': round(stamp, 3),
                                'class_id': cid,
                                'class_name': str(names.get(cid, cid)),
                                'confidence': round(float(s), 5),
                                'bbox': [round(float(v), 2) for v in b],
                                '_shape': frame.shape[:2]
                            })
                    observations.extend(detections)
                    from .road_damage.service import draw
                    vis = draw(frame, detections)
                    for detection in detections:
                        evidence = f"evidence_{detection['id']}.jpg"
                        cv2.imwrite(str(folder/evidence), vis)
                        detection['evidence'] = f'/media/{j}/{evidence}'
                        detection['thumbnail'] = detection['evidence']
                else:
                    import numpy as np
                    from .road_lanes.adapter import infer,overlay
                    da,ll=infer(model,frame,device); vis=overlay(frame,da,ll,settings['opacity'])
                    name=f'mask_{stamp:09.3f}'.replace('.','_'); cv2.imwrite(str(folder/f'{name}_drivable.png'),da*255); cv2.imwrite(str(folder/f'{name}_lanes.png'),ll*255)
                    da_rgba=np.zeros((*da.shape,4),dtype='uint8'); da_rgba[da==1]=[35,210,80,255]
                    ll_rgba=np.zeros((*ll.shape,4),dtype='uint8'); ll_rgba[ll>0]=[235,55,210,255]
                    cv2.imwrite(str(folder/f'{name}_drivable_overlay.png'),da_rgba); cv2.imwrite(str(folder/f'{name}_lane_overlay.png'),ll_rgba); cv2.imwrite(str(folder/'source_preview.jpg'),frame)
                    observations.append({'id':f'{frame_index:08d}','timestamp_s':round(stamp,3),'type':'segmentation_masks','drivable_pixels':int((da==1).sum()),'lane_pixels':int((ll>0).sum()),'source_width':int(frame.shape[1]),'source_height':int(frame.shape[0]),'drivable_mask':f'/media/{j}/{name}_drivable.png','lane_mask':f'/media/{j}/{name}_lanes.png','drivable_overlay':f'/media/{j}/{name}_drivable_overlay.png','lane_overlay':f'/media/{j}/{name}_lane_overlay.png'})
                if writer is None and cap:
                    h,w=frame.shape[:2]; writer=cv2.VideoWriter(str(folder/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                    if not writer.isOpened(): raise RuntimeError('Could not create annotated video writer')
                if writer is not None:
                    if previous_vis is not None:
                        repeats=max(0,round((stamp-previous_output_stamp)*fps)-1)
                        for _ in range(repeats): writer.write(previous_vis); writer_count+=1
                    writer.write(vis); writer_count+=1; previous_vis=vis.copy(); previous_output_stamp=stamp
                if len(sampled)==1: cv2.imwrite(str(folder/'preview.jpg'),vis)
                if cap: update(j,progress=min(.98,frame_index/max(total,1)))
                else: update(j,progress=.9)
                if not cap: break  # A still image is one observation, not a repeated video frame.
        if cap: cap.release()
        if writer:
            while writer_count<total and previous_vis is not None: writer.write(previous_vis); writer_count+=1
            writer.release()
        if kind=='road_damage':
            from .road_damage.service import associate
            # Associate same-class detections conservatively; keep frame observations distinct.
            associate(observations)
            detections=observations
            for x in detections: x.pop('_shape',None)
            cluster_count = len({x.get('candidate_id') for x in detections if x.get('candidate_id')})
        else:
            detections=[]
            cluster_count = None
        result={'kind':kind,'name':row['name'],'status':'completed','input_hash':row['input_hash'],'model_revision':model_version,'weights_path':str(WEIGHTS),'weights_sha256':checksum,'device':device,'runtime_s':round(time.monotonic()-start,3),'processing_settings':settings,'sampled_frames':len(sampled),'sample_timestamps_s':sampled,'observation_coverage_s':duration if cap else None,'class_map':{str(k):str(v) for k,v in names.items()} if kind=='road_damage' else None,'observations':detections if kind=='road_damage' else observations,'raw_frame_detections_count':len(detections) if kind=='road_damage' else len(observations),'unique_candidate_clusters':cluster_count,'cluster_count_warning':'Candidates are conservative short-window image associations and are not globally unique pothole/defect counts.' if kind=='road_damage' else None,'playback':f'/media/{j}/annotated.mp4' if is_video else f'/media/{j}/preview.jpg','preview':f'/media/{j}/preview.jpg','limitations':['No physical dimensions, depth, severity, cause, accident risk or road-wide safety inferred.','Model outputs require human inspection.'] if kind=='road_damage' else ['Forward-facing road scene use; perspective outside the tested use case may fail.','Lane output does not imply a safe lane change; drivable area is not a safety score.']}
        (folder/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8'); _write_pdf(result,folder/'report.pdf')
        update(j,status='completed',progress=1,result=json.dumps(result))
        with db() as c: c.execute('INSERT OR REPLACE INTO module_model_status VALUES(?,?,?,?,?,?)',(kind,'inference verified',None,model_version,checksum,time.time()))
    except Exception as e:
        if cap: cap.release()
        if writer: writer.release()
        update(j,status='failed',progress=0,error=f'{type(e).__name__}: {e}')
        with db() as c: c.execute('INSERT OR REPLACE INTO module_model_status VALUES(?,?,?,?,?,?)',(row['kind'],'failed',f'{type(e).__name__}: {e}',None,None,time.time()))

async def _upload(kind,file,settings):
    suffix=Path(file.filename or '').suffix.lower()
    if suffix not in IMAGES|VIDEOS: raise HTTPException(400,'Use JPG, PNG, WEBP or supported MP4/MOV/AVI/MKV/WEBM media')
    j=uuid.uuid4().hex; folder=DATA/j; folder.mkdir(parents=True,exist_ok=False); path=folder/('input'+suffix); size=0; digest=hashlib.sha256()
    try:
        with path.open('wb') as out:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>MAX_BYTES: raise HTTPException(413,'Upload limit is 200 MB')
                digest.update(chunk); out.write(chunk)
        if not size: raise HTTPException(400,'Empty input file')
        with db() as c: c.execute('INSERT INTO module_jobs VALUES(?,?,?,?,?,?,?,?,?,?)',(j,kind,file.filename,'queued',0,None,None,time.time(),digest.hexdigest(),settings.model_dump_json()))
    except Exception:
        path.unlink(missing_ok=True); folder.rmdir(); raise
    pool.submit(_process,j); return {'id':j,'kind':kind,'status':'queued','input_hash':digest.hexdigest()}

@router.post('/road-damage/jobs',status_code=202)
async def damage_upload(file:UploadFile=File(...),settings:str=Form('{}')):
    from .road_damage.schemas import DamageSettings
    try: cfg=DamageSettings.model_validate_json(settings)
    except Exception as e: raise HTTPException(422,str(e))
    return await _upload('road_damage',file,cfg)

@router.post('/road-lanes/jobs',status_code=202)
async def lanes_upload(file:UploadFile=File(...),settings:str=Form('{}')):
    from .road_lanes.schemas import LaneSettings
    try: cfg=LaneSettings.model_validate_json(settings)
    except Exception as e: raise HTTPException(422,str(e))
    return await _upload('road_lanes',file,cfg)

def _list_jobs(kind):
    with db() as c: ids=[r[0] for r in c.execute('SELECT id FROM module_jobs WHERE kind=? ORDER BY created DESC',(kind,))]
    return [getjob(i) for i in ids]

@router.get('/road-damage/jobs')
def list_damage_jobs(): return _list_jobs('road_damage')

@router.get('/road-lanes/jobs')
def list_lanes_jobs(): return _list_jobs('road_lanes')

@router.get('/module-jobs/{j}')
def module_job(j:str): return getjob(j)

@router.post('/module-jobs/{j}/observations/{oid}/review')
def review(j:str,oid:str,body:DamageReview):
    d=getjob(j)
    if body.status not in {'confirmed','rejected','uncertain','pending'}: raise HTTPException(422,'Review must be confirmed, rejected, uncertain, or pending')
    if not d['result'] or not any(o['id']==oid for o in d['result'].get('observations',[])): raise HTTPException(404,'Observation not found')
    with db() as c: c.execute('INSERT OR REPLACE INTO module_reviews VALUES(?,?,?,?,?)',(j,oid,body.status,body.note,time.time()))
    refreshed=getjob(j)
    if refreshed['result']:
        _write_pdf(refreshed['result'] | {'name':refreshed['name'],'status':refreshed['status']},DATA/j/'report.pdf')
        (DATA/j/'report.json').write_text(json.dumps(refreshed['result'],indent=2),encoding='utf8')
    return {'status':body.status,'note':body.note}

@router.get('/module-jobs/{j}/report.json')
def report(j:str):
    d=getjob(j)
    if not d['result']: raise HTTPException(409,'Report is not ready')
    (DATA/j/'report.json').write_text(json.dumps(d['result'],indent=2),encoding='utf8')
    return d['result']

@router.get('/module-jobs/{j}/report.pdf')
def pdf(j:str):
    d=getjob(j)
    if not d['result']: raise HTTPException(409,'Report is not ready')
    # Regenerate from persisted review states so a review is reflected in exports.
    current=d['result'] | {'name':d['name'],'status':d['status']}
    _write_pdf(current,DATA/j/'report.pdf')
    return FileResponse(DATA/j/'report.pdf',media_type='application/pdf',filename=f"{d['kind']}-{j}.pdf")

@router.get('/model-status')
def model_status():
    from .road_damage.adapter import WEIGHTS as dw,REVISION as dr
    from .road_lanes.adapter import WEIGHTS as lw,REPO_REVISION as lr
    def state(key,path,revision):
        with db() as c: row=c.execute('SELECT * FROM module_model_status WHERE model_key=?',(key,)).fetchone()
        return {'revision':row['revision'] or revision,'weights_path':str(path),'status':row['status'] if row else ('weights downloaded; inference not yet verified' if path.is_file() else 'not installed'),'reason':row['reason'] if row else (None if path.is_file() else f'Checkpoint not found: {path}'),'checksum':row['checksum'] if row else None,'size_bytes':path.stat().st_size if path.is_file() else None}
    return {'collision_detector':{'status':'existing independent pipeline','lazy_loaded':True},'road_damage':state('road_damage',dw,dr),'road_lanes':state('road_lanes',lw,lr)}
