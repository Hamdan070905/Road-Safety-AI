import json, os, sqlite3, threading, uuid, shutil, subprocess, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .engine import approach
from .calibration import estimate_homography, transform_point
from .calibration import inside_polygon
from .trajectory import process_tracks
from .conflict import analyze_pair, merge_events, pet_from_intervals
from .collision import detect_collision_candidates
from .movement import detect_movement_candidates

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.getenv('ROADGUARD_DATA',ROOT/'data')); DATA.mkdir(parents=True,exist_ok=True)
DB=DATA/'roadguard.sqlite'
app=FastAPI(title='RoadGuard AI',version='0.1.0')
worker=ThreadPoolExecutor(max_workers=1)
model_lock=threading.Lock()

def connect():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row
    c.execute('PRAGMA journal_mode=WAL'); return c
with connect() as c:
    c.executescript('''CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,name TEXT,location TEXT,status TEXT,progress REAL,result TEXT,error TEXT,created REAL);
    CREATE TABLE IF NOT EXISTS reviews(job_id TEXT,event_id TEXT,status TEXT,PRIMARY KEY(job_id,event_id));
    CREATE TABLE IF NOT EXISTS review_notes(job_id TEXT NOT NULL,event_id TEXT NOT NULL,note TEXT NOT NULL DEFAULT '',updated REAL NOT NULL,PRIMARY KEY(job_id,event_id));
    CREATE TABLE IF NOT EXISTS calibrations(id INTEGER PRIMARY KEY AUTOINCREMENT,job_id TEXT NOT NULL,version INTEGER NOT NULL,points TEXT NOT NULL,homography TEXT NOT NULL,polygon TEXT NOT NULL,rms_error_m REAL NOT NULL,created REAL NOT NULL,UNIQUE(job_id,version));
    CREATE TABLE IF NOT EXISTS camera_modes(job_id TEXT PRIMARY KEY,mode TEXT NOT NULL DEFAULT 'fixed',camera_moved INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS conflict_areas(job_id TEXT NOT NULL,area_id TEXT NOT NULL,polygon_ground TEXT NOT NULL,PRIMARY KEY(job_id,area_id));
    CREATE TABLE IF NOT EXISTS observations(job_id TEXT PRIMARY KEY,recording_key TEXT NOT NULL,start_ts REAL,end_ts REAL,duplicate_of TEXT,calibration_version INTEGER,engine TEXT);
    CREATE TABLE IF NOT EXISTS counting_lines(job_id TEXT NOT NULL,line_id TEXT NOT NULL,a TEXT NOT NULL,b TEXT NOT NULL,PRIMARY KEY(job_id,line_id));
    CREATE TABLE IF NOT EXISTS passages(job_id TEXT NOT NULL,line_id TEXT NOT NULL,track_id INTEGER NOT NULL,first_ts REAL,PRIMARY KEY(job_id,line_id,track_id));''')
    c.execute('CREATE TABLE IF NOT EXISTS movement_configs(job_id TEXT PRIMARY KEY,config TEXT NOT NULL,version INTEGER NOT NULL,updated REAL NOT NULL)')
    c.execute('CREATE TABLE IF NOT EXISTS analysis_runs(job_id TEXT NOT NULL,version INTEGER NOT NULL,result TEXT NOT NULL,created REAL NOT NULL,PRIMARY KEY(job_id,version))')
    c.execute('INSERT OR IGNORE INTO analysis_runs(job_id,version,result,created) SELECT id,1,result,created FROM jobs WHERE result IS NOT NULL')
    c.execute("UPDATE jobs SET status='failed',error='Processing interrupted; upload again to retry' WHERE status IN ('queued','processing')")

def update(j,**fields):
    with connect() as c: c.execute('UPDATE jobs SET '+','.join(f'{k}=?' for k in fields)+' WHERE id=?',[*fields.values(),j])

def _tensor_list(value):
    if value is None: return []
    if hasattr(value,'cpu'): value=value.cpu()
    if hasattr(value,'tolist'): value=value.tolist()
    return value

COCO_LABELS={0:'person',1:'bicycle',2:'car',3:'motorcycle',5:'bus',7:'truck'}

def _evidence_frame(result):
    try: return result.plot(conf=False)
    except TypeError: return result.plot()

def _deduplicate_boxes(boxes, ids, classes, confidences, iou_limit=.85):
    """Drop near-identical same-class boxes before histories/events are updated."""
    keep=[]
    order=sorted(range(len(boxes)),key=lambda i:confidences[i],reverse=True)
    for i in order:
        x1,y1,x2,y2=map(float,boxes[i]); duplicate=False
        for k in keep:
            if int(classes[i])!=int(classes[k]): continue
            a1,b1,a2,b2=map(float,boxes[k]); iw=max(0,min(x2,a2)-max(x1,a1)); ih=max(0,min(y2,b2)-max(y1,b1)); inter=iw*ih
            union=max(0,x2-x1)*max(0,y2-y1)+max(0,a2-a1)*max(0,b2-b1)-inter
            if union and inter/union>=iou_limit: duplicate=True; break
        if not duplicate: keep.append(i)
    return keep

def run(j, detector=None):
    """Upload worker. A previously saved calibration selects the metric engine."""
    import cv2, hashlib
    cap=None; writer=None
    try:
        if detector is None:
            os.environ.setdefault('YOLO_CONFIG_DIR',str(DATA/'ultralytics'))
            Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True,exist_ok=True)
            from ultralytics import YOLO
            detector=YOLO(os.getenv('ROADGUARD_MODEL','yolo11n.pt'))
        update(j,status='processing',progress=0)
        folder=DATA/j; source=folder/'input.mp4'; cap=cv2.VideoCapture(str(source))
        fps=cap.get(cv2.CAP_PROP_FPS); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not fps or not 1<=fps<=240: raise ValueError('Invalid or unsupported video timing; provide a decodable video with valid FPS metadata')
        with connect() as c:
            cal=c.execute('SELECT * FROM calibrations WHERE job_id=? ORDER BY version DESC LIMIT 1',(j,)).fetchone()
            mode=c.execute('SELECT mode,camera_moved FROM camera_modes WHERE job_id=?',(j,)).fetchone()
            areas=c.execute('SELECT area_id,polygon_ground FROM conflict_areas WHERE job_id=?',(j,)).fetchall()
            movement_row=c.execute('SELECT config,version FROM movement_configs WHERE job_id=?',(j,)).fetchone()
            lines=[dict(x) for x in c.execute('SELECT line_id,a,b FROM counting_lines WHERE job_id=?',(j,)).fetchall()]
        camera_mode=mode['mode'] if mode else 'fixed'; camera_moved=bool(mode['camera_moved']) if mode else False
        calibrated=bool(cal and camera_mode=='fixed' and not camera_moved)
        H=json.loads(cal['homography']) if calibrated else None; valid_poly=json.loads(cal['polygon']) if calibrated else None
        area_polys=[(a['area_id'],json.loads(a['polygon_ground'])) for a in areas]
        movement_config=json.loads(movement_row['config']) if movement_row else {}
        if movement_row: movement_config['version']=movement_row['version']
        device=os.getenv('ROADGUARD_DEVICE','auto')
        if device=='auto':
            try:
                import torch
                device='cuda:0' if torch.cuda.is_available() else 'cpu'
            except ImportError: device='cpu'
        if str(device).startswith('mps') and os.getenv('ROADGUARD_ALLOW_UNTESTED_MPS')!='1': device='cpu'
        histories={}; collision_samples=[]; occupancy={}; events=[]; streaks={}; streak_started={}; event_states={}; frames=0; passages=set(); frame_times=[]; overlay_dir=folder/'overlay_frames'; overlay_dir.mkdir(exist_ok=True); started=time.monotonic(); first_time=None; previous_stamp=None; last_stamp=None
        tracks_file=open(folder/'tracks.jsonl','w',encoding='utf-8'); prior_scene_hist=None; scene_id=0
        try:
            while True:
                ok,frame=cap.read()
                if not ok: break
                # A large global appearance discontinuity marks a scene boundary
                # so short track fragments are never associated across cuts.
                hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV); scene_hist=cv2.calcHist([hsv],[0,1],None,[16,16],[0,180,0,256]); cv2.normalize(scene_hist,scene_hist)
                if prior_scene_hist is not None and cv2.compareHist(prior_scene_hist,scene_hist,cv2.HISTCMP_BHATTACHARYYA)>.65: scene_id+=1
                prior_scene_hist=scene_hist
                source_stamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                if first_time is not None and source_stamp-first_time>300: break
                orig_h,orig_w=frame.shape[:2]; h,w=orig_h,orig_w
                if w>960: frame=cv2.resize(frame,(960,int(h*960/w))); h,w=frame.shape[:2]
                if writer is None: writer=cv2.VideoWriter(str(folder/'annotated_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                if first_time is None: first_time=source_stamp
                stamp=source_stamp-first_time
                if frames and stamp==0: stamp=previous_stamp+1/fps; source_stamp=first_time+stamp
                if frames and stamp<=previous_stamp: raise ValueError('Video presentation timestamps are not increasing')
                previous_stamp=stamp; last_stamp=stamp
                result=detector.track(frame,persist=True,tracker='bytetrack.yaml',classes=[0,1,2,3,5,7],verbose=False,device=device)[0]
                boxes=getattr(result,'boxes',None); current=[]
                if boxes is not None and getattr(boxes,'id',None) is not None:
                    xyxy=_tensor_list(boxes.xyxy); ids=_tensor_list(boxes.id); classes=_tensor_list(boxes.cls); confs=_tensor_list(getattr(boxes,'conf',None))
                    if ids and isinstance(ids[0],list): ids=[x[0] for x in ids]
                    if classes and isinstance(classes[0],list): classes=[x[0] for x in classes]
                    if confs and isinstance(confs[0],list): confs=[x[0] for x in confs]
                    if not confs: confs=[1.0]*len(xyxy)
                    indices=_deduplicate_boxes(xyxy,ids,classes,confs)
                    for idx in indices:
                        box,tid,cls,det_conf=xyxy[idx],ids[idx],classes[idx],float(confs[idx])
                        if det_conf<.25: continue
                        x1,y1,x2,y2=map(float,box); tid=int(tid); cls=int(cls)
                        # Road contact approximation: bottom-centre, not box centroid.
                        px=((x1+x2)/2)*orig_w/w; py=y2*orig_h/h
                        point={'id':tid,'t':stamp,'scene_id':scene_id,'image_x':px,'image_y':py,'image_x_normalized':px/orig_w,'image_y_normalized':py/orig_h,'class':cls,'detector_confidence':det_conf,'x':px/orig_w,'y':py/orig_h,'bbox':[x1/orig_w,y1/orig_h,x2/orig_w,y2/orig_h],'coordinate_system':'normalized_image','units':'normalized'}
                        collision_samples.append(point)
                        if calibrated and inside_polygon([px,py],valid_poly):
                            point['x'],point['y']=transform_point(H,[px,py]); point['coordinate_system']='road_plane'; point['units']='m'
                            for aid,poly in area_polys:
                                inside=cv2.pointPolygonTest(__import__('numpy').array(poly,dtype='float32'),(float(point['x']),float(point['y'])),False)>=0
                                record=occupancy.setdefault((tid,aid),{'entry':None,'intervals':[],'last_seen':None})
                                if record['last_seen'] is not None and stamp-record['last_seen']>.75: record['entry']=None
                                if inside and record['entry'] is None: record['entry']=stamp
                                elif not inside and record['entry'] is not None:
                                    if record['last_seen'] is not None and stamp-record['last_seen']<=.75: record['intervals'].append((record['entry'],stamp))
                                    record['entry']=None
                                record['last_seen']=stamp
                        for line in lines:
                            axy=json.loads(line['a']); bxy=json.loads(line['b']); side=lambda q:(bxy[0]-axy[0])*(q[1]-axy[1])-(bxy[1]-axy[1])*(q[0]-axy[0])
                            prior=histories.get(tid,[])
                            if prior and (side([prior[-1]['image_x'],prior[-1]['image_y']])*side([px,py])<0 or abs(side([px,py]))<1e-7 and abs(side([prior[-1]['image_x'],prior[-1]['image_y']]))>1e-7):
                                with connect() as c: c.execute('INSERT OR IGNORE INTO passages VALUES(?,?,?,?)',(j,line['line_id'],tid,stamp))
                        histories.setdefault(tid,[]).append(point); histories[tid]=histories[tid][-40:]
                        tracks_file.write(json.dumps(point)+'\n'); passages.add(tid); current.append(point)
                # Count a unique passage at most once per identity and analysis.
                moved=[]
                for p in current:
                    track_samples=histories[p['id']]
                    if calibrated:
                        track_samples=[]
                        for sample in histories[p['id']]:
                            if inside_polygon([sample['image_x'],sample['image_y']],valid_poly):
                                metric_point=dict(sample); metric_point['x'],metric_point['y']=transform_point(H,[sample['image_x'],sample['image_y']]); metric_point['coordinate_system']='road_plane'; metric_point['units']='m'; track_samples.append(metric_point)
                    q=process_tracks(track_samples,max_speed=45 if calibrated else 3.0)
                    if q and q[-1]['reliable'] and abs(q[-1]['t']-stamp)<1e-6:
                        moved.append(q[-1] | {'class':p['class']})
                active=set()
                for i,a in enumerate(moved):
                    for b in moved[i+1:]:
                        pair=tuple(sorted((a['id'],b['id']))); metric=None
                        if calibrated and a['coordinate_system']==b['coordinate_system']=='road_plane':
                            metric=analyze_pair(a,b)
                        elif not calibrated:
                            # Fallback screening only; its CPA proxy is never called TTC.
                            metric=approach(a,b,horizon=2.0,radius=.045)
                            if metric: metric={'time_to_closest_approach_s':metric['closest_approach_seconds'],'normalized_clearance':metric['normalized_clearance'],'ttc_s':None,'ttc_unavailable_reason':'Uncalibrated image-space screening; physical TTC unavailable','minimum_predicted_clearance_m':None,'pet_s':None,'pet_unavailable_reason':'Requires calibrated shared conflict-area occupancy'}
                        if not metric: continue
                        active.add(pair)
                        if pair not in streaks: streak_started[pair]=stamp; streaks[pair]=0
                        streaks[pair]+=1
                        # Require sustained evidence in presentation time; a few adjacent
                        # frames alone are too sensitive to detector jitter.
                        if streaks[pair]>=3 and stamp-streak_started[pair]>=.4:
                            state=event_states.get(pair)
                            by_id={a['id']:a,b['id']:b}; ordered=[by_id[k] for k in pair]
                            if state is None:
                                eid=uuid.uuid4().hex[:8]; image=f'{eid}.jpg'
                                event={'id':eid,'type':'potential_conflict' if calibrated else 'image_space_candidate','timestamp':round(stamp,3),'start_timestamp':round(streak_started[pair],3),'peak_timestamp':round(stamp,3),'end_timestamp':round(stamp,3),'participants':list(pair),'participant_classes':[x['class'] for x in ordered],'participant_class_names':[COCO_LABELS.get(x['class'],'other') for x in ordered],'metrics':metric,'review':'pending','thumbnail':f'/media/{j}/{image}','reason':'Predicted overlap of assumed road-user footprints from measured road-contact trajectories.' if calibrated else 'Image-space paths converge in this view; perspective and scale prevent physical conflict inference.','measurement_mode':'calibrated road plane (metres)' if calibrated else 'uncalibrated image-space screening only','calibration_version':int(cal['version']) if calibrated else None,'configuration_version':'conflict-v1','measurement_quality':{'track_ages_s':[x['age_s'] for x in ordered],'track_gaps':[x['gaps'] for x in ordered],'track_detector_confidences':{str(x['id']):next((p.get('detector_confidence') for p in current if p['id']==x['id']),None) for x in ordered},'reliable_identities':True,'uncertainty':metric.get('uncertainty','Image-space geometry; no metric interpretation')}}
                                events.append(event); state={'event':event,'last_seen':stamp,'peak_clearance':metric.get('minimum_predicted_clearance_m') if calibrated else metric.get('normalized_clearance')}; event_states[pair]=state
                                cv2.imwrite(str(folder/image),_evidence_frame(result))
                            else:
                                state['last_seen']=stamp; event=state['event']; event['end_timestamp']=round(stamp,3)
                                clearance=metric.get('minimum_predicted_clearance_m') if calibrated else metric.get('normalized_clearance')
                                if clearance is not None and (state['peak_clearance'] is None or clearance<state['peak_clearance']):
                                    state['peak_clearance']=clearance; event['peak_timestamp']=round(stamp,3); event['timestamp']=round(stamp,3); event['metrics']=metric
                                    cv2.imwrite(str(folder/(event['id']+'.jpg')),_evidence_frame(result))
                                ordered=[a,b] if a['id']<b['id'] else [b,a]; event['measurement_quality']['track_ages_s']=[x['age_s'] for x in ordered]; event['measurement_quality']['track_gaps']=[x['gaps'] for x in ordered]
                for pair in list(streaks):
                    if pair not in active:
                        streaks.pop(pair,None); streak_started.pop(pair,None)
                        state=event_states.get(pair)
                        if state:
                            state['event']['end_timestamp']=round(state['last_seen'],3)
                            state['event']['end_timestamp_s']=round(state['last_seen'],3)
                            if stamp-state['last_seen']>.75: event_states.pop(pair,None)
                plotted=_evidence_frame(result); writer.write(plotted); frame_times.append(stamp); cv2.imwrite(str(overlay_dir/f'{frames:08d}.jpg'),plotted,[cv2.IMWRITE_JPEG_QUALITY,86]); frames+=1
                if frames%15==0: update(j,progress=min(.95,frames/max(total,1)))
        finally: tracks_file.close()
        cap.release(); cap=None
        if writer: writer.release(); writer=None
        if frames==0: raise ValueError('No readable video frames')
        collision_diagnostics=[]
        collision_events=detect_collision_candidates(collision_samples,moving_camera=(camera_mode!='fixed' or camera_moved),diagnostics=collision_diagnostics)
        (folder/'collision_diagnostics.json').write_text(json.dumps({'camera_mode':camera_mode,'camera_moved':camera_moved,'calibration_valid':calibrated,'engine':'calibrated-footprint-v1' if calibrated else 'legacy-image-screen-v1','collision_engine':'image-space collision screen','records':collision_diagnostics},indent=2),encoding='utf-8')
        movement_events=detect_movement_candidates(collision_samples,movement_config,calibrated=calibrated,moving_camera=(camera_mode!='fixed' or camera_moved))
        candidate_events=collision_events+movement_events
        for event in candidate_events:
            event['id']=uuid.uuid4().hex[:8]
            event['thumbnail']=f"/media/{j}/{event['id']}.jpg"
            # Mark the relevant interval in the annotated video and capture a
            # timestamped evidence still from the original source.
            for idx,t in enumerate(frame_times):
                if event['start_timestamp']<=t<=event['end_timestamp']:
                    overlay_path=overlay_dir/f'{idx:08d}.jpg'
                    overlay=cv2.imread(str(overlay_path))
                    if overlay is not None:
                        cv2.rectangle(overlay,(0,0),(min(overlay.shape[1]-1,760),38),(20,30,35),-1)
                        label='POSSIBLE COLLISION - REVIEW REQUIRED' if event['type']=='possible_collision' else event['type'].replace('_',' ').upper()
                        cv2.putText(overlay,label[:70],(12,27),cv2.FONT_HERSHEY_SIMPLEX,.62,(70,235,190),2,cv2.LINE_AA)
                        cv2.imwrite(str(overlay_path),overlay,[cv2.IMWRITE_JPEG_QUALITY,90])
            evidence=cv2.VideoCapture(str(source)); evidence.set(cv2.CAP_PROP_POS_MSEC,max(0,first_time+event['timestamp'])*1000); ok,frame=evidence.read(); evidence.release()
            if ok:
                for point in collision_samples:
                    if abs(point['t']-event['timestamp'])<=.05 and point['id'] in event['participants']:
                        x1,y1,x2,y2=point['bbox']; h0,w0=frame.shape[:2]
                        cv2.rectangle(frame,(int(x1*w0),int(y1*h0)),(int(x2*w0),int(y2*h0)),(40,220,190),3)
                cv2.rectangle(frame,(0,0),(min(frame.shape[1]-1,800),42),(20,30,35),-1)
                label='POSSIBLE COLLISION - REVIEW REQUIRED' if event['type']=='possible_collision' else event['type'].replace('_',' ').upper()
                cv2.putText(frame,label[:70],(12,29),cv2.FONT_HERSHEY_SIMPLEX,.62,(70,235,190),2,cv2.LINE_AA)
                cv2.imwrite(str(folder/f"{event['id']}.jpg"),frame)
        events.extend(candidate_events)
        # PET only when configured ground conflict area has observed, sequential occupancy.
        if calibrated and area_polys:
            for e in events:
                intervals=[]
                for aid,_poly in area_polys:
                    per_id=[occupancy.get((tid,aid),{}).get('intervals',[]) for tid in e['participants']]
                    pairs=[]
                    for left in per_id[0]:
                        for right in per_id[1]:
                            if left[1] <= right[0]: pairs.append((right[0]-left[1],left[1],right[0]))
                            if right[1] <= left[0]: pairs.append((left[0]-right[1],right[1],left[0]))
                    if pairs: intervals.append((min(pairs),aid))
                if intervals:
                    (pet,exit_t,entry_t),aid=min(intervals); e['metrics']['pet_s']=pet; e['metrics']['pet_unavailable_reason']=None; e['metrics']['pet_conflict_area_id']=aid; e['metrics']['pet_occupancy_interval_s']={'first_exit':exit_t,'next_entry':entry_t}
        events=merge_events(events,window_s=3.0)
        playback=f'/media/{j}/annotated_raw.mp4'
        if shutil.which('ffmpeg'):
            out=folder/'annotated.mp4'; concat=folder/'overlay_frames.txt'
            with open(concat,'w',encoding='utf-8') as f:
                for idx,t in enumerate(frame_times):
                    img=(overlay_dir/f'{idx:08d}.jpg').resolve().as_posix().replace("'","'\\\\''")
                    f.write(f"file '{img}'\n")
                    if idx+1<len(frame_times): f.write(f"duration {max(.001,frame_times[idx+1]-t):.6f}\n")
                last=(overlay_dir/f'{len(frame_times)-1:08d}.jpg').resolve().as_posix().replace("'","'\\\\''")
                f.write(f"file '{last}'\n")
            p=subprocess.run(['ffmpeg','-y','-f','concat','-safe','0','-i',str(concat),'-fps_mode','vfr','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],capture_output=True)
            if p.returncode==0: playback=f'/media/{j}/annotated.mp4'
            shutil.rmtree(overlay_dir,ignore_errors=True)
            for e in events:
                clip=folder/(e['id']+'.mp4'); start=max(0,e['start_timestamp']-3); dur=max(1,e['end_timestamp']-start+4)
                clip_source=folder/'annotated.mp4' if playback.endswith('annotated.mp4') else folder/'annotated_raw.mp4'
                p=subprocess.run(['ffmpeg','-y','-ss',str(start),'-i',str(clip_source),'-t',str(dur),'-c:v','libx264','-pix_fmt','yuv420p',str(clip)],capture_output=True)
                if p.returncode==0: e['clip']=f'/media/{j}/{clip.name}'
        if overlay_dir.exists(): shutil.rmtree(overlay_dir,ignore_errors=True)
        hasher=hashlib.sha256()
        with open(source,'rb') as inp:
            for chunk in iter(lambda:inp.read(1024*1024),b''): hasher.update(chunk)
        digest=hasher.hexdigest()
        with connect() as c:
            duplicate=c.execute('SELECT job_id FROM observations WHERE recording_key=? AND job_id<>? LIMIT 1',(digest,j)).fetchone()
            c.execute('INSERT OR REPLACE INTO observations VALUES(?,?,?,?,?,?,?)',(j,digest,first_time,first_time+last_stamp,duplicate['job_id'] if duplicate else None,int(cal['version']) if calibrated else None,'calibrated-v1' if calibrated else 'image-fallback-v1'))
        with connect() as c: counted=c.execute('SELECT COUNT(DISTINCT track_id) FROM passages WHERE job_id=?',(j,)).fetchone()[0]
        with connect() as c: analysis_version=c.execute('SELECT COALESCE(MAX(version),0)+1 FROM analysis_runs WHERE job_id=?',(j,)).fetchone()[0]
        result={'analysis_version':analysis_version,'events':events,'frames':frames,'duration':max(0,last_stamp) if last_stamp is not None else 0,'timeline_origin_source_timestamp_s':first_time,'start_timestamp_s':first_time,'end_timestamp_s':first_time+last_stamp if last_stamp is not None else first_time,'tracks':len(passages),'unique_passages':counted if lines else None,'runtime':round(time.monotonic()-started,2),'model':os.getenv('ROADGUARD_MODEL','yolo11n.pt'),'device':device,'playback':playback,'collision_diagnostics':f'/media/{j}/collision_diagnostics.json','measurement_mode':'calibrated road plane (metres)' if calibrated else 'uncalibrated image-space fallback','engine':'calibrated-footprint-v1' if calibrated else 'legacy-image-screen-v1','camera_mode':camera_mode,'calibration_version':int(cal['version']) if calibrated else None,'configuration_version':'conflict-v1','movement_configuration_version':int(movement_row['version']) if movement_row else 0,'observation_interval':{'start':first_time,'end':first_time+last_stamp,'duration_s':max(0,last_stamp)},'duplicate_recording_of':duplicate['job_id'] if duplicate else None,'timestamp_aligned_overlay':playback.endswith('annotated.mp4'),'limitations':['Planar road homography; lens distortion is not corrected.','Calibration residual is not independent ground-truth validation.','PET is reported only from observed sequential occupancy in configured shared conflict areas.','Variable-frame-rate output uses presentation timestamp deltas through FFmpeg when available; otherwise an explicitly untimed CFR preview is used.']}
        result_json=json.dumps(result)
        with connect() as c:
            c.execute('INSERT INTO analysis_runs VALUES(?,?,?,?)',(j,analysis_version,result_json,time.time()))
            c.execute('UPDATE jobs SET status=?,progress=?,result=?,error=NULL WHERE id=?',('completed',1,result_json,j))
        (folder/'report.json').write_text(json.dumps(result,indent=2))
    except Exception as e: update(j,status='failed',error=f'{type(e).__name__}: {e}')
    finally:
        if cap: cap.release()
        if writer: writer.release()

def getjob(j):
    with connect() as c:
        row=c.execute('SELECT * FROM jobs WHERE id=?',(j,)).fetchone()
        if row is None: raise HTTPException(404,'Job not found')
        d=dict(row); d['result']=json.loads(d['result']) if d['result'] else None
        if d['result']:
            reviews={r['event_id']:r['status'] for r in c.execute('SELECT * FROM reviews WHERE job_id=?',(j,))}
            notes={r['event_id']:r['note'] for r in c.execute('SELECT * FROM review_notes WHERE job_id=?',(j,))}
            for e in d['result']['events']:
                e['review']=reviews.get(e['id'],'pending'); e['review_note']=notes.get(e['id'],'')
        return d

@app.get('/api/health')
def health():
    import importlib.util
    try:
        import torch
        device='cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError: device='cpu'
    return {'status':'ok','detector_installed':importlib.util.find_spec('ultralytics') is not None,'model':os.getenv('ROADGUARD_MODEL','yolo11n.pt'),'device':device,'mode':'calibrated footprint engine when a valid fixed-camera calibration is selected; otherwise labeled image-space screening','collision_screening':'possible-collision candidates from temporal image-space evidence; review required','movement_risks':'wrong-way, stopped vehicle, blocked crossing, speeding, and sudden deceleration are configurable; metric speed requires valid fixed-camera calibration','moving_camera':'experimental; metric measurements unavailable','road_damage':'not integrated','lane_segmentation':'not integrated'}

@app.get('/api/jobs')
def jobs():
    with connect() as c: ids=[r[0] for r in c.execute('SELECT id FROM jobs ORDER BY created DESC')]
    return [getjob(j) for j in ids]

@app.get('/api/jobs/{j}')
def job(j:str): return getjob(j)

@app.post('/api/jobs',status_code=202)
async def upload(file:UploadFile=File(...),location:str=Form('Unassigned'),camera_mode:str=Form('fixed')):
    if camera_mode not in ('fixed','moving'): raise HTTPException(400,'Camera mode must be fixed or moving')
    if Path(file.filename or '').suffix.lower() not in ['.mp4','.mov','.avi','.mkv','.webm']: raise HTTPException(400,'Choose a video file')
    j=uuid.uuid4().hex; folder=DATA/j; folder.mkdir(); size=0
    try:
        with open(folder/'input.mp4','wb') as f:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>200*1024*1024: raise HTTPException(413,'Maximum upload is 200 MB')
                f.write(chunk)
        if not size: raise HTTPException(400,'Empty file')
    except Exception:
        shutil.rmtree(folder); raise
    with connect() as c:
        c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)',(j,file.filename,location[:120],'queued',0,None,None,time.time()))
        c.execute('INSERT INTO camera_modes(job_id,mode) VALUES(?,?)',(j,camera_mode))
    worker.submit(run,j); return {'id':j}

class Review(BaseModel):
    status:str
    note:str=''
@app.post('/api/jobs/{j}/events/{eid}/review')
def review(j:str,eid:str,r:Review):
    d=getjob(j)
    if r.status not in ['confirmed','rejected','uncertain','pending']: raise HTTPException(400,'Invalid review status')
    found=bool(d['result'] and any(e['id']==eid for e in d['result']['events']))
    if not found:
        with connect() as c: versions=c.execute('SELECT result FROM analysis_runs WHERE job_id=?',(j,)).fetchall()
        found=any(any(e['id']==eid for e in json.loads(row['result']).get('events',[])) for row in versions)
    if not found: raise HTTPException(404,'Event not found')
    with connect() as c:
        c.execute('INSERT OR REPLACE INTO reviews VALUES(?,?,?)',(j,eid,r.status))
        c.execute('INSERT OR REPLACE INTO review_notes VALUES(?,?,?,?)',(j,eid,r.note[:2000],time.time()))
    return {'status':r.status,'note':r.note[:2000]}

class Calibration(BaseModel):
    image_points: list[list[float]]
    ground_points_m: list[list[float]]
    valid_polygon_image: list[list[float]]

@app.get('/api/jobs/{j}/frame')
def representative_frame(j:str,timestamp:float=0):
    import cv2
    getjob(j); path=DATA/j/'input.mp4'; cap=cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC,max(0,timestamp)*1000); ok,frame=cap.read()
        if not ok: raise HTTPException(422,'Unable to decode requested source frame')
        ok,encoded=cv2.imencode('.jpg',frame)
        if not ok: raise HTTPException(500,'Frame encoding failed')
        from fastapi.responses import Response
        return Response(encoded.tobytes(),media_type='image/jpeg',headers={'X-Source-Timestamp':str(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)})
    finally: cap.release()

@app.get('/api/jobs/{j}/calibrations')
def calibrations(j:str):
    getjob(j)
    with connect() as c: rows=c.execute('SELECT * FROM calibrations WHERE job_id=? ORDER BY version',(j,)).fetchall()
    return [dict(r) | {'points':json.loads(r['points']),'homography':json.loads(r['homography']),'polygon':json.loads(r['polygon'])} for r in rows]

@app.post('/api/jobs/{j}/calibrations')
def save_calibration(j:str,body:Calibration):
    getjob(j)
    if len(body.valid_polygon_image)<3: raise HTTPException(400,'Mark at least three points for the valid road analysis polygon')
    try: fit=estimate_homography(body.image_points,body.ground_points_m)
    except ValueError as e: raise HTTPException(422,str(e))
    import cv2, numpy as np
    cap=cv2.VideoCapture(str(DATA/j/'input.mp4'))
    if not cap.isOpened(): raise HTTPException(422,'Source video frame cannot be opened for validation')
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    polygon=np.asarray(body.valid_polygon_image,dtype=np.float32)
    if polygon.ndim!=2 or polygon.shape[1]!=2 or not np.isfinite(polygon).all(): raise HTTPException(422,'Valid-region points must be finite image pixel pairs')
    if abs(cv2.contourArea(polygon))<100: raise HTTPException(422,'Valid analysis region is degenerate or too small')
    if np.any(polygon[:,0]<0) or np.any(polygon[:,0]>=width) or np.any(polygon[:,1]<0) or np.any(polygon[:,1]>=height): raise HTTPException(422,'Valid analysis region must lie inside the source frame')
    if any(cv2.pointPolygonTest(polygon,tuple(map(float,p)),False)<0 for p in body.image_points): raise HTTPException(422,'Calibration points must lie inside the valid analysis region')
    spread=cv2.contourArea(cv2.convexHull(np.asarray(body.image_points,dtype=np.float32)))
    if spread<abs(cv2.contourArea(polygon))*.05: raise HTTPException(422,'Calibration points are too clustered; distribute them across the marked analysis region')
    with connect() as c:
        version=c.execute('SELECT COALESCE(MAX(version),0)+1 FROM calibrations WHERE job_id=?',(j,)).fetchone()[0]
        c.execute('INSERT INTO calibrations(job_id,version,points,homography,polygon,rms_error_m,created) VALUES(?,?,?,?,?,?,?)',(j,version,json.dumps({'image':body.image_points,'ground_m':body.ground_points_m}),json.dumps(fit['homography']),json.dumps(body.valid_polygon_image),fit['rms_error_m'],time.time()))
    return {'version':version,'homography':fit['homography'],'rms_error_m':fit['rms_error_m'],'units':'metres','limitations':['Planar-road approximation; lens distortion is not corrected.','Calibration is user-measured; report error is training-point residual, not independent validation.','Measurements outside the marked valid image polygon must be withheld.']}

class ConflictArea(BaseModel): area_id:str; polygon_ground_m:list[list[float]]
@app.put('/api/jobs/{j}/conflict-areas/{area_id}')
def configure_conflict_area(j:str,area_id:str,body:ConflictArea):
    getjob(j)
    if len(body.polygon_ground_m)<3: raise HTTPException(422,'Conflict area needs at least three measured ground points')
    with connect() as c: c.execute('INSERT OR REPLACE INTO conflict_areas VALUES(?,?,?)',(j,area_id,json.dumps(body.polygon_ground_m)))
    return {'area_id':area_id,'units':'metres','status':'saved; PET will use observed entry/exit occupancy only'}

class CountLine(BaseModel): a_image:list[float]; b_image:list[float]
@app.put('/api/jobs/{j}/counting-lines/{line_id}')
def configure_counting_line(j:str,line_id:str,body:CountLine):
    getjob(j)
    if body.a_image==body.b_image: raise HTTPException(422,'Counting line endpoints must differ')
    with connect() as c: c.execute('INSERT OR REPLACE INTO counting_lines VALUES(?,?,?,?)',(j,line_id,json.dumps(body.a_image),json.dumps(body.b_image)))
    return {'line_id':line_id,'units':'source-image pixels','status':'saved; reanalysis required to count crossings'}

@app.post('/api/jobs/{j}/analyze',status_code=202)
def reanalyze(j:str):
    getjob(j)
    with connect() as c: running=c.execute("SELECT 1 FROM jobs WHERE id=? AND status IN ('queued','processing')",(j,)).fetchone()
    if running: raise HTTPException(409,'Job is already processing')
    worker.submit(run,j); return {'id':j,'status':'queued'}

class CameraMode(BaseModel): mode:str; camera_moved:bool=False
@app.put('/api/jobs/{j}/camera-mode')
def set_camera_mode(j:str,body:CameraMode):
    getjob(j)
    if body.mode not in ('fixed','moving'): raise HTTPException(400,'mode must be fixed or moving')
    with connect() as c: c.execute('INSERT OR REPLACE INTO camera_modes VALUES(?,?,?)',(j,body.mode,int(body.camera_moved)))
    with connect() as c: has_cal=bool(c.execute('SELECT 1 FROM calibrations WHERE job_id=?',(j,)).fetchone())
    return {'mode':body.mode,'camera_moved':body.camera_moved,'metric_measurements_available':has_cal and body.mode=='fixed' and not body.camera_moved,'message':'Moving-camera analysis is experimental. Fixed-ground geometry is invalidated after camera movement.' if body.mode=='moving' or body.camera_moved else 'Calibration must be reviewed after any camera movement.'}

class MovementConfiguration(BaseModel): config:dict

@app.get('/api/jobs/{j}/movement-config')
def get_movement_config(j:str):
    getjob(j)
    with connect() as c: row=c.execute('SELECT config,version,updated FROM movement_configs WHERE job_id=?',(j,)).fetchone()
    return {'version':row['version'],'updated':row['updated'],'config':json.loads(row['config'])} if row else {'version':0,'config':{}}

@app.put('/api/jobs/{j}/movement-config')
def save_movement_config(j:str,body:MovementConfiguration):
    getjob(j); cfg=body.config
    allowed={'wrong_way','stopped_vehicle','blocked_crossing','speeding','sudden_deceleration'}
    if set(cfg)-allowed: raise HTTPException(422,'Unsupported movement-risk configuration key')
    def polygon_ok(poly):
        return isinstance(poly,list) and len(poly)>=3 and all(isinstance(p,list) and len(p)==2 and all(isinstance(x,(int,float)) and 0<=x<=1 for x in p) for p in poly)
    for name in ('stopped_vehicle','blocked_crossing'):
        option=cfg.get(name) or {}
        if option.get('enabled') and not polygon_ok(option.get('zone')): raise HTTPException(422,f'{name} requires a normalized image-space zone polygon')
        if name=='stopped_vehicle' and any(not polygon_ok(p) for p in option.get('queue_exclusion_zones',[])): raise HTTPException(422,'Every queue exclusion zone must be a normalized polygon')
        if option.get('enabled') and float(option.get('dwell_s',10 if name=='stopped_vehicle' else 2))<=0: raise HTTPException(422,'Dwell thresholds must be positive')
    wrong=cfg.get('wrong_way') or {}
    if wrong.get('enabled'):
        dirs=wrong.get('permitted_directions')
        if not isinstance(dirs,list) or not dirs or any(not isinstance(d,list) or len(d)!=2 or not all(isinstance(v,(int,float)) for v in d) for d in dirs): raise HTTPException(422,'Wrong-way screening needs one or more permitted direction vectors')
        if any(not polygon_ok(p) for p in wrong.get('permitted_turn_zones',[])): raise HTTPException(422,'Permitted-turn zones must be normalized polygons')
    for name in ('speeding','sudden_deceleration'):
        option=cfg.get(name) or {}
        key='speed_limit_mps' if name=='speeding' else 'threshold_mps2'
        if option.get('enabled') and (not isinstance(option.get(key),(int,float)) or option[key]<=0): raise HTTPException(422,f'{name} requires a positive {key}')
    with connect() as c:
        version=c.execute('SELECT COALESCE(MAX(version),0)+1 FROM movement_configs WHERE job_id=?',(j,)).fetchone()[0]
        c.execute('INSERT OR REPLACE INTO movement_configs VALUES(?,?,?,?)',(j,json.dumps(cfg),version,time.time()))
    return {'version':version,'config':cfg,'status':'saved; reanalysis required','image_zones':'normalized coordinates','speed_and_deceleration':'require valid fixed-camera calibration'}

@app.get('/api/locations')
def location_analytics():
    with connect() as c: rows=c.execute('SELECT j.id,j.location,j.result,o.duplicate_of FROM jobs j LEFT JOIN observations o ON o.job_id=j.id WHERE j.status=\'completed\'').fetchall()
    groups={}
    for row in rows:
        result=json.loads(row['result'] or '{}'); g=groups.setdefault(row['location'] or 'Unassigned',{'jobs':0,'duration_s':0,'confirmed_events':0,'candidate_events':0,'possible_collision_candidates':0,'reviewer_confirmed_collisions':0,'unique_passages':0,'counted_jobs':0,'duplicate_recordings_excluded':0})
        g['jobs']+=1
        if row['duplicate_of']: g['duplicate_recordings_excluded']+=1; continue
        g['duration_s']+=result.get('duration',0); g['candidate_events']+=len(result.get('events',[]))
        with connect() as c: reviews={x['event_id']:x['status'] for x in c.execute('SELECT event_id,status FROM reviews WHERE job_id=?',(row['id'],)).fetchall()}
        g['confirmed_events']+=sum(reviews.get(e['id'],e.get('review'))=='confirmed' for e in result.get('events',[]))
        g['possible_collision_candidates']+=sum(e.get('type')=='possible_collision' for e in result.get('events',[]))
        g['reviewer_confirmed_collisions']+=sum(e.get('type')=='possible_collision' and reviews.get(e['id'],e.get('review'))=='confirmed' for e in result.get('events',[]))
        with connect() as c:
            n=c.execute('SELECT COUNT(DISTINCT track_id) FROM passages WHERE job_id=?',(row['id'],)).fetchone()[0]
            if n: g['unique_passages']+=n; g['counted_jobs']+=1
    for g in groups.values():
        duration=g['duration_s']; g['coverage']='adequate' if duration>=600 else 'limited (<10 observed minutes)'
        g['confirmed_events_per_hour']=g['confirmed_events']/(duration/3600) if duration>=600 else None
        g['conflicts_per_1000_road_users']=g['confirmed_events']/g['unique_passages']*1000 if g['unique_passages']>=100 else None
        g['rate_unavailable_reason']='At least ten unique, nonduplicate observed minutes required' if duration<600 else None
        g['passage_rate_unavailable_reason']=None if g['unique_passages']>=100 else 'At least 100 unique line-counted passages required' if g['counted_jobs'] else 'No configured counting line produced passage counts'
    return groups

@app.get('/api/insights')
def deterministic_insights():
    with connect() as c: rows=c.execute('SELECT j.id,j.location,j.result FROM jobs j WHERE j.status=\'completed\'').fetchall()
    out=[]
    for row in rows:
        result=json.loads(row['result'] or '{}')
        with connect() as c: reviews={x['event_id']:x['status'] for x in c.execute('SELECT event_id,status FROM reviews WHERE job_id=?',(row['id'],)).fetchall()}
        reviewed=[e for e in result.get('events',[]) if reviews.get(e['id'],e.get('review'))=='confirmed']
        if reviewed: out.append({'location':row['location'],'observation_period_s':result.get('duration'),'coverage':'limited; single recording unless combined across unique recordings','pattern':'Reviewed potential conflict interactions were recorded.','supporting_events':[e['id'] for e in reviewed],'measurements':[e.get('metrics',{}) for e in reviewed],'uncertainty':'Human review does not establish causation or driver fault.','suggestion':'Inspect the cited clips and consider an on-site road-safety review.'})
    return out

@app.get('/api/jobs/{j}/report')
def report(j:str): return getjob(j)

@app.get('/api/jobs/{j}/report.pdf')
def report_pdf(j:str):
    """Create a compact, dependency-free PDF from the current stored report."""
    from fastapi.responses import Response
    d=getjob(j); result=d.get('result')
    if not result: raise HTTPException(409,'Report is available after processing completes')
    lines=[f"RoadGuard AI report - {d['name']}",f"Location: {d['location']}",f"Engine: {result.get('engine')} | Mode: {result.get('measurement_mode')}",f"Observed duration: {result.get('duration',0):.2f} seconds",f"Candidates: {len(result.get('events',[]))}","Human review is required; this report does not establish fault or crash probability."]
    for e in result.get('events',[]):
        m=e.get('metrics') or {}; review='Collision confirmed by reviewer' if e.get('type')=='possible_collision' and e.get('review')=='confirmed' else e.get('review','pending'); lines.extend([f"{e.get('type','event')} {e['id']} | {review} | {e.get('start_timestamp',e.get('timestamp')):.2f}-{e.get('end_timestamp',e.get('timestamp')):.2f}s",f"Participants: {', '.join(map(str,e.get('participant_class_names',e.get('participants',[]))))}",f"TTC: {m.get('ttc_s') if m.get('ttc_s') is not None else m.get('ttc_unavailable_reason','unavailable')}",f"PET: {m.get('pet_s') if m.get('pet_s') is not None else m.get('pet_unavailable_reason','unavailable')}",f"Reason: {e.get('reason','')}",f"Review note: {e.get('review_note','')}"])
    def pdf_escape(s): return str(s).encode('ascii','replace').decode().replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
    content='BT /F1 10 Tf 50 790 Td 13 TL '+' '.join(f'({pdf_escape(line[:150])}) Tj T*' for line in lines[:54])+' ET'
    stream=content.encode('ascii'); objects=[b'<< /Type /Catalog /Pages 2 0 R >>',b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream']
    out=bytearray(b'%PDF-1.4\n'); offsets=[0]
    for i,obj in enumerate(objects,1): offsets.append(len(out)); out.extend(f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n')
    xref=len(out); out.extend(f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'.encode()); out.extend(b''.join(f'{n:010d} 00000 n \n'.encode() for n in offsets[1:])); out.extend(f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF'.encode())
    return Response(bytes(out),media_type='application/pdf',headers={'Content-Disposition':f'attachment; filename="roadguard-{j}.pdf"'})

@app.get('/api/jobs/{j}/analyses')
def analysis_versions(j:str):
    getjob(j)
    with connect() as c:
        rows=c.execute('SELECT version,result,created FROM analysis_runs WHERE job_id=? ORDER BY version DESC',(j,)).fetchall()
        reviews={x['event_id']:x['status'] for x in c.execute('SELECT event_id,status FROM reviews WHERE job_id=?',(j,)).fetchall()}
    out=[]
    for r in rows:
        result=json.loads(r['result'])
        for event in result.get('events',[]): event['review']=reviews.get(event['id'],'pending')
        out.append({'version':r['version'],'created':r['created'],'result':result})
    return out

@app.get('/media/{j}/{name}')
def media(j:str,name:str):
    getjob(j)
    if Path(name).name!=name or name not in ['input.mp4','annotated.mp4','annotated_raw.mp4','tracks.jsonl'] and not (DATA/j/name).is_file(): raise HTTPException(404)
    path=(DATA/j/name).resolve()
    if path.parent!=(DATA/j).resolve() or not path.is_file(): raise HTTPException(404)
    return FileResponse(path)
app.mount('/',StaticFiles(directory=ROOT/'frontend',html=True),name='frontend')
