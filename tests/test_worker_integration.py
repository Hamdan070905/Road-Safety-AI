"""Synthetic integration fixture; not a real-video accuracy evaluation."""
import json
import uuid
import numpy as np
import cv2
from fastapi.testclient import TestClient
from backend import main


class Boxes:
    def __init__(self, boxes):
        self.xyxy=[b for b,_,_ in boxes]
        self.id=[i for _,i,_ in boxes]
        self.cls=[c for _,_,c in boxes]

class Result:
    def __init__(self, frame, boxes): self.boxes=Boxes(boxes); self.frame=frame
    def plot(self): return self.frame

class SyntheticDetector:
    """Two crossing tracks with known image trajectories and fixed IDs."""
    def __init__(self): self.index=0
    def track(self, frame, **kwargs):
        i=self.index; self.index+=1
        y=60+6*i; x=90+6*i
        return [Result(frame, [([140,y-20,160,y],1,2),([x-10,100,x+10,120],2,2)])]

def make_job(calibrated=True):
    jid='synthetic-'+uuid.uuid4().hex[:10]; folder=main.DATA/jid; folder.mkdir(parents=True)
    path=folder/'input.mp4'; w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,240))
    for _ in range(24): w.write(np.zeros((240,320,3),np.uint8))
    w.release()
    with main.connect() as c:
        c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)',(jid,'synthetic-crossing.mp4','Synthetic test','queued',0,None,None,1_800_000_000))
        c.execute('INSERT INTO camera_modes VALUES(?,?,?)',(jid,'fixed',0))
        if calibrated:
            pts={'image':[[0,0],[320,0],[0,240],[320,240]],'ground_m':[[0,0],[32,0],[0,24],[32,24]]}
            h=cv2.getPerspectiveTransform(np.float32(pts['image'][:4]),np.float32(pts['ground_m'][:4])).tolist()
            c.execute('INSERT INTO calibrations(job_id,version,points,homography,polygon,rms_error_m,created) VALUES(?,?,?,?,?,?,?)',(jid,1,json.dumps(pts),json.dumps(h),json.dumps([[0,0],[319,0],[319,239],[0,239]]),0,1_800_000_000))
    return jid

def test_worker_calibration_selects_metric_engine_and_api_review_persists(monkeypatch):
    jid=make_job(True); calls=[]; original=main.analyze_pair
    def spy(a,b,**kwargs): calls.append((a['coordinate_system'],b['coordinate_system'])); return original(a,b,**kwargs)
    monkeypatch.setattr(main,'analyze_pair',spy)
    main.run(jid,detector=SyntheticDetector())
    c=TestClient(main.app); job=c.get(f'/api/jobs/{jid}').json(); result=job['result']
    assert job['status']=='completed',job.get('error')
    assert calls and all(a==b=='road_plane' for a,b in calls)
    assert result['engine']=='calibrated-footprint-v1' and result['calibration_version']==1
    assert result['events'] and result['events'][0]['metrics']['ttc_s'] is not None
    eid=result['events'][0]['id']
    assert c.post(f'/api/jobs/{jid}/events/{eid}/review',json={'status':'confirmed','note':'Observed crossing paths'}).status_code==200
    saved=c.get(f'/api/jobs/{jid}').json()['result']['events'][0]
    assert saved['review']=='confirmed' and saved['review_note']=='Observed crossing paths'
    assert c.get(f'/api/jobs/{jid}/report').json()['result']['events'][0]['metrics']['ttc_s'] is not None
    pdf=c.get(f'/api/jobs/{jid}/report.pdf')
    assert pdf.status_code==200 and pdf.content.startswith(b'%PDF-1.4')

def test_saved_calibration_api_changes_worker_engine_and_persists_metrics():
    jid=make_job(False); c=TestClient(main.app)
    response=c.post(f'/api/jobs/{jid}/calibrations',json={
        'image_points':[[10,10],[310,10],[10,230],[310,230]],
        'ground_points_m':[[0,0],[30,0],[0,22],[30,22]],
        'valid_polygon_image':[[0,0],[319,0],[319,239],[0,239]],
    })
    assert response.status_code==200,response.text
    assert response.json()['version']==1 and response.json()['units']=='metres'
    main.run(jid,detector=SyntheticDetector())
    report=c.get(f'/api/jobs/{jid}/report').json()['result']
    assert report['engine']=='calibrated-footprint-v1' and report['calibration_version']==1
    assert report['events'] and report['events'][0]['measurement_mode']=='calibrated road plane (metres)'
    assert 'ttc_s' in report['events'][0]['metrics']

def test_worker_without_calibration_uses_labeled_fallback():
    jid=make_job(False); main.run(jid,detector=SyntheticDetector())
    result=TestClient(main.app).get(f'/api/jobs/{jid}').json()['result']
    assert result['engine']=='legacy-image-screen-v1'
    assert result['measurement_mode']=='uncalibrated image-space fallback'
    assert all(e['metrics']['ttc_s'] is None for e in result['events'])

def test_counting_line_deduplicates_track_crossings():
    jid=make_job(False)
    with main.connect() as c: c.execute('INSERT INTO counting_lines VALUES(?,?,?,?)',(jid,'test-line',json.dumps([150,0]),json.dumps([150,239])))
    main.run(jid,detector=SyntheticDetector())
    with main.connect() as c: n=c.execute('SELECT COUNT(*) FROM passages WHERE job_id=? AND line_id=?',(jid,'test-line')).fetchone()[0]
    assert n==1

def test_upload_to_report_and_evidence_export(monkeypatch):
    source=main.DATA/f'synthetic-upload-{uuid.uuid4().hex[:8]}.mp4'; w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,240))
    for _ in range(24): w.write(np.zeros((240,320,3),np.uint8))
    w.release(); queued=[]
    monkeypatch.setattr(main.worker,'submit',lambda fn,jid:queued.append((fn,jid)))
    c=TestClient(main.app)
    response=c.post('/api/jobs',files={'file':('synthetic-upload.mp4',source.read_bytes(),'video/mp4')},data={'location':'Synthetic test location'})
    assert response.status_code==202
    jid=response.json()['id']; assert queued and queued[0][1]==jid
    main.run(jid,detector=SyntheticDetector())
    job=c.get(f'/api/jobs/{jid}').json(); assert job['status']=='completed' and job['result']['frames']==24
    report=c.get(f'/api/jobs/{jid}/report'); assert report.status_code==200 and report.json()['result']['engine']=='legacy-image-screen-v1'
    assert c.get(job['result']['playback']).status_code==200

def test_upload_camera_mode_is_stored_without_geometry_for_moving_camera(monkeypatch):
    source=main.DATA/f'moving-{uuid.uuid4().hex[:8]}.mp4'; w=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,240))
    for _ in range(3): w.write(np.zeros((240,320,3),np.uint8))
    w.release(); monkeypatch.setattr(main.worker,'submit',lambda *args:None)
    c=TestClient(main.app)
    response=c.post('/api/jobs',files={'file':('moving.mp4',source.read_bytes(),'video/mp4')},data={'location':'Moving test','camera_mode':'moving'})
    assert response.status_code==202,response.text
    with main.connect() as db: mode=db.execute('SELECT mode FROM camera_modes WHERE job_id=?',(response.json()['id'],)).fetchone()[0]
    assert mode=='moving'

def test_movement_configuration_is_validated_and_persisted():
    jid=make_job(False); c=TestClient(main.app)
    valid={'config':{'wrong_way':{'enabled':True,'permitted_directions':[[0,1]],'min_duration_s':1.5},'blocked_crossing':{'enabled':False}}}
    response=c.put(f'/api/jobs/{jid}/movement-config',json=valid)
    assert response.status_code==200,response.text
    saved=c.get(f'/api/jobs/{jid}/movement-config').json()
    assert saved['version']==1 and saved['config']['wrong_way']['enabled'] is True
    bad=c.put(f'/api/jobs/{jid}/movement-config',json={'config':{'speeding':{'enabled':True}}})
    assert bad.status_code==422
