"""Isolated contract and persistence tests; these do not claim model inference."""
import uuid, shutil
from pathlib import Path
import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient
from backend.modules.road_damage.service import associate
from backend.modules.road_lanes.adapter import preprocess, infer

def test_damage_association_is_temporal_and_does_not_merge_same_frame_boxes():
    rows=[
        {'id':'a','timestamp_s':0,'class_id':3,'bbox':[10,10,30,30],'_shape':(100,100)},
        {'id':'b','timestamp_s':0,'class_id':3,'bbox':[12,10,32,30],'_shape':(100,100)},
        {'id':'c','timestamp_s':1,'class_id':3,'bbox':[11,10,31,30],'_shape':(100,100)},
        {'id':'d','timestamp_s':5,'class_id':3,'bbox':[11,10,31,30],'_shape':(100,100)},
    ]
    out=associate(rows)
    assert out[0]['candidate_id']!=out[1]['candidate_id']
    assert out[0]['candidate_id']==out[2]['candidate_id']
    assert out[3]['candidate_id'] not in {x['candidate_id'] for x in out[:3]}

def test_yolopv2_upstream_letterbox_input_and_mask_coordinate_restoration():
    x=preprocess(np.zeros((360,640,3),dtype=np.uint8))
    assert tuple(x.shape)==(1,3,384,640) and x.min()==0 and abs(float(x.max())-114/255)<1e-5
    class MockModel:
        def __call__(self, inp):
            seg=__import__('torch').zeros((1,2,384,640)); seg[:,1]=1
            lane=__import__('torch').ones((1,1,384,640))
            return (None,seg,lane)
    da,ll=infer(MockModel(),np.zeros((360,640,3),dtype=np.uint8),'cpu')
    assert da.shape==(360,640) and ll.shape==(360,640)
    assert np.all(da==1) and np.all(ll==1)

def test_independent_job_failure_is_persisted_without_falling_back_to_collision_model(monkeypatch):
    from backend import main
    from backend.modules import jobs as module_jobs
    from backend.modules.road_damage import adapter as damage
    tmp_path=Path(__file__).resolve().parents[1]/'data'/('module-test-'+uuid.uuid4().hex[:10]); tmp_path.mkdir()
    class DirectExecutor:
        def submit(self,fn,*args): fn(*args)
    monkeypatch.setattr(module_jobs,'DATA',tmp_path)
    monkeypatch.setattr(module_jobs,'DB',tmp_path/'roadguard.sqlite')
    monkeypatch.setattr(module_jobs,'pool',DirectExecutor())
    monkeypatch.setattr(damage,'WEIGHTS',tmp_path/'missing-best.pt')
    # Valid PNG proves this is a model-readiness failure and not upload validation.
    import cv2
    ok,encoded=cv2.imencode('.png',np.zeros((64,96,3),dtype=np.uint8)); assert ok
    client=TestClient(main.app)
    response=client.post('/api/road-damage/jobs',files={'file':('road.png',encoded.tobytes(),'image/png')},data={'settings':'{"confidence":0.25,"iou":0.45,"sample_interval_s":1}'})
    assert response.status_code==202
    job=client.get('/api/module-jobs/'+response.json()['id']).json()
    assert job['kind']=='road_damage' and job['status']=='failed'
    assert 'RDD2022 checkpoint is not installed' in job['error']
    assert job['result'] is None
    assert client.get('/api/jobs').status_code==200
    shutil.rmtree(tmp_path,ignore_errors=True)

def test_image_upload_is_processed_once_and_gets_independent_exports(monkeypatch):
    from types import SimpleNamespace
    from backend import main
    from backend.modules import jobs as module_jobs
    from backend.modules.road_damage import adapter as damage
    temp=Path(__file__).resolve().parents[1]/'data'/('module-image-test-'+uuid.uuid4().hex[:10]); temp.mkdir()
    class Direct:
        def submit(self,fn,*args): return fn(*args)
    class FakeModel: names={0:'longitudinal_crack',1:'transverse_crack',2:'alligator_crack',3:'pothole'}
    (temp/'best.pt').write_bytes(b'test-checkpoint')
    monkeypatch.setattr(module_jobs,'DATA',temp); monkeypatch.setattr(module_jobs,'DB',temp/'db.sqlite'); monkeypatch.setattr(module_jobs,'pool',Direct()); monkeypatch.setattr(main,'DATA',temp)
    monkeypatch.setattr(damage,'WEIGHTS',temp/'best.pt'); monkeypatch.setattr(damage,'load_model',lambda:FakeModel()); monkeypatch.setattr(damage,'predict',lambda *a,**k:SimpleNamespace(boxes=None))
    ok,png=cv2.imencode('.png',np.zeros((48,80,3),np.uint8)); assert ok
    try:
        client=TestClient(main.app)
        response=client.post('/api/road-damage/jobs',files={'file':('still.png',png.tobytes(),'image/png')})
        job=client.get('/api/module-jobs/'+response.json()['id']).json()
        assert response.status_code==202 and job['status']=='completed',job
        assert job['result']['sampled_frames']==1 and job['result']['observations']==[]
        assert (temp/job['id']/'report.json').is_file() and (temp/job['id']/'report.pdf').is_file()
        assert client.get(job['result']['preview']).status_code==200
        assert client.get(f"/api/module-jobs/{job['id']}/report.pdf").content.startswith(b'%PDF-1.4')
    finally: shutil.rmtree(temp,ignore_errors=True)
