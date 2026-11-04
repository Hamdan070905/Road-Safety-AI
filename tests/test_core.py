import os, tempfile
os.environ['ROADGUARD_DATA']=tempfile.mkdtemp()
from fastapi.testclient import TestClient
from backend.main import app
from backend.engine import approach
client=TestClient(app)
def point(x,v): return {'x':x,'y':.5,'vx':v,'vy':0}
def test_converging(): assert approach(point(.2,.2),point(.8,-.2))['closest_approach_seconds']==1.5
def test_diverging(): assert approach(point(.2,-.2),point(.8,.2)) is None
def test_parallel(): assert approach(point(.2,.2),point(.8,.2)) is None
def test_safe_separation(): assert approach(point(.2,.2),{'x':.8,'y':.9,'vx':-.2,'vy':0}) is None
def test_health(): assert client.get('/api/health').status_code==200
def test_empty_jobs(): assert client.get('/api/jobs').json()==[]
def test_missing(): assert client.get('/api/jobs/missing').status_code==404
def test_bad_upload(): assert client.post('/api/jobs',files={'file':('bad.txt',b'123')}).status_code==400
def test_empty_upload(): assert client.post('/api/jobs',files={'file':('empty.mp4',b'')}).status_code==400
def test_dashboard(): assert 'RoadGuard' in client.get('/').text
