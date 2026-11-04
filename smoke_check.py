import os,tempfile,time,json
from pathlib import Path
os.environ['ROADGUARD_DATA']=tempfile.mkdtemp()
import cv2,numpy as np
from fastapi.testclient import TestClient
from backend.main import app
p=Path(tempfile.mkdtemp())/'smoke.mp4'
w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,240))
for i in range(12): w.write(np.zeros((240,320,3),np.uint8))
w.release()
c=TestClient(app)
r=c.post('/api/jobs',files={'file':('smoke.mp4',p.read_bytes(),'video/mp4')},data={'location':'Synthetic smoke test'})
assert r.status_code==202,r.text
j=r.json()['id']
for _ in range(180):
    d=c.get('/api/jobs/'+j).json()
    if d['status'] in ['completed','failed']:break
    time.sleep(1)
assert d['status']=='completed',d
assert d['result']['frames']==12,d
assert c.get(d['result']['playback']).status_code==200
print(json.dumps({'status':d['status'],'frames':12,'runtime':d['result']['runtime'],'events':len(d['result']['events']),'note':'Synthetic blank-video plumbing test, not accuracy evaluation'}))
