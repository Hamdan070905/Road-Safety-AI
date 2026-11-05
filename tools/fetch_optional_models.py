"""Fetch only the two explicitly reviewed optional RoadGuard checkpoints.

The first download is pinned to an immutable HF commit or GitHub release tag.
Its SHA-256 is then stored beside the checkpoint and checked on every later run.
The existing collision checkpoint is never touched.
"""
import argparse, hashlib, os, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODELS={
 'road_damage':('https://huggingface.co/dronefreak/rdd2022-yolov8m/resolve/40684513b39331fc2b49f01123cc489ad5876624/best.pt?download=true',ROOT/'models/road_damage/best.pt'),
 'road_lanes':('https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt',ROOT/'models/road_lanes/yolopv2.pt'),
}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def fetch(name):
    url,path=MODELS[name]; path.parent.mkdir(parents=True,exist_ok=True); side=path.with_suffix(path.suffix+'.sha256')
    if path.exists():
        actual=sha(path)
        if side.exists() and side.read_text().split()[0]!=actual: raise RuntimeError(f'Existing checkpoint checksum mismatch: {path}')
        if not side.exists(): side.write_text(actual+'  '+path.name+'\n',encoding='ascii')
        print(f'{name}: cached {path}; sha256={actual}'); return
    partial=path.with_suffix(path.suffix+'.part')
    req=urllib.request.Request(url,headers={'User-Agent':'RoadGuardAI/1.0 model-cache'})
    h=hashlib.sha256()
    try:
        with urllib.request.urlopen(req,timeout=60) as response, partial.open('wb') as out:
            length=response.headers.get('Content-Length'); total=int(length) if length and length.isdigit() else None; size=0
            while block:=response.read(1024*1024):
                out.write(block); h.update(block); size+=len(block)
                if total: print(f'\r{name}: {size*100//total}%',end='',flush=True)
        if total and size!=total: raise RuntimeError(f'Incomplete checkpoint: downloaded {size}, expected {total} bytes')
        digest=h.hexdigest(); os.replace(partial,path); side.write_text(digest+'  '+path.name+'\n',encoding='ascii')
        print(f'\n{name}: cached {path}; bytes={size}; sha256={digest}')
    except Exception:
        partial.unlink(missing_ok=True); raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--model',choices=[*MODELS,'all'],default='all'); args=parser.parse_args()
    for key in MODELS if args.model=='all' else [args.model]: fetch(key)
