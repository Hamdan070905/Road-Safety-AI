"""YOLOPv2 TorchScript adapter following the upstream demo preprocessing and mask decoding.

No vehicle outputs are decoded or sent to RoadGuard's collision tracker.
"""
from pathlib import Path
import cv2
import numpy as np
import torch

REPO_REVISION='V0.0.1 / release commit 8e69e4c'
WEIGHTS=Path(__file__).resolve().parents[3]/'models'/'road_lanes'/'yolopv2.pt'

def load_model(device='cpu'):
    if not WEIGHTS.is_file(): raise FileNotFoundError(f'YOLOPv2 TorchScript checkpoint missing: {WEIGHTS}')
    model=torch.jit.load(str(WEIGHTS),map_location=device).to(device).eval()
    return model

def preprocess(frame, size=640):
    # Upstream demo first resizes source to 1280x720, then stride-32 letterboxes to 640.
    base=cv2.resize(frame,(1280,720),interpolation=cv2.INTER_LINEAR)
    r=min(size/720,size/1280); nw,nh=round(1280*r),round(720*r)
    # Official letterbox(auto=True,stride=32) removes whole stride multiples
    # from padding. YOLOPv2's mask decoder then crops rows 12:372.
    dw=((size-nw)%32)/2; dh=((size-nh)%32)/2
    resized=cv2.resize(base,(nw,nh),interpolation=cv2.INTER_LINEAR)
    left,right=round(dw-.1),round(dw+.1); top,bottom=round(dh-.1),round(dh+.1)
    boxed=cv2.copyMakeBorder(resized,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(114,114,114))
    rgb=boxed[:,:,::-1].transpose(2,0,1).copy()
    return torch.from_numpy(rgb).float().unsqueeze(0)/255.0

def infer(model, frame, device='cpu'):
    x=preprocess(frame).to(device)
    with torch.inference_mode():
        output=model(x)
    if not isinstance(output,(tuple,list)) or len(output)<3:
        raise RuntimeError('YOLOPv2 output does not match official [pred, anchor_grid], seg, lane interface')
    seg,lane=output[1],output[2]
    if seg.ndim!=4 or lane.ndim!=4: raise RuntimeError('Unexpected YOLOPv2 segmentation tensor shapes')
    da=torch.nn.functional.interpolate(seg[:,:,12:372,:],scale_factor=2,mode='bilinear',align_corners=False).argmax(1)[0].byte().cpu().numpy()
    ll=torch.round(torch.nn.functional.interpolate(lane[:,:,12:372,:],scale_factor=2,mode='bilinear',align_corners=False)).squeeze(1)[0].byte().cpu().numpy()
    h,w=frame.shape[:2]
    da=cv2.resize(da,(w,h),interpolation=cv2.INTER_NEAREST)
    ll=cv2.resize(ll,(w,h),interpolation=cv2.INTER_NEAREST)
    return da,ll

def overlay(frame, da, ll, opacity=.45):
    out=frame.copy(); color=np.zeros_like(frame)
    color[da==1]=(45,190,55); color[ll>0]=(220,70,235)
    mask=(da==1)|(ll>0); out[mask]=cv2.addWeighted(frame,1-opacity,color,opacity,0)[mask]
    return out
