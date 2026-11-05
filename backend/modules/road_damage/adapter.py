"""Lazy RDD2022 Ultralytics adapter with tiled inference and coordinate restoration."""
from pathlib import Path
import os
import cv2
import numpy as np

REPO = 'dronefreak/rdd2022-yolov8m'
REVISION = '40684513b39331fc2b49f01123cc489ad5876624'
WEIGHTS = Path(__file__).resolve().parents[3] / 'models' / 'road_damage' / 'best.pt'

def load_model():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(
            f'RDD2022 checkpoint is not installed: {WEIGHTS}. '
            'Fetch the pinned best.pt revision explicitly; generic COCO weights are not a substitute.'
        )
    os.environ.setdefault('YOLO_CONFIG_DIR', str(WEIGHTS.parents[2] / 'data' / 'ultralytics'))
    from ultralytics import YOLO
    return YOLO(str(WEIGHTS))

def _box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def nms_boxes(detections, iou_threshold=0.45):
    """Greedy Non-Maximum Suppression across detections of the same class."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    kept = []
    while sorted_dets:
        best = sorted_dets.pop(0)
        kept.append(best)
        sorted_dets = [
            d for d in sorted_dets
            if d['class_id'] != best['class_id'] or _box_iou(d['bbox'], best['bbox']) < iou_threshold
        ]
    return kept

def predict(model, image, confidence=0.20, iou=0.45, device='cpu', classes=None, imgsz=640, tiled=False, per_class_conf=None):
    """Run model prediction with source coordinate restoration and optional tiled cropping."""
    h_orig, w_orig = image.shape[:2]
    all_raw_boxes = []

    # Determine base inference confidence: minimum of global confidence and any per-class threshold
    # to guarantee initial model inference does not discard candidates before per-class filtering.
    base_conf = float(confidence)
    if per_class_conf:
        min_class_conf = min(float(v) for v in per_class_conf.values())
        base_conf = min(base_conf, min_class_conf)

    # 1. Full frame inference (Ultralytics handles internal letterbox resizing)
    res_full = model.predict(source=image, imgsz=imgsz, conf=base_conf, iou=iou, device=device, classes=classes, verbose=False)[0]
    boxes = res_full.boxes
    if boxes is not None:
        xy = boxes.xyxy.cpu().tolist() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
        cls = boxes.cls.cpu().tolist() if hasattr(boxes.cls, 'cpu') else boxes.cls
        conf = boxes.conf.cpu().tolist() if hasattr(boxes.conf, 'cpu') else boxes.conf
        if len(xy):
            for b, c, s in zip(xy, cls, conf):
                all_raw_boxes.append({'class_id': int(c), 'confidence': float(s), 'bbox': [float(v) for v in b]})

    # 2. Overlapping crop inference for high-resolution images if tiled=True
    if tiled and (w_orig > 1000 or h_orig > 1000):
        tile_size = 960 if (w_orig > 2000 or h_orig > 2000) else 640
        stride = int(tile_size * 0.75)
        
        y_steps = list(range(0, max(1, h_orig - tile_size + 1), stride))
        if y_steps[-1] + tile_size < h_orig:
            y_steps.append(h_orig - tile_size)
        x_steps = list(range(0, max(1, w_orig - tile_size + 1), stride))
        if x_steps[-1] + tile_size < w_orig:
            x_steps.append(w_orig - tile_size)

        for y in y_steps:
            for x in x_steps:
                tile = image[y:y+tile_size, x:x+tile_size]
                res_tile = model.predict(source=tile, imgsz=640, conf=base_conf, iou=iou, device=device, classes=classes, verbose=False)[0]
                t_boxes = res_tile.boxes
                if t_boxes is not None and len(t_boxes):
                    t_xy = t_boxes.xyxy.cpu().tolist()
                    t_cls = t_boxes.cls.cpu().tolist()
                    t_conf = t_boxes.conf.cpu().tolist()
                    for b, c, s in zip(t_xy, t_cls, t_conf):
                        all_raw_boxes.append({
                            'class_id': int(c),
                            'confidence': float(s),
                            'bbox': [float(b[0] + x), float(b[1] + y), float(b[2] + x), float(b[3] + y)]
                        })

    # Apply per-class confidence filtering if specified
    if per_class_conf:
        names = model.names
        filtered = []
        for d in all_raw_boxes:
            c_name = names.get(d['class_id'], str(d['class_id']))
            thresh = per_class_conf.get(c_name, confidence)
            if d['confidence'] >= thresh:
                filtered.append(d)
        all_raw_boxes = filtered
    else:
        all_raw_boxes = [d for d in all_raw_boxes if d['confidence'] >= confidence]

    # Multi-tile suppression / coordinate restoration
    return nms_boxes(all_raw_boxes, iou_threshold=iou)
