# Model manifest

- Detector default: Ultralytics YOLO11n (`yolo11n.pt`), configurable via `ROADGUARD_MODEL`. YOLO11s is not evaluated here.
- Tracker: Ultralytics ByteTrack (`bytetrack.yaml`) with `lap` pinned.
- Device: CUDA when PyTorch reports it available, otherwise CPU. MPS is opt-in and unvalidated.
- Classes screened: person, bicycle, car, motorcycle, bus, truck.
- Actual local inference exists on ordinary highway footage, but no labeled evaluation has been performed. Precision, recall, false alarms/minute, duplicate rate, and physical measurement accuracy are Not evaluated.
- No accident predictor, lane segmentation model, road-damage model, moving-camera metric model, or specialized driver/helmet/red-light model is integrated.
- YOLOPv2 and road-damage inference are blocked pending verified model weights, compatible inference, class mapping, and license review. Generic COCO weights are not used as road-damage detection.
- Weights licensing is governed by upstream Ultralytics terms; review before distribution or commercial use.
