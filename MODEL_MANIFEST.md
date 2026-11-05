# Model manifest

## Collision / near-miss pipeline (preserved)
- YOLO11n checkpoint `yolo11n.pt`, Ultralytics 8.3.203; ByteTrack `bytetrack.yaml` with `lap` 0.5.12. Existing jobs, APIs, thresholds and SQLite data are independent of the optional modules below. Optional inference is lazy and submitted to the same one-worker executor.

## Road Damage Detection
- Upstream: [dronefreak/rdd2022-yolov8m](https://huggingface.co/dronefreak/rdd2022-yolov8m), pinned revision `40684513b39331fc2b49f01123cc489ad5876624`, file `best.pt`, license AGPL-3.0.
- Local: `models/road_damage/best.pt` (52,034,002 bytes), SHA-256 `bcbaa230a4fce93c15cfd1dd4e6bb40050ce9099750ea66e270f00c10ecfdcc6` in `best.pt.sha256`.
- Loaded with existing Ultralytics 8.3.203 using the model card's YOLO API; actual `model.names`: 0 longitudinal_crack, 1 transverse_crack, 2 alligator_crack, 3 pothole. Read dynamically at inference.
- Upstream-reported RDD2022 split benchmarks: mAP@50 62.03%; pothole AP@50 74.15%. These are not RoadGuard measurements and have dataset/domain limitations. See card for full metrics.

- Dataset source reference: [Sekilab RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector); the model card identifies its RDD2022 source and data terms.

## Road & Lane Segmentation
- Upstream: [CAIC-AD/YOLOPv2](https://github.com/CAIC-AD/YOLOPv2), official release `V0.0.1`, release commit `8e69e4c`, checkpoint `yolopv2.pt`, MIT license.
- Local: `models/road_lanes/yolopv2.pt` (156,380,200 bytes), SHA-256 `f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c` in `yolopv2.pt.sha256`.
- Uses `torch.jit.load`, official demo's fixed 1280x720 resize, stride-32 auto-letterbox, `[pred, anchor_grid], seg, lane` output and the upstream 12:372 crop/interpolation mask decoding. Collision object outputs are ignored.
- Upstream requirements pin Torch 1.7 / torchvision .8; current isolated TorchScript inference succeeded on existing Torch 2.15.0+cpu and torchvision installed in `.venv312`. No legacy package downgrade performed.

## Machine
- Windows, Python 3.12 `.venv312`, CPU-only (`torch 2.15.0+cpu`, CUDA unavailable), OpenCV 4.11.0, NumPy 2.2.6.
- Run `.venv312\Scripts\python.exe tools\fetch_optional_models.py --model all` (or choose `road_damage` / `road_lanes`) to cache pinned files. The script verifies a previously recorded local SHA-256 before reusing a cached checkpoint.
- Weights are ignored by Git; do not distribute the AGPL RDD model or data without reviewing the upstream terms and attribution.


