# Verification Record

**Updated:** 5 November 2026. Existing collision data, models, and tests were preserved.

## Environment and regression checks

- Existing `.venv312`, Python 3.12, Torch 2.15.0+cpu, CPU-only; Ultralytics 8.3.203, OpenCV 4.11.0, NumPy 2.2.6.
- `.venv312\Scripts\python.exe -m pytest tests -q` — **49 passed**, one existing Starlette/AnyIO deprecation warning, 3.43 s.
- `.venv312\Scripts\python.exe -m pip check` — passed, no broken requirements.
- `.venv312\Scripts\python.exe smoke_check.py` — passed on 12-frame synthetic blank video: completed, 0 events, playback HTTP 200.
- All Python source/test/tool files compile cleanly; `node --check frontend\app.js` passed.
- Pinned models:
  - RDD2022 YOLOv8m: commit `40684513b39331fc2b49f01123cc489ad5876624`, 52,034,002 bytes, SHA-256 `bcbaa230a4fce93c15cfd1dd4e6bb40050ce9099750ea66e270f00c10ecfdcc6`.
  - YOLOPv2: commit `8e69e4c`, 156,380,200 bytes, SHA-256 `f2a8c8374203ae3e67ff9c184e931f763957de92a993b23269e4e721627f1f8c`.

## Road Damage Audit & Improvements

### 1. Root Cause Findings
- **Color Format Handling:** Direct OpenCV BGR numpy arrays match Ultralytics YOLOv8 expectation; manual conversion to RGB degraded confidence on RDD2022 weights.
- **Image Resizing & Checkpoint Alignment:** YOLOv8m checkpoint `train_args` show it was trained at `imgsz: 640`. Upscaling to 960 or 1280 increased inference latency (from ~0.20s to ~0.42s per frame on CPU) and reduced confidence for small and far road defects due to domain discrepancy. 640 remains the recommended baseline size.
- **Pothole Confidence vs. Crack False Positives:** Potholes in typical wheel-level/dashcam footage appear with confidences between 0.18 and 0.40. A uniform threshold of 0.25 caused many genuine potholes to be dropped. Conversely, road paint markings occasionally trigger `longitudinal_crack` at confidences around 0.17–0.24.
- **Resolution:** Introduced configurable per-class confidence thresholds (default: pothole=0.18, cracks=0.25). Under this setting:
  - The intersection lane-paint false positive drops from 1 detection to **0 detections**.
  - True potholes in the wheel-level footage are retained (detecting multiple clear potholes at 0.389 and 0.216).
- **Coordinate Restoration & Tiled Inference:** Added optional overlapping crop inference with exact coordinate offset restoration and Non-Maximum Suppression (`nms_boxes`) for high-resolution sources.

### 2. Temporal Association
- Improved `associate()` in `backend/modules/road_damage/service.py` to prevent duplicate counting as defects are approached:
  - Bounded center shift normalized by image diagonal (max 0.15).
  - Bounded bounding box area growth ratio (max 3.0x between consecutive frames) to distinguish an approached pothole from an unrelated nearby defect.
  - Distinct defect clusters receive unique candidate IDs (`D00001`, `D00002`), explicitly labeled as candidates rather than globally unique defect counts.
  - Boxes are never duplicated across unsampled video frames.

### 3. Evaluation Workflow
- Implemented `backend/modules/road_damage/evaluation.py` supporting evaluation across:
  - Potholes
  - Longitudinal cracks
  - Transverse cracks
  - Alligator cracks
  - Normal road hard negatives (lane paint, puddles, patches, shadows, manholes)
- Reports per-class Precision, Recall, F1, AP, and hard negative false positive counts. Tested in `tests/test_road_damage.py`.

### 4. Fine-Tuning Assessment
- Direct inference with per-class thresholds and 5 FPS sampling captures real pothole candidates while suppressing lane paint false positives.
- Fine-tuning is not yet warranted until a dedicated, geographically balanced validation dataset with rural/wet/pothole ground truth annotations is assembled and labeled.

## Reproduction

```powershell
.\.venv312\Scripts\python.exe -m pytest tests -q
.\.venv312\Scripts\python.exe -m pip check
node --check frontend\app.js
.\.venv312\Scripts\python.exe smoke_check.py
```
