# RoadGuard AI

Local research prototype for identifying potential traffic interactions in camera footage, preserving reviewable evidence, and summarizing observed events. Events require human review and do not establish crash probability, causation, or fault.

## Launch

Install Python 3.12 and FFmpeg. Windows: `start.bat` creates `.venv312`, installs pinned dependencies, and starts at `http://127.0.0.1:8000`. macOS/Linux: `bash start.sh` creates `.venv` and runs the same install/start flow. Both bind to localhost. `ROADGUARD_MODEL=yolo11s.pt` selects YOLO11s; default is YOLO11n. CPU is supported; CUDA is selected only if PyTorch reports it available. MPS stays disabled unless explicitly enabled for local experiments because this pipeline has not been validated on Apple hardware.

## Fixed-camera workflow

1. Upload a recording and wait for tracking to finish.
2. Open **Calibration** and select the recording.
3. Load a representative source frame, click at least four distributed road-plane points, and enter their measured ground coordinates in metres in the same order. Set a valid analysis region. The app estimates a homography and stores a calibration version with fitting residual. The top-down preview displays the entered measured points.
4. Optionally configure a counting line using two source-frame clicks and a shared conflict area using a measured ground polygon. PET is derived only from observed entry and exit times for that configured area.
5. Reanalyze. The worker snapshots the latest calibration version and uses road-contact points, per-track timestamp quality gates, the calibrated footprint engine, and event consolidation. Without a valid fixed-camera calibration, it uses an explicitly labeled image-space fallback and withholds metric TTC/PET/clearance.
6. Inspect video and event evidence, review as confirmed/rejected/uncertain, and export the report. Reviews persist in SQLite.

Camera motion invalidates ground calibration. Mark it moved or switch mode to moving before reanalysis. Moving-camera mode is experimental tracking/evidence only; fixed homography metrics are suppressed.

Calibration assumes a planar road and does not correct lens distortion. Fitting residual is not independent accuracy validation. Footprints use configurable in-code circular approximations, and constant-velocity TTC is a screening estimate. PET is unavailable until a configured shared area has sequential observed occupancy. Null metrics include unavailable reasons. No road dimensions or lane widths are invented.

## Location observations

Configured image-space counting lines deduplicate crossings by track ID. Byte-identical recordings are identified and excluded from repeated exposure totals. Location summaries report confirmed events, observed time, counted passages, coverage, hourly confirmed-event rates, and conflicts per 1,000 counted users. Rates are withheld below ten unique observed minutes or 100 counted passages, or when no counting line is configured. These are descriptive reviewed-event rates, not crash probabilities. Identity fragmentation can overcount traffic passages.

Deterministic insights list reviewed events, recorded metrics, the observation period, uncertainty, and an on-site investigation suggestion. They do not infer cause.

## Data, tests, and limitations

Data defaults to `data/`; set `ROADGUARD_DATA` to move SQLite and evidence. Schema startup adds tables and preserves existing jobs, footage, and reviews. Upload limit is 200 MB; analysis is capped at five source minutes. Variable-frame-rate annotated output uses source timestamp deltas through FFmpeg; without FFmpeg playback falls back to an untimed constant-rate preview, indicated in the report.

Run `python -m pytest tests -q`, `node --check frontend/app.js`, and `python smoke_check.py`. Synthetic tests are not real-video accuracy evaluation. Held-out annotated footage and independently measured ground references are required for precision/recall and metric validation.

See [PROGRESS.md](PROGRESS.md), [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), [MODEL_MANIFEST.md](MODEL_MANIFEST.md), [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md), and [VERIFICATION.md](VERIFICATION.md).

## Verified current workflow additions

Upload includes a fixed or moving camera choice. A saved fixed-camera calibration can enable metric screening; moving-camera mode keeps results in image space. Review status and notes persist in SQLite. The review filter selection persists in the browser. Each completed job exposes a JSON report and a generated PDF report based on its stored result and current review state. PDF export is a compact text report.

The current build does not implement the additional wrong-way, stopped-vehicle, blocked-crossing, speeding, or sudden-deceleration candidates. Count-line passage counting is available, but queue occupancy analysis and conflict-zone heatmaps are not. See [FEATURE_STATUS.md](FEATURE_STATUS.md) for a full checked inventory and [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for evaluation and model gaps.

## Existing environment startup

To launch without reinstalling dependencies into the existing Python 3.12 environment:

```powershell
.venv312\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Upload a video, select its camera mode, and wait for completion. For fixed-camera metric screening, open Calibration, select the recording, load a source frame, click distributed road points, enter independently measured ground coordinates in metres, save the calibration and analysis polygon, and reanalyze. Inspect a candidate from the event list, save a review with an optional note, refresh to confirm persistence, then export JSON or PDF from the analysis workspace. With no candidates, the valid result is an empty event list.
