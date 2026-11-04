# Verification Record

**Updated:** 5 November 2026

## Environment

Used existing `.venv312` (Python 3.12); no packages installed. Runtime versions: OpenCV 4.11.0, NumPy 2.2.6, Ultralytics 8.3.203. Requirements select one OpenCV distribution.

## Executed checks

- `.venv312\Scripts\python.exe -m pip check` — passed, no broken requirements.
- `.venv312\Scripts\python.exe -m pytest tests -q` — **41 passed**, one deprecation warning (3.27 s).
- `.venv312\Scripts\python.exe -m py_compile backend\main.py backend\collision.py backend\movement.py backend\conflict.py` — passed.
- `node --check frontend\app.js` — passed.
- Real worker execution on supplied original — completed, analysis version 7, 179 frames, 5.939 s, runtime 9.22 s, YOLO11n + ByteTrack.
- Live API check — health OK; stored job completed; annotated playback, event MP4, thumbnail, collision diagnostics and PDF all returned HTTP 200. PDF header `%PDF-1.4`.

## Collision regression footage

Original input: `data/ac1ee0f1deb3425fb6c22bbf690bdc1e/input.mp4` (job `clip_04.mp4`). Before fix (analysis v1), no events were stored. The old conflict engine only projected future closest approach from same-ID normalized bottom centers; it had no post-contact/impact response logic, and a participant fragmented from ID 3 to ID 15. The apparent interaction was already unfolding, so the future-CPA gate rejected it.

After fix (v7), one `possible_collision` candidate (`c0e4e6ca`) was stored at 2.202 s, interval 1.502–2.603 s, participants IDs 3, 7, 15. Signals include approach, 56 overlap observations, peak box IoU 0.081, and an image-space movement response/heading change. Identity reliability is explicitly false due to fragments 3/15. Review remains pending. TTC, PET and metric clearance are null with reasons. This development clip is not held-out evaluation and the automated candidate is not a verified collision.

Artifacts:

- Annotated playback: `data/ac1ee0f1deb3425fb6c22bbf690bdc1e/annotated.mp4`
- Evidence clip: `data/ac1ee0f1deb3425fb6c22bbf690bdc1e/c0e4e6ca.mp4`
- Evidence thumbnail: `data/ac1ee0f1deb3425fb6c22bbf690bdc1e/c0e4e6ca.jpg`
- Stored JSON: `data/ac1ee0f1deb3425fb6c22bbf690bdc1e/report.json`
- PDF downloaded from current report endpoint: `data/ac1ee0f1deb3425fb6c22bbf690bdc1e/report.pdf`
- Pair diagnostics: `data/ac1ee0f1deb3425fb6c22bbf690bdc1e/collision_diagnostics.json`

API asset checks validated returned media types and bytes. Browser automation discovery failed with “browser access host is unavailable”; upload-to-review interaction, visual timestamp seeking, and desktop/mobile layout remain untested in a browser.

## Test coverage and evaluation limits

The 41 tests include collision positives, parallel/perspective overlap and queue negatives, safe passing, turns, occlusion, scene cuts/ID fragments, moving camera, movement-risk positives/negatives, geometry, review/report persistence, API configuration and event-type deduplication. No real-video manual labels or independent surveyed geometry exist. Precision, recall, false alarms/minute, duplicate rate, and real TTC/PET/speed accuracy are **Not evaluated**.

## Reproduction

```powershell
.venv312\Scripts\python.exe -m pytest tests -q
.venv312\Scripts\python.exe -m pip check
.venv312\Scripts\python.exe -m py_compile backend\main.py backend\collision.py backend\movement.py backend\conflict.py
node --check frontend\app.js
```

Start: `.venv312\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`.
