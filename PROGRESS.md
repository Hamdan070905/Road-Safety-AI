# Progress

- Verified existing YOLO11n + ByteTrack processing on available original footage and reran the current upload-to-export synthetic smoke path.
- Current `.venv312` checks: 24 tests pass; `pip check`, Python compilation, and frontend `node --check` pass. Smoke processes 12 blank frames, completes with zero events, and serves playback.
- Added upload-time fixed/moving camera selection, persistent event review notes, browser-persistent event filter selection, and a dependency-free stored-result PDF report endpoint.
- Added integration checks for review note persistence, PDF generation, and upload camera-mode storage.
- Wrote FEATURE_STATUS.md against current code. Added movement-risk modules (wrong-way, stopped-vehicle, blocked crossing, speed/deceleration) and conflict-zone heatmaps are not implemented and remain explicit gaps.
- Real footage evaluation remains not evaluated: the available original is ordinary highway traffic with no manual event annotations or independent road measurements. Do not interpret unreviewed candidates as real near misses.
- Browser host unavailable during this run; UI visual/mobile QA remains unverified.
