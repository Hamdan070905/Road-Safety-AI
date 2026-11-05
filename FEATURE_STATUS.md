# Feature status

Checked against repository and executed workflows on 5 November 2026. “Executed successfully” means the named software workflow ran; it does not establish field accuracy.

| Feature | Status | Evidence / limitation |
|---|---|---|
| Upload validation and persistent jobs | Implemented and executed successfully | API integration and smoke workflow; SQLite-backed jobs, extension/size/empty-file checks. |
| YOLO11 and ByteTrack on actual footage | Implemented and executed successfully | Supplied `clip_04.mp4`, 179 frames; actual YOLO11n + ByteTrack worker run. No labeled detector/tracker accuracy evaluation. |
| Calibration mapping, validation, versions | Implemented and executed successfully | Geometry/API synthetic tests; no surveyed independent references. |
| Calibrated versus image-space selection | Implemented and executed successfully | Worker tests and real uncalibrated run; physical values withheld in image-space mode. |
| TTC/PET/clearance semantics | Implemented and executed successfully | Geometry tests. TTC is predicted closest approach, not collision; PET requires observed shared-area occupancy; unavailable values include reasons. No real metric validation. |
| Event persistence and deduplication | Implemented and executed successfully | Worker/database tests; collision and near-miss are separate event types. |
| Collision candidates | Implemented and executed successfully | Actual supplied clip version 7 produced one pending image-space candidate at 2.202 s, interval 1.502–2.603 s. Requires reviewer; not a confirmed collision or accuracy evaluation. |
| Playable annotated evidence and timestamps | Implemented and executed successfully | Real run playback, event clip and thumbnail retrieved from API (HTTP 200); source-aligned timestamp 2.202 s. |
| Review persistence and notes | Implemented and executed successfully | Integration test persists review and notes; collision remains pending in this run. |
| Location exposure deduplication and rates | Implemented but unverified | Hash deduplication and thresholds exist; statistical correctness across real recordings not validated. |
| Dashboard, JSON, PDF from stored results | Implemented and executed successfully | Real stored result served by API; PDF endpoint returned valid PDF. Browser visual interaction remains unverified. |
| Event timeline, playback seeking, review filters | Implemented but unverified | Code connected and JS syntax checked; browser automation host unavailable. |
| Movement-risk configuration and candidates | Implemented and executed successfully (synthetic tests) | Wrong-way, stopped vehicle with queue suppression, blocked crossing, calibrated speeding and deceleration logic/config API. No real footage validation; queue classification is limited context suppression. |
| Passage counts and queue observations | Implemented but unverified | Unique line-crossing counts exist; full queue analytics and identity-switch robustness are incomplete. |
| Location summaries and heatmaps | Implemented but unverified | Stored results feed summaries; no fabricated map markers. Geospatial heatmap is not implemented. |
| Fixed/moving camera separation | Implemented and executed successfully | Moving-camera mode gates metric/collision screening; no validated ego-motion compensation. |
| Road Damage Detection (RDD2022 YOLOv8m) | Implemented and executed successfully | Pinned AGPL checkpoint loaded. Real highway video API worker: 22 samples, 10 frame detections, 6 candidate clusters, playable annotated output, persisted review and PDF/JSON. Live single-image API also completed with model-provided class map. Visually inspected false positive over lane paint; no labeled damaged-road positives. |
| Road & Lane Segmentation (YOLOPv2) | Implemented and executed successfully | Pinned MIT TorchScript checkpoint loaded with documented preprocessing/decoding. Real highway video API worker: 22 samples, source-duration-aligned output, masks/overlays, review and exports. Weak/absent marking footage unavailable. |
| Driver monitoring, helmet, red-light violations | Missing | Require dedicated models/inputs. |
| Real-video accuracy evaluation | Blocked | No manual event annotations or independent ground measurements. Precision, recall, false alarms/minute and physical metric accuracy are Not evaluated. |
| Infrastructure browser workflows | Implemented but unverified | Browser access host unavailable; API workflows exercised, browser interactions/mobile layout not visually tested. |



