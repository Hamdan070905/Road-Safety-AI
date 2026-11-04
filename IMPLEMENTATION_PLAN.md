# Implemented workflow and remaining validation

1. Foundation: retained jobs, evidence, reviews and report format; safe additive SQLite schema upgrades; bounded upload; configurable YOLO; timestamp-derived duration and per-track gates.
2. Calibration: source frame selection, measured metre coordinates, homography validation, analysis polygon, versioning and preview are integrated with fixed-camera worker selection.
3. Event engine: calibrated bottom-center trajectories use metric units and footprint-overlap TTC/clearance. PET is computed only from observed sequential occupancy in configured ground conflict areas. Uncalibrated fallback stays separate.
4. Evidence and review: timestamp-based VFR encoding via FFmpeg when available, event clips, API output and persistent reviews.
5. Analytics: configured counting lines, track passage deduplication, repeated-file exclusion, observation intervals, denominator-gated reviewed-event rates, and deterministic reviewed summaries.
6. Moving camera: experimental tracking-only capability gate; no metric homography applied.
7. Remaining: dependency installation/test execution in this restricted host, automated browser workflow, camera-motion vision detector, polygonal vehicle footprints/class-specific uncertainty, held-out evaluation CLI, and external calibration/accuracy validation.
