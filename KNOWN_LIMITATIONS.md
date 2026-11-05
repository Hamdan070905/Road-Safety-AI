# Known limitations

## Collision/near-miss

- Collision findings remain image-space screening candidates, not contact confirmation. They need human review and no real-world accuracy evaluation has been performed.
- Metric TTC/PET/speed require surveyed fixed-camera calibration and valid timing; moving dashcam remains nonmetric. No crash probability or fault assignment is produced.

## Road damage

- RDD2022 detects only longitudinal crack, transverse crack, alligator crack and pothole boxes. Thin cracks, domain shift, road repairs, shadows and unseen categories can cause misses or false detections.
- In the available normal-looking highway clip the API returned 10 sampled-frame detections grouped into six conservative candidates; visual inspection found a longitudinal-crack box over a painted lane marking. This is a visible false-positive example, not a precision estimate. No positive damaged-road footage with labels was found; application recall is unvalidated.
- Candidate association uses same-class, short-gap image proximity. Moving viewpoint, long occlusion, perspective, or repeated passes make global defect counts unreliable. Candidate clusters are not pothole counts or severity scores.
- No physical width/depth, engineering severity, accident cause/risk or road-wide safety is inferred. Detector confidence is not severity.
- The model weights are AGPL-3.0; review license and upstream dataset terms before redistribution.

## Road and lane segmentation

- YOLOPv2 was verified on clear forward-facing highway footage. The weak/absent-marking condition is unvalidated because no suitable real clip was available. Lane predictions can be absent or false; the overlay is not safe lane-change detection.
- Model domain is driving-scene video. Strongly different camera placement, fisheye views, side-looking fixed cameras, poor weather/night and unusual markings may reduce reliability. Drivable area is not a road-safety score.
- Segmentation masks are sampled at configured intervals; playback holds each sampled overlay until the next sample to preserve source duration. Intermediate frames do not have fresh inference.
- CPU inference is relatively slow (about 17.6 s RDD and 28.1 s YOLOPv2 for this 21.52 s source on this machine). No FPS benchmark is claimed.

## Application validation

- Browser automation host was unavailable; responsive browser layout and real user browser controls were not visually tested. API workflows and downloadable output endpoints were exercised.
- No labeled damaged/normal test split or weak/clear marking evaluation dataset is available. Accuracy metrics are not evaluated.
- Driver monitoring, helmets, red-light violations and pothole depth remain outside these modules.
