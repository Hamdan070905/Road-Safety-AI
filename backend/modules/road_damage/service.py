"""Incremental road-damage inference utilities and conservative association."""
import cv2
import numpy as np

def associate(observations, max_gap_s=2.0, max_center_shift=0.15, max_scale_change=3.0):
    """
    Associate same-class detections across time conservatively.
    
    Prevents:
    - Merging distinct nearby defects (enforces distance gating and scale bounds).
    - Counting approaching defects multiple times without tracking.
    - Freezing boxes across unsampled frames.
    
    Candidate groupings are clearly labeled as defect candidates, not globally unique ground-truth counts.
    """
    tracks = []
    next_id = 1
    # Sort strictly by timestamp, then class_id
    for obs in sorted(observations, key=lambda x: (x['timestamp_s'], x['class_id'])):
        x1, y1, x2, y2 = obs['bbox']
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w_obs = max(1.0, x2 - x1)
        h_obs = max(1.0, y2 - y1)
        area_obs = w_obs * h_obs

        candidates = []
        for tr in tracks:
            last = tr[-1]
            dt = obs['timestamp_s'] - last['timestamp_s']
            if dt <= 0 or dt > max_gap_s or obs['class_id'] != last['class_id']:
                continue
            lx1, ly1, lx2, ly2 = last['bbox']
            lcx = (lx1 + lx2) / 2.0
            lcy = (ly1 + ly2) / 2.0
            l_area = max(1.0, (lx2 - lx1) * (ly2 - ly1))

            # Center shift normalized by image diagonal
            dist = ((cx - lcx)**2 + (cy - lcy)**2)**0.5
            h_img, w_img = obs.get('_shape', (1, 1))
            diag = max((w_img*w_img + h_img*h_img)**0.5, 1.0)
            norm_shift = dist / diag

            # Scale change ratio
            scale_ratio = max(area_obs, l_area) / min(area_obs, l_area)

            # In moving dashcam footage, a pothole moves downwards/outwards and grows in size as approached.
            # Reject if center shift is too large or size changed unnaturally.
            if norm_shift <= max_center_shift and scale_ratio <= max_scale_change:
                candidates.append((norm_shift, tr))

        if candidates:
            # Attach to closest candidate track
            best_track = min(candidates, key=lambda p: p[0])[1]
            obs['candidate_id'] = best_track[0]['candidate_id']
            best_track.append(obs)
        else:
            obs['candidate_id'] = f'D{next_id:05d}'
            next_id += 1
            tracks.append([obs])

    return observations

def draw(image, detections):
    """Draw bounding boxes with defect class and confidence (never labeled as severity)."""
    out = image.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d['bbox'])
        cand = d.get('candidate_id', '')
        cand_str = f" [{cand}]" if cand else ""
        label = f"{d['class_name']} conf:{d['confidence']:.2f}{cand_str}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (20, 60, 245), 2)
        cv2.putText(out, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 60, 245), 2)
    return out
