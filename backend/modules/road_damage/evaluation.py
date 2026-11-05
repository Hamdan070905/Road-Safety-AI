"""Annotation import and evaluation workflow for road damage detection.

Evaluates:
- Potholes (class 3)
- Longitudinal cracks (class 0)
- Transverse cracks (class 1)
- Alligator cracks (class 2)
- Normal road hard negatives (puddles, patches, shadows, lane paint, manholes)

Supports split by original recording/location, reporting per-class Precision, Recall, AP, and FP.
"""
import json
import sys
from pathlib import Path
import numpy as np

DAMAGE_CLASSES = {
    0: 'longitudinal_crack',
    1: 'transverse_crack',
    2: 'alligator_crack',
    3: 'pothole'
}

def box_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def compute_ap50_class(tp_list, fp_list, total_gt):
    """
    Compute 11-point interpolated Average Precision (AP50) from ranked predictions.
    tp_list: list of 1 if prediction is true positive, 0 otherwise
    fp_list: list of 1 if prediction is false positive, 0 otherwise
    total_gt: total ground-truth targets for this class
    """
    if total_gt == 0:
        return 0.0
    if len(tp_list) == 0:
        return 0.0
    
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    
    recalls = tp_cumsum / float(total_gt)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)
    
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p_candidates = precisions[recalls >= t]
        p = np.max(p_candidates) if len(p_candidates) > 0 else 0.0
        ap += p / 11.0
    return float(ap)

def evaluate_dataset(ground_truth_records, prediction_records, iou_thresh=0.5):
    """
    Evaluate predictions against ground truth using one-to-one same-class matching at IoU threshold.
    
    ground_truth_records: dict mapping item_id -> list of {
        'class_id': int,
        'bbox': [x1, y1, x2, y2],
        'is_negative_example': bool (optional),
        'uncertain': bool (optional)
    }
    prediction_records: dict mapping item_id -> list of {
        'class_id': int,
        'bbox': [x1, y1, x2, y2],
        'confidence': float
    }
    iou_thresh: float (default 0.5 for AP50)
    """
    stats_by_class = {
        cid: {'tp': 0, 'fp': 0, 'fn': 0, 'predictions': [], 'total_gt': 0}
        for cid in DAMAGE_CLASSES.keys()
    }
    hard_negative_fps = 0
    all_item_ids = set(ground_truth_records.keys()) | set(prediction_records.keys())

    # First pass: count total non-negative, non-uncertain ground truths
    for item_id, gt_boxes in ground_truth_records.items():
        for g in gt_boxes:
            if not g.get('is_negative_example') and not g.get('uncertain'):
                cid = g['class_id']
                if cid in stats_by_class:
                    stats_by_class[cid]['total_gt'] += 1

    # Second pass: evaluate predictions per image (including negative/empty images)
    for item_id in all_item_ids:
        gt_boxes = ground_truth_records.get(item_id, [])
        preds = prediction_records.get(item_id, [])
        
        # Sort predictions by confidence descending
        preds_sorted = sorted(preds, key=lambda p: float(p.get('confidence', 0.0)), reverse=True)
        
        # One-to-one matching mask for this image's ground truths
        matched_gt = set()
        negatives = [g for g in gt_boxes if g.get('is_negative_example')]
        
        for p in preds_sorted:
            cid = p['class_id']
            if cid not in stats_by_class:
                continue
            
            p_box = p['bbox']
            conf = float(p.get('confidence', 0.0))
            
            # Check hard negative hit (lane paint, puddle, manhole, shadow)
            hit_neg = any(box_iou(p_box, n['bbox']) >= 0.3 for n in negatives)
            if hit_neg:
                hard_negative_fps += 1

            # Match against unmatched valid ground-truth boxes of identical class
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, g in enumerate(gt_boxes):
                if g.get('is_negative_example') or g.get('uncertain'):
                    continue
                if g['class_id'] == cid and g_idx not in matched_gt:
                    iou = box_iou(p_box, g['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                        
            # One-to-one verification: each ground truth box can be matched by at most one prediction
            if best_iou >= iou_thresh and best_gt_idx >= 0:
                stats_by_class[cid]['tp'] += 1
                stats_by_class[cid]['predictions'].append((conf, 1, 0)) # (confidence, tp, fp)
                matched_gt.add(best_gt_idx)
            else:
                stats_by_class[cid]['fp'] += 1
                stats_by_class[cid]['predictions'].append((conf, 0, 1))

        # Count false negatives (unmatched ground truth defects)
        for g_idx, g in enumerate(gt_boxes):
            if g.get('is_negative_example') or g.get('uncertain'):
                continue
            if g_idx not in matched_gt:
                cid = g['class_id']
                if cid in stats_by_class:
                    stats_by_class[cid]['fn'] += 1

    report = {
        'metric': f'AP{int(iou_thresh*100)}',
        'iou_threshold': iou_thresh,
        'per_class': {},
        'overall': {},
        'hard_negative_fps': hard_negative_fps
    }
    total_tp = 0; total_fp = 0; total_fn = 0
    all_aps = []

    for cid, name in DAMAGE_CLASSES.items():
        c_stats = stats_by_class[cid]
        tp = c_stats['tp']
        fp = c_stats['fp']
        fn = c_stats['fn']
        total_gt = c_stats['total_gt']
        
        # Sort all class predictions globally by confidence descending for AP computation
        preds_sorted = sorted(c_stats['predictions'], key=lambda x: x[0], reverse=True)
        tp_list = [x[1] for x in preds_sorted]
        fp_list = [x[2] for x in preds_sorted]
        
        ap = compute_ap50_class(tp_list, fp_list, total_gt)
        all_aps.append(ap)
        
        precision = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
        recall = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        report['per_class'][name] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'ground_truth_count': total_gt,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'ap50': round(ap, 4)
        }
        total_tp += tp; total_fp += fp; total_fn += fn

    overall_prec = total_tp / max(total_tp + total_fp, 1) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / max(total_tp + total_fn, 1) if (total_tp + total_fn) > 0 else 0.0
    mean_ap50 = float(np.mean(all_aps)) if all_aps else 0.0
    
    report['overall'] = {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': round(overall_prec, 4),
        'recall': round(overall_rec, 4),
        'f1': round(2 * overall_prec * overall_rec / max(overall_prec + overall_rec, 1e-6), 4),
        'map50': round(mean_ap50, 4)
    }
    return report

def load_evaluation_dataset(dataset_path: Path):
    """Load JSON/YOLO formatted evaluation dataset with recording splits."""
    if not dataset_path.exists():
        raise FileNotFoundError(f'Evaluation dataset not found: {dataset_path}')
    return json.loads(dataset_path.read_text(encoding='utf-8'))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python -m backend.modules.road_damage.evaluation <ground_truth.json> <predictions.json> [--iou 0.5]")
        sys.exit(1)
    gt_file = Path(sys.argv[1])
    pred_file = Path(sys.argv[2])
    iou = 0.5
    if '--iou' in sys.argv:
        idx = sys.argv.index('--iou')
        iou = float(sys.argv[idx + 1])
    gt_data = json.loads(gt_file.read_text(encoding='utf-8'))
    pred_data = json.loads(pred_file.read_text(encoding='utf-8'))
    res = evaluate_dataset(gt_data, pred_data, iou_thresh=iou)
    print(json.dumps(res, indent=2))
