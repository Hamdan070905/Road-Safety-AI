import pytest
from backend.modules.road_damage.adapter import predict, nms_boxes
from backend.modules.road_damage.service import associate, draw
from backend.modules.road_damage.evaluation import evaluate_dataset, box_iou, compute_ap50_class
import numpy as np

def test_box_iou():
    b1 = [0, 0, 10, 10]
    b2 = [0, 0, 10, 10]
    assert box_iou(b1, b2) == pytest.approx(1.0)
    b3 = [10, 10, 20, 20]
    assert box_iou(b1, b3) == 0.0

def test_nms_boxes():
    dets = [
        {'class_id': 3, 'confidence': 0.8, 'bbox': [10, 10, 50, 50]},
        {'class_id': 3, 'confidence': 0.6, 'bbox': [12, 12, 48, 48]}, # Duplicate
        {'class_id': 3, 'confidence': 0.5, 'bbox': [100, 100, 150, 150]}, # Separate pothole
        {'class_id': 0, 'confidence': 0.7, 'bbox': [10, 10, 50, 50]}, # Different class
    ]
    kept = nms_boxes(dets, iou_threshold=0.4)
    assert len(kept) == 3
    # Both class 3 separate defects and class 0 defect are kept
    assert {d['class_id'] for d in kept} == {0, 3}

def test_temporal_association_candidate_clustering():
    # Simulated pothole approached by vehicle over 3 frames (grows, shifts downwards)
    obs = [
        {'id': '1', 'timestamp_s': 1.0, 'class_id': 3, 'class_name': 'pothole', 'confidence': 0.35, 'bbox': [200, 150, 250, 180], '_shape': (720, 1280)},
        {'id': '2', 'timestamp_s': 1.5, 'class_id': 3, 'class_name': 'pothole', 'confidence': 0.42, 'bbox': [195, 200, 260, 240], '_shape': (720, 1280)},
        # Distinct nearby pothole at same time
        {'id': '3', 'timestamp_s': 1.5, 'class_id': 3, 'class_name': 'pothole', 'confidence': 0.38, 'bbox': [500, 200, 560, 240], '_shape': (720, 1280)},
    ]
    associated = associate(obs, max_gap_s=1.0)
    # The first two should share candidate_id, the third should get a different one
    assert associated[0]['candidate_id'] == associated[1]['candidate_id']
    assert associated[0]['candidate_id'] != associated[2]['candidate_id']

def test_evaluation_workflow_one_to_one_and_ranking():
    gt = {
        'frame_positive': [
            {'class_id': 3, 'bbox': [10, 10, 50, 50]}, # true pothole 1
            {'class_id': 0, 'bbox': [100, 100, 120, 200], 'is_negative_example': True}, # lane paint hard negative
        ],
        'frame_negative': [
            # Negative image: normal road, no defects
        ]
    }
    preds = {
        'frame_positive': [
            {'class_id': 3, 'bbox': [12, 12, 48, 48], 'confidence': 0.85}, # TP match
            {'class_id': 3, 'bbox': [14, 14, 49, 49], 'confidence': 0.65}, # Duplicate prediction on same target -> must be FP!
            {'class_id': 0, 'bbox': [102, 102, 118, 198], 'confidence': 0.30}, # FP hitting lane paint
        ],
        'frame_negative': [
            {'class_id': 3, 'bbox': [50, 50, 80, 80], 'confidence': 0.20}, # FP on negative/empty image
        ]
    }
    report = evaluate_dataset(gt, preds, iou_thresh=0.5)
    
    # Check one-to-one matching: 1 TP, 1 duplicate FP on frame_positive, 1 FP on frame_negative -> total TP=1, total FP=2
    assert report['per_class']['pothole']['tp'] == 1
    assert report['per_class']['pothole']['fp'] == 2
    assert report['per_class']['pothole']['fn'] == 0
    assert report['per_class']['pothole']['ground_truth_count'] == 1
    
    # Check AP computation from ranked predictions
    # Predictions for pothole: (0.85 -> TP), (0.65 -> FP), (0.20 -> FP)
    # Recall steps: at 0.85 -> rec=1.0, prec=1.0. At 0.65 -> rec=1.0, prec=0.5. At 0.20 -> rec=1.0, prec=0.333.
    # Max precision across all recalls 0..1 is 1.0, so AP50 = 1.0
    assert report['per_class']['pothole']['ap50'] == 1.0
    
    # Hard negative correctly flagged
    assert report['hard_negative_fps'] == 1
    assert report['metric'] == 'AP50'

def test_initial_inference_threshold_does_not_discard_lower_per_class():
    class MockModel:
        names = {0: 'longitudinal_crack', 3: 'pothole'}
        def predict(self, *args, **kwargs):
            from types import SimpleNamespace
            import torch
            # Mock returns confidence passed into predict
            conf_used = kwargs.get('conf')
            # Emulate returning a box with conf 0.18
            boxes = SimpleNamespace(
                xyxy=torch.tensor([[10., 10., 50., 50.]]),
                cls=torch.tensor([3.]),
                conf=torch.tensor([0.18])
            )
            return [SimpleNamespace(boxes=boxes)]

    mock = MockModel()
    # Global confidence is 0.25, but per-class threshold for pothole is 0.18
    res = predict(mock, np.zeros((100, 100, 3), dtype=np.uint8), confidence=0.25, per_class_conf={'pothole': 0.18, 'longitudinal_crack': 0.25})
    assert len(res) == 1
    assert res[0]['class_id'] == 3
    assert res[0]['confidence'] == pytest.approx(0.18, abs=1e-4)
