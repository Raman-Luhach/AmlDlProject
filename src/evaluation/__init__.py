"""Evaluation suite for YOLACT object detection.

Provides COCO-style mAP evaluation and low-level detection metrics,
implemented entirely in NumPy (no pycocotools dependency).
"""

from src.evaluation.evaluator import COCOEvaluator
from src.evaluation.metrics import (
    average_precision,
    compute_detection_metrics,
    compute_iou_matrix,
    match_predictions_single_image,
    precision_recall_curve,
)

__all__ = [
    'COCOEvaluator',
    'average_precision',
    'compute_detection_metrics',
    'compute_iou_matrix',
    'match_predictions_single_image',
    'precision_recall_curve',
]
