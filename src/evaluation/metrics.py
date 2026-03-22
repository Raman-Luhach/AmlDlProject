"""Low-level detection metrics computed with NumPy only (no pycocotools).

Provides IoU computation, precision-recall curves, average precision,
and a convenience function that wraps everything together.

All box inputs are expected in [x1, y1, x2, y2] format.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_iou_matrix(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """Compute pairwise IoU between two sets of axis-aligned boxes.

    Args:
        boxes_a: (M, 4) array of boxes in [x1, y1, x2, y2] format.
        boxes_b: (N, 4) array of boxes in [x1, y1, x2, y2] format.

    Returns:
        (M, N) IoU matrix where element (i, j) is IoU(boxes_a[i], boxes_b[j]).
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    # Intersection coordinates
    x1 = np.maximum(boxes_a[:, 0][:, None], boxes_b[:, 0][None, :])
    y1 = np.maximum(boxes_a[:, 1][:, None], boxes_b[:, 1][None, :])
    x2 = np.minimum(boxes_a[:, 2][:, None], boxes_b[:, 2][None, :])
    y2 = np.minimum(boxes_a[:, 3][:, None], boxes_b[:, 3][None, :])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter_area
    iou = np.where(union > 0, inter_area / union, 0.0)

    return iou


def precision_recall_curve(
    scores: np.ndarray,
    tp_fp: np.ndarray,
    num_gt: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute precision-recall curve from sorted detection results.

    Detections must be pre-sorted by descending score.

    Args:
        scores: (N,) confidence scores (descending order).
        tp_fp: (N,) binary array where 1 = true positive, 0 = false positive.
        num_gt: Total number of ground-truth boxes for this class.

    Returns:
        recalls: (N,) recall values at each detection.
        precisions: (N,) precision values at each detection.
    """
    if num_gt == 0:
        return np.zeros(len(tp_fp), dtype=np.float64), np.zeros(len(tp_fp), dtype=np.float64)

    tp_cumsum = np.cumsum(tp_fp)
    fp_cumsum = np.cumsum(1 - tp_fp)

    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    return recalls, precisions


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute average precision using all-point interpolation (COCO style).

    The precision at each recall level is replaced by the maximum precision
    at any recall >= that level, and AP is the area under this monotonically
    decreasing envelope.

    Args:
        recalls: (N,) recall values (non-decreasing).
        precisions: (N,) precision values.

    Returns:
        AP value as a float in [0, 1].
    """
    if len(recalls) == 0:
        return 0.0

    # Prepend (0, 1) and append (1, 0) sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Make precision monotonically decreasing (right to left)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    change_indices = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum the rectangular areas
    ap = np.sum((mrec[change_indices + 1] - mrec[change_indices]) * mpre[change_indices + 1])

    return float(ap)


def match_predictions_single_image(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Match predictions to ground-truth boxes for a single image.

    Predictions are matched greedily in descending score order. Each GT box
    can be matched at most once.

    Args:
        pred_boxes: (N, 4) predicted boxes in [x1, y1, x2, y2].
        pred_scores: (N,) confidence scores.
        gt_boxes: (M, 4) ground-truth boxes in [x1, y1, x2, y2].
        iou_threshold: Minimum IoU to consider a match.

    Returns:
        tp_fp: (N,) binary array, 1 = true positive, 0 = false positive.
    """
    num_preds = len(pred_boxes)
    num_gt = len(gt_boxes)

    tp_fp = np.zeros(num_preds, dtype=np.float64)

    if num_preds == 0:
        return tp_fp

    if num_gt == 0:
        return tp_fp  # All false positives

    # Sort predictions by score descending
    sorted_idx = np.argsort(-pred_scores)

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)  # (N, M)

    gt_matched = np.zeros(num_gt, dtype=bool)

    for idx in sorted_idx:
        ious = iou_matrix[idx]  # (M,)

        # Find best matching GT (highest IoU)
        best_gt_idx = np.argmax(ious)
        best_iou = ious[best_gt_idx]

        if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
            tp_fp[idx] = 1.0
            gt_matched[best_gt_idx] = True

    return tp_fp


def compute_detection_metrics(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, np.ndarray]],
    iou_thresholds: Optional[List[float]] = None,
    max_detections: int = 300,
) -> Dict[str, float]:
    """Compute detection metrics across a dataset at specified IoU thresholds.

    This is a convenience wrapper that computes AP, AR, and F1 at given IoU
    thresholds for a single class (SKU-110K scenario).

    Args:
        predictions: List of dicts per image, each with keys:
            - 'boxes': (N, 4) np.ndarray in [x1, y1, x2, y2]
            - 'scores': (N,) np.ndarray
            - 'labels': (N,) np.ndarray  (optional, ignored for single-class)
        ground_truths: List of dicts per image, each with keys:
            - 'boxes': (M, 4) np.ndarray in [x1, y1, x2, y2]
            - 'labels': (M,) np.ndarray  (optional)
        iou_thresholds: List of IoU thresholds for AP computation.
            Defaults to [0.5].
        max_detections: Maximum number of detections per image to consider.

    Returns:
        Dictionary containing:
            - 'AP@{t}': AP at each IoU threshold t
            - 'AP_mean': Mean AP across all thresholds
            - 'AR_mean': Mean AR across all thresholds
            - 'F1_mean': Harmonic mean of AP_mean and AR_mean
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    num_images = len(predictions)
    results: Dict[str, float] = {}

    ap_values = []
    ar_values = []

    for iou_thresh in iou_thresholds:
        # Gather all detections across images with image index
        all_scores = []
        all_tp_fp = []
        total_gt = 0

        for img_idx in range(num_images):
            pred = predictions[img_idx]
            gt = ground_truths[img_idx]

            pred_boxes = np.asarray(pred.get('boxes', np.zeros((0, 4))), dtype=np.float64)
            pred_scores = np.asarray(pred.get('scores', np.zeros(0)), dtype=np.float64)
            gt_boxes = np.asarray(gt.get('boxes', np.zeros((0, 4))), dtype=np.float64)

            # Limit detections
            if len(pred_scores) > max_detections:
                top_k_idx = np.argsort(-pred_scores)[:max_detections]
                pred_boxes = pred_boxes[top_k_idx]
                pred_scores = pred_scores[top_k_idx]

            total_gt += len(gt_boxes)

            tp_fp = match_predictions_single_image(
                pred_boxes, pred_scores, gt_boxes, iou_thresh
            )

            all_scores.append(pred_scores)
            all_tp_fp.append(tp_fp)

        if total_gt == 0:
            results[f'AP@{iou_thresh:.2f}'] = 0.0
            ap_values.append(0.0)
            ar_values.append(0.0)
            continue

        # Concatenate across images and sort by score
        all_scores = np.concatenate(all_scores)
        all_tp_fp = np.concatenate(all_tp_fp)

        sorted_idx = np.argsort(-all_scores)
        all_scores = all_scores[sorted_idx]
        all_tp_fp = all_tp_fp[sorted_idx]

        recalls, precisions = precision_recall_curve(all_scores, all_tp_fp, total_gt)

        ap = average_precision(recalls, precisions)
        ar = float(recalls[-1]) if len(recalls) > 0 else 0.0

        results[f'AP@{iou_thresh:.2f}'] = ap
        ap_values.append(ap)
        ar_values.append(ar)

    results['AP_mean'] = float(np.mean(ap_values)) if ap_values else 0.0
    results['AR_mean'] = float(np.mean(ar_values)) if ar_values else 0.0

    ap_m = results['AP_mean']
    ar_m = results['AR_mean']
    results['F1_mean'] = (
        2.0 * ap_m * ar_m / (ap_m + ar_m) if (ap_m + ar_m) > 0 else 0.0
    )

    return results


if __name__ == '__main__':
    # Quick self-test with synthetic data
    print("=== Detection Metrics Self-Test ===\n")

    rng = np.random.RandomState(42)

    # Create synthetic predictions and ground truths
    preds = []
    gts = []
    for _ in range(10):
        n_gt = rng.randint(5, 20)
        gt_boxes = np.sort(rng.uniform(0, 500, (n_gt, 4)).reshape(n_gt, 2, 2), axis=2).reshape(n_gt, 4)
        gt_boxes[:, 2] = gt_boxes[:, 0] + rng.uniform(20, 80, n_gt)
        gt_boxes[:, 3] = gt_boxes[:, 1] + rng.uniform(20, 80, n_gt)

        # Add some noise to create predictions
        n_pred = rng.randint(5, 25)
        pred_boxes = gt_boxes[:min(n_pred, n_gt)].copy()
        pred_boxes += rng.normal(0, 5, pred_boxes.shape)
        if n_pred > n_gt:
            extra = rng.uniform(0, 500, (n_pred - n_gt, 4))
            extra[:, 2] = extra[:, 0] + rng.uniform(20, 80, n_pred - n_gt)
            extra[:, 3] = extra[:, 1] + rng.uniform(20, 80, n_pred - n_gt)
            pred_boxes = np.vstack([pred_boxes, extra])

        pred_scores = rng.uniform(0.1, 1.0, len(pred_boxes))

        preds.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': np.ones(len(pred_boxes), dtype=np.int64),
        })
        gts.append({
            'boxes': gt_boxes,
            'labels': np.ones(n_gt, dtype=np.int64),
        })

    metrics = compute_detection_metrics(
        preds, gts,
        iou_thresholds=[0.5, 0.75],
    )

    print("Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Test IoU matrix
    a = np.array([[0, 0, 10, 10]], dtype=np.float64)
    b = np.array([[5, 5, 15, 15]], dtype=np.float64)
    iou = compute_iou_matrix(a, b)
    expected_iou = 25.0 / (100 + 100 - 25)
    print(f"\nIoU test: computed={iou[0, 0]:.4f}, expected={expected_iou:.4f}")
    assert abs(iou[0, 0] - expected_iou) < 1e-6, "IoU mismatch!"

    print("\n=== All tests passed ===")
