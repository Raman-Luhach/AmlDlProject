"""COCO-style mAP evaluator implemented with NumPy only.

Follows the COCO evaluation protocol:
  - For each IoU threshold in [0.50, 0.55, ..., 0.95], compute per-class
    precision-recall curves and the corresponding AP.
  - Report AP@0.5, AP@0.75, AP@[.5:.95], and AR at max detections 1/10/100/300.

This implementation does NOT depend on pycocotools.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.evaluation.metrics import (
    average_precision,
    compute_iou_matrix,
    match_predictions_single_image,
    precision_recall_curve,
)


class COCOEvaluator:
    """COCO-style mAP evaluation for object detection.

    Computes AP and AR at multiple IoU thresholds and max-detection limits,
    following the standard COCO evaluation protocol.

    Args:
        iou_thresholds: IoU thresholds at which to evaluate.
            Defaults to [0.50, 0.55, ..., 0.95].
        max_detections: List of max-detection limits for AR computation.
            Defaults to [1, 10, 100, 300].
    """

    def __init__(
        self,
        iou_thresholds: Optional[List[float]] = None,
        max_detections: Optional[List[int]] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            iou_thresholds: IoU thresholds for AP computation.
            max_detections: Maximum detection counts for AR computation.
        """
        if iou_thresholds is None:
            self.iou_thresholds = [
                round(0.5 + i * 0.05, 2) for i in range(10)
            ]  # [0.50, 0.55, ..., 0.95]
        else:
            self.iou_thresholds = sorted(iou_thresholds)

        if max_detections is None:
            self.max_detections = [1, 10, 100, 300]
        else:
            self.max_detections = sorted(max_detections)

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Run full COCO-style evaluation.

        Args:
            predictions: List of dicts per image, each with keys:
                - 'boxes': array-like (N, 4) in [x1, y1, x2, y2]
                - 'scores': array-like (N,)
                - 'labels': array-like (N,)  (class indices)
            ground_truths: List of dicts per image, each with keys:
                - 'boxes': array-like (M, 4) in [x1, y1, x2, y2]
                - 'labels': array-like (M,)

        Returns:
            Dictionary containing:
                - 'AP@0.50'          : AP at IoU=0.50
                - 'AP@0.75'          : AP at IoU=0.75
                - 'AP@[.50:.95]'     : Mean AP across all IoU thresholds
                - 'AR@1'             : Average recall with max 1 detection/image
                - 'AR@10'            : Average recall with max 10 detections/image
                - 'AR@100'           : Average recall with max 100 detections/image
                - 'AR@300'           : Average recall with max 300 detections/image
                - Per-threshold AP   : 'AP@{iou:.2f}' for each threshold
        """
        num_images = len(predictions)
        assert num_images == len(ground_truths), (
            f"Predictions ({num_images}) and ground truths ({len(ground_truths)}) "
            "must have the same number of images."
        )

        # Convert everything to numpy
        preds_np = self._to_numpy(predictions)
        gts_np = self._to_numpy(ground_truths)

        # Compute AP at each IoU threshold
        ap_per_threshold: Dict[float, float] = {}
        for iou_thresh in self.iou_thresholds:
            ap = self._compute_ap_at_threshold(preds_np, gts_np, iou_thresh)
            ap_per_threshold[iou_thresh] = ap

        # Compute AR at each max-detection limit
        ar_per_maxdet: Dict[int, float] = {}
        for max_det in self.max_detections:
            ar = self._compute_ar_at_maxdet(preds_np, gts_np, max_det)
            ar_per_maxdet[max_det] = ar

        # Assemble results
        results: Dict[str, float] = {}

        # Per-threshold AP
        for iou_thresh, ap in ap_per_threshold.items():
            results[f'AP@{iou_thresh:.2f}'] = ap

        # Standard COCO summary metrics
        results['AP@0.50'] = ap_per_threshold.get(0.5, 0.0)
        results['AP@0.75'] = ap_per_threshold.get(0.75, 0.0)
        results['AP@[.50:.95]'] = float(np.mean(list(ap_per_threshold.values())))

        for max_det, ar in ar_per_maxdet.items():
            results[f'AR@{max_det}'] = ar

        return results

    def _compute_ap_at_threshold(
        self,
        predictions: List[Dict[str, np.ndarray]],
        ground_truths: List[Dict[str, np.ndarray]],
        iou_threshold: float,
    ) -> float:
        """Compute AP at a single IoU threshold across all images.

        Collects all detections across images, sorts globally by score,
        computes a dataset-wide precision-recall curve, and returns the AP.

        Args:
            predictions: Per-image prediction dicts (numpy arrays).
            ground_truths: Per-image ground-truth dicts (numpy arrays).
            iou_threshold: IoU threshold for matching.

        Returns:
            AP value.
        """
        all_scores: List[np.ndarray] = []
        all_tp_fp: List[np.ndarray] = []
        total_gt = 0

        for img_idx in range(len(predictions)):
            pred = predictions[img_idx]
            gt = ground_truths[img_idx]

            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            gt_boxes = gt['boxes']

            total_gt += len(gt_boxes)

            tp_fp = match_predictions_single_image(
                pred_boxes, pred_scores, gt_boxes, iou_threshold
            )

            all_scores.append(pred_scores)
            all_tp_fp.append(tp_fp)

        if total_gt == 0:
            return 0.0

        all_scores_arr = np.concatenate(all_scores) if all_scores else np.zeros(0)
        all_tp_fp_arr = np.concatenate(all_tp_fp) if all_tp_fp else np.zeros(0)

        if len(all_scores_arr) == 0:
            return 0.0

        # Sort globally by score descending
        sorted_idx = np.argsort(-all_scores_arr)
        all_scores_arr = all_scores_arr[sorted_idx]
        all_tp_fp_arr = all_tp_fp_arr[sorted_idx]

        recalls, precisions = precision_recall_curve(
            all_scores_arr, all_tp_fp_arr, total_gt
        )

        return average_precision(recalls, precisions)

    def _compute_ar_at_maxdet(
        self,
        predictions: List[Dict[str, np.ndarray]],
        ground_truths: List[Dict[str, np.ndarray]],
        max_det: int,
    ) -> float:
        """Compute average recall at a given max-detection limit.

        For each IoU threshold and each image, computes the recall when
        keeping at most max_det detections (by highest score). The result
        is the mean recall across all images and all IoU thresholds.

        Args:
            predictions: Per-image prediction dicts.
            ground_truths: Per-image ground-truth dicts.
            max_det: Maximum number of detections per image.

        Returns:
            Average recall value.
        """
        recall_values: List[float] = []

        for iou_thresh in self.iou_thresholds:
            for img_idx in range(len(predictions)):
                pred = predictions[img_idx]
                gt = ground_truths[img_idx]

                pred_boxes = pred['boxes']
                pred_scores = pred['scores']
                gt_boxes = gt['boxes']

                num_gt = len(gt_boxes)
                if num_gt == 0:
                    continue  # Skip images with no GT for recall

                # Limit detections
                if len(pred_scores) > max_det:
                    top_k = np.argsort(-pred_scores)[:max_det]
                    pred_boxes = pred_boxes[top_k]
                    pred_scores = pred_scores[top_k]

                tp_fp = match_predictions_single_image(
                    pred_boxes, pred_scores, gt_boxes, iou_thresh
                )

                recall = float(np.sum(tp_fp)) / num_gt
                recall_values.append(recall)

        return float(np.mean(recall_values)) if recall_values else 0.0

    def compute_ap(
        self, recalls: np.ndarray, precisions: np.ndarray
    ) -> float:
        """Compute AP from a precision-recall curve.

        Delegates to the all-point interpolation implementation in metrics.py.

        Args:
            recalls: (N,) recall values.
            precisions: (N,) precision values.

        Returns:
            AP value.
        """
        return average_precision(recalls, precisions)

    def match_predictions(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        iou_threshold: float,
    ) -> np.ndarray:
        """Match predictions to GT for a single image.

        Delegates to metrics.match_predictions_single_image.

        Args:
            pred_boxes: (N, 4) predicted boxes.
            pred_scores: (N,) confidence scores.
            gt_boxes: (M, 4) ground-truth boxes.
            iou_threshold: Minimum IoU for a match.

        Returns:
            (N,) binary TP/FP array.
        """
        return match_predictions_single_image(
            pred_boxes, pred_scores, gt_boxes, iou_threshold
        )

    @staticmethod
    def _to_numpy(
        data: List[Dict[str, Any]],
    ) -> List[Dict[str, np.ndarray]]:
        """Convert tensor/list fields to numpy arrays.

        Args:
            data: List of dicts with array-like values.

        Returns:
            Same structure with all values as numpy arrays.
        """
        converted = []
        for d in data:
            entry: Dict[str, np.ndarray] = {}
            for k, v in d.items():
                if hasattr(v, 'cpu'):
                    # PyTorch tensor
                    entry[k] = v.detach().cpu().numpy()
                elif isinstance(v, np.ndarray):
                    entry[k] = v
                elif isinstance(v, (list, tuple)):
                    entry[k] = np.asarray(v)
                else:
                    entry[k] = np.asarray(v)
            # Ensure required keys exist with proper shapes
            if 'boxes' not in entry:
                entry['boxes'] = np.zeros((0, 4), dtype=np.float64)
            if 'scores' not in entry:
                entry['scores'] = np.zeros(0, dtype=np.float64)
            if 'labels' not in entry:
                entry['labels'] = np.zeros(0, dtype=np.int64)
            converted.append(entry)
        return converted

    def print_results(self, results: Dict[str, float]) -> None:
        """Print evaluation results in a formatted table.

        Args:
            results: Results dictionary from evaluate().
        """
        print("\n" + "=" * 60)
        print("COCO-Style Evaluation Results")
        print("=" * 60)

        # Summary metrics
        summary_keys = ['AP@[.50:.95]', 'AP@0.50', 'AP@0.75']
        print("\n  Average Precision (AP):")
        for k in summary_keys:
            if k in results:
                print(f"    {k:<20s} = {results[k]:.4f}")

        # AR metrics
        print("\n  Average Recall (AR):")
        for max_det in self.max_detections:
            key = f'AR@{max_det}'
            if key in results:
                print(f"    {key:<20s} = {results[key]:.4f}")

        # Per-threshold AP
        print("\n  Per-Threshold AP:")
        for iou_thresh in self.iou_thresholds:
            key = f'AP@{iou_thresh:.2f}'
            if key in results:
                print(f"    {key:<20s} = {results[key]:.4f}")

        print("=" * 60)


if __name__ == '__main__':
    # Self-test with synthetic data
    print("=== COCOEvaluator Self-Test ===\n")

    rng = np.random.RandomState(42)

    # Generate synthetic ground truths and predictions
    predictions = []
    ground_truths = []

    for _ in range(20):
        n_gt = rng.randint(3, 15)
        gt_boxes = np.zeros((n_gt, 4), dtype=np.float64)
        gt_boxes[:, 0] = rng.uniform(0, 400, n_gt)
        gt_boxes[:, 1] = rng.uniform(0, 400, n_gt)
        gt_boxes[:, 2] = gt_boxes[:, 0] + rng.uniform(30, 100, n_gt)
        gt_boxes[:, 3] = gt_boxes[:, 1] + rng.uniform(30, 100, n_gt)

        # Some TPs (noisy copies of GT) and some FPs (random)
        n_tp = rng.randint(1, n_gt + 1)
        n_fp = rng.randint(0, 5)
        tp_boxes = gt_boxes[:n_tp].copy() + rng.normal(0, 3, (n_tp, 4))
        fp_boxes = np.zeros((n_fp, 4), dtype=np.float64)
        if n_fp > 0:
            fp_boxes[:, 0] = rng.uniform(0, 400, n_fp)
            fp_boxes[:, 1] = rng.uniform(0, 400, n_fp)
            fp_boxes[:, 2] = fp_boxes[:, 0] + rng.uniform(30, 100, n_fp)
            fp_boxes[:, 3] = fp_boxes[:, 1] + rng.uniform(30, 100, n_fp)

        pred_boxes = np.vstack([tp_boxes, fp_boxes])
        pred_scores = np.concatenate([
            rng.uniform(0.5, 1.0, n_tp),
            rng.uniform(0.1, 0.5, n_fp),
        ])

        predictions.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': np.ones(len(pred_boxes), dtype=np.int64),
        })
        ground_truths.append({
            'boxes': gt_boxes,
            'labels': np.ones(n_gt, dtype=np.int64),
        })

    # Evaluate
    evaluator = COCOEvaluator()
    results = evaluator.evaluate(predictions, ground_truths)
    evaluator.print_results(results)

    print("\n=== Self-test passed ===")
