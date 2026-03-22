"""Soft-NMS and Hard-NMS implementations for post-processing detections.

Soft-NMS (Bodla et al., 2017) decays scores of overlapping detections
instead of hard-suppressing them, which is especially beneficial in
dense scenes where objects overlap significantly.

Two decay methods are supported:
    Gaussian: s_i *= exp(-iou^2 / sigma)
    Linear:   s_i *= (1 - iou)  if iou > threshold

References:
    Bodla et al. "Soft-NMS -- Improving Object Detection With One Line of Code"
    https://arxiv.org/abs/1704.04503
"""

from typing import Tuple

import torch


def _compute_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Compute IoU between a single box and a set of boxes.

    Args:
        box: Single box tensor of shape (4,) as [x1, y1, x2, y2].
        boxes: Set of boxes tensor of shape (N, 4) as [x1, y1, x2, y2].

    Returns:
        IoU values tensor of shape (N,).
    """
    # Intersection coordinates
    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Union
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection

    return intersection / (union + 1e-6)


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    method: str = 'gaussian',
    iou_threshold: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Soft-NMS to a set of detections.

    Instead of hard-suppressing overlapping detections, Soft-NMS decays
    their scores based on IoU overlap with the highest-scoring detection.
    This preserves detections in crowded scenes where objects overlap.

    The algorithm iteratively selects the highest-scoring detection, then
    decays scores of all remaining detections based on their IoU with the
    selected detection. Detections with decayed scores below the threshold
    are removed.

    Args:
        boxes: Bounding boxes (N, 4) as [x1, y1, x2, y2].
        scores: Confidence scores (N,).
        sigma: Gaussian decay parameter. Lower = more aggressive suppression.
        score_threshold: Minimum score to keep after decay.
        method: Decay method, 'gaussian' or 'linear'.
        iou_threshold: IoU threshold for linear method.

    Returns:
        Tuple of:
            - kept_boxes: Filtered boxes (K, 4).
            - kept_scores: Filtered/decayed scores (K,).
            - keep_indices: Original indices of kept detections (K,).
    """
    if boxes.numel() == 0:
        return (
            boxes,
            scores,
            torch.zeros(0, dtype=torch.long, device=boxes.device),
        )

    device = boxes.device
    boxes_work = boxes.clone()
    scores_work = scores.clone()
    n = boxes_work.size(0)

    # Track original indices
    original_indices = torch.arange(n, device=device)

    kept_boxes = []
    kept_scores = []
    kept_indices = []

    for _ in range(n):
        # Find the detection with the maximum score
        max_idx = scores_work.argmax()
        max_score = scores_work[max_idx]

        if max_score < score_threshold:
            break

        # Record this detection with its current (possibly decayed) score
        kept_boxes.append(boxes_work[max_idx].clone())
        kept_scores.append(max_score.clone())
        kept_indices.append(original_indices[max_idx].item())

        # Compute IoU of the selected box with all remaining
        ious = _compute_iou(boxes_work[max_idx], boxes_work)

        # Decay scores based on IoU
        if method == 'gaussian':
            decay = torch.exp(-(ious ** 2) / sigma)
        elif method == 'linear':
            decay = torch.ones_like(ious)
            high_iou = ious > iou_threshold
            decay[high_iou] = 1.0 - ious[high_iou]
        else:
            raise ValueError(f"Unknown method: {method}. Use 'gaussian' or 'linear'.")

        scores_work = scores_work * decay

        # Set the selected detection's score to 0 so it is not selected again
        scores_work[max_idx] = 0.0

    if len(kept_boxes) == 0:
        return (
            torch.zeros(0, 4, device=device),
            torch.zeros(0, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        )

    return (
        torch.stack(kept_boxes),
        torch.stack(kept_scores),
        torch.tensor(kept_indices, dtype=torch.long, device=device),
    )


def hard_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply standard (hard) Non-Maximum Suppression.

    Greedily selects the highest-scoring detection and suppresses all
    remaining detections with IoU above the threshold. Useful as a
    comparison baseline against Soft-NMS.

    Args:
        boxes: Bounding boxes (N, 4) as [x1, y1, x2, y2].
        scores: Confidence scores (N,).
        iou_threshold: IoU threshold above which detections are suppressed.

    Returns:
        Tuple of:
            - kept_boxes: Filtered boxes (K, 4).
            - kept_scores: Filtered scores (K,).
            - keep_indices: Original indices of kept detections (K,).
    """
    if boxes.numel() == 0:
        return (
            boxes,
            scores,
            torch.zeros(0, dtype=torch.long, device=boxes.device),
        )

    device = boxes.device
    original_boxes = boxes.clone()
    original_scores = scores.clone()

    # Sort by score descending
    order = scores.argsort(descending=True)

    keep = []

    while order.numel() > 0:
        idx = order[0].item()
        keep.append(idx)

        if order.numel() == 1:
            break

        # IoU of current best with remaining
        remaining = order[1:]
        ious = _compute_iou(original_boxes[idx], original_boxes[remaining])

        # Keep only those with IoU below threshold
        mask = ious <= iou_threshold
        order = remaining[mask]

    keep_indices = torch.tensor(keep, dtype=torch.long, device=device)
    return original_boxes[keep_indices], original_scores[keep_indices], keep_indices


def batched_soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    method: str = 'gaussian',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Soft-NMS independently per class.

    Processes each class separately and concatenates results, ensuring
    that suppression only occurs between detections of the same class.

    Args:
        boxes: Bounding boxes (N, 4) as [x1, y1, x2, y2].
        scores: Confidence scores (N,).
        labels: Class labels (N,).
        sigma: Gaussian Soft-NMS sigma.
        score_threshold: Minimum score after decay.
        method: Decay method, 'gaussian' or 'linear'.

    Returns:
        Tuple of:
            - kept_boxes: Filtered boxes (K, 4).
            - kept_scores: Filtered scores (K,).
            - kept_labels: Labels for kept detections (K,).
            - keep_indices: Original indices of kept detections (K,).
    """
    device = boxes.device
    unique_labels = labels.unique()

    all_boxes = []
    all_scores = []
    all_labels = []
    all_indices = []

    for cls in unique_labels:
        cls_mask = labels == cls
        cls_indices = cls_mask.nonzero(as_tuple=False).squeeze(1)
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        kept_boxes, kept_scores, keep_idx = soft_nms(
            cls_boxes, cls_scores, sigma=sigma,
            score_threshold=score_threshold, method=method,
        )

        if kept_boxes.size(0) > 0:
            all_boxes.append(kept_boxes)
            all_scores.append(kept_scores)
            all_labels.append(
                torch.full((kept_boxes.size(0),), cls.item(),
                           dtype=torch.long, device=device)
            )
            all_indices.append(cls_indices[keep_idx])

    if len(all_boxes) == 0:
        return (
            torch.zeros(0, 4, device=device),
            torch.zeros(0, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        )

    return (
        torch.cat(all_boxes),
        torch.cat(all_scores),
        torch.cat(all_labels),
        torch.cat(all_indices),
    )


if __name__ == '__main__':
    print("=" * 60)
    print("Soft-NMS vs Hard-NMS Comparison")
    print("=" * 60)

    # Create test scenario with overlapping boxes
    # Two clusters of near-duplicate detections
    boxes = torch.tensor([
        [0.10, 0.10, 0.50, 0.50],  # Object 1, high score
        [0.12, 0.12, 0.52, 0.52],  # Near-duplicate of Object 1
        [0.15, 0.15, 0.55, 0.55],  # Another near-duplicate
        [0.60, 0.60, 0.90, 0.90],  # Object 2, well separated
        [0.62, 0.62, 0.92, 0.92],  # Near-duplicate of Object 2
    ], dtype=torch.float32)

    scores = torch.tensor([0.95, 0.80, 0.70, 0.85, 0.65], dtype=torch.float32)

    print(f"\nInput boxes ({boxes.size(0)} detections):")
    for i in range(boxes.size(0)):
        print(f"  Box {i}: {boxes[i].tolist()}, score: {scores[i]:.2f}")

    # Hard NMS
    print("\n--- Hard NMS (iou_threshold=0.5) ---")
    hard_boxes, hard_scores, hard_idx = hard_nms(boxes, scores, iou_threshold=0.5)
    print(f"Kept {hard_boxes.size(0)} detections:")
    for i in range(hard_boxes.size(0)):
        print(f"  Box {hard_idx[i].item()}: {hard_boxes[i].tolist()}, score: {hard_scores[i]:.2f}")

    # Soft NMS - Gaussian
    print("\n--- Soft NMS (Gaussian, sigma=0.5) ---")
    soft_boxes_g, soft_scores_g, soft_idx_g = soft_nms(
        boxes, scores, sigma=0.5, score_threshold=0.1, method='gaussian'
    )
    print(f"Kept {soft_boxes_g.size(0)} detections:")
    for i in range(soft_boxes_g.size(0)):
        print(f"  Box {soft_idx_g[i].item()}: score: {soft_scores_g[i]:.4f}")

    # Soft NMS - Linear
    print("\n--- Soft NMS (Linear, iou_threshold=0.3) ---")
    soft_boxes_l, soft_scores_l, soft_idx_l = soft_nms(
        boxes, scores, score_threshold=0.1, method='linear', iou_threshold=0.3
    )
    print(f"Kept {soft_boxes_l.size(0)} detections:")
    for i in range(soft_boxes_l.size(0)):
        print(f"  Box {soft_idx_l[i].item()}: score: {soft_scores_l[i]:.4f}")

    # Test batched soft NMS
    print("\n--- Batched Soft NMS (per-class) ---")
    labels = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    b_boxes, b_scores, b_labels, b_idx = batched_soft_nms(
        boxes, scores, labels, sigma=0.5, score_threshold=0.1
    )
    print(f"Kept {b_boxes.size(0)} detections:")
    for i in range(b_boxes.size(0)):
        print(f"  Box {b_idx[i].item()}: class={b_labels[i].item()}, score: {b_scores[i]:.4f}")

    # Edge case: empty input
    print("\n--- Edge case: empty input ---")
    empty_b, empty_s, empty_i = soft_nms(
        torch.zeros(0, 4), torch.zeros(0), sigma=0.5
    )
    print(f"Empty input result: boxes={empty_b.shape}, scores={empty_s.shape}")
