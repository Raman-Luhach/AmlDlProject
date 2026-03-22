"""Loss functions for YOLACT training.

Provides:
    - FocalLoss: Focal loss for handling class imbalance in dense detection.
    - YOLACTLoss: Combined multi-task loss (classification + box regression + mask).

The YOLACT loss follows the original paper with three components:
    1. Classification: Focal loss on all anchors (pos + neg)
    2. Box regression: Smooth L1 on positive anchors only
    3. Mask segmentation: Binary cross-entropy on positive anchors only,
       comparing assembled prototype masks against ground truth masks.

Loss weights: cls=1.0, box=1.5, mask=6.125 (default from paper).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.anchors import encode_boxes, anchors_to_xyxy


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in dense detection.

    Focal loss down-weights well-classified examples and focuses training
    on hard negatives. This is critical for dense object detectors where
    background anchors vastly outnumber foreground anchors.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for the positive class. (1 - alpha) is used
               for the negative class. Default: 0.25.
        gamma: Focusing parameter that reduces loss for well-classified
               examples. gamma=0 is equivalent to cross-entropy. Default: 2.0.
        num_classes: Total number of classes including background. Default: 2.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            predictions: (N, num_classes) raw logits (not softmax/sigmoid).
            targets: (N,) class indices in [0, num_classes).

        Returns:
            Scalar focal loss averaged over the number of positive samples.
        """
        num_samples = predictions.size(0)
        if num_samples == 0:
            return predictions.sum() * 0.0

        # Convert logits to probabilities via softmax
        probs = F.softmax(predictions, dim=-1)  # (N, num_classes)

        # One-hot encode targets
        targets_one_hot = F.one_hot(
            targets, num_classes=self.num_classes
        ).float()  # (N, num_classes)

        # Gather the probability of the true class: p_t
        p_t = (probs * targets_one_hot).sum(dim=-1)  # (N,)
        p_t = p_t.clamp(min=1e-6, max=1.0 - 1e-6)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Compute alpha weight: alpha for positives, (1-alpha) for background
        # alpha_t = alpha for foreground classes, (1 - alpha) for background
        alpha_t = torch.where(
            targets > 0,
            torch.tensor(self.alpha, device=predictions.device),
            torch.tensor(1.0 - self.alpha, device=predictions.device),
        )

        # Focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        ce_loss = -torch.log(p_t)
        loss = alpha_t * focal_weight * ce_loss

        return loss.sum()


def compute_iou_matrix(
    boxes_a: torch.Tensor, boxes_b: torch.Tensor
) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes in [x1, y1, x2, y2] format.

    Args:
        boxes_a: (N, 4) tensor.
        boxes_b: (M, 4) tensor.

    Returns:
        iou: (N, M) tensor of IoU values.
    """
    # Intersection coordinates
    max_xy = torch.min(
        boxes_a[:, 2:].unsqueeze(1),  # (N, 1, 2)
        boxes_b[:, 2:].unsqueeze(0),  # (1, M, 2)
    )
    min_xy = torch.max(
        boxes_a[:, :2].unsqueeze(1),  # (N, 1, 2)
        boxes_b[:, :2].unsqueeze(0),  # (1, M, 2)
    )
    inter = (max_xy - min_xy).clamp(min=0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]  # (N, M)

    # Areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])  # (N,)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # (M,)
    union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter_area  # (N, M)

    return inter_area / union.clamp(min=1e-6)


class YOLACTLoss(nn.Module):
    """Combined multi-task loss for YOLACT training.

    Computes three losses:
        1. Classification loss (Focal Loss) on matched anchors
        2. Box regression loss (Smooth L1) on positive anchors
        3. Mask segmentation loss (Binary Cross-Entropy) on positive anchors

    Anchor matching strategy:
        - Positive: IoU with any GT box > pos_iou_threshold
        - Negative: max IoU with all GT boxes < neg_iou_threshold
        - Ignored: IoU between neg and pos thresholds (not used in loss)
        - Each GT box is also assigned to its best-matching anchor (bipartite)

    Args:
        num_classes: Total number of classes including background. Default: 2.
        pos_iou_threshold: IoU threshold for positive anchor matching. Default: 0.5.
        neg_iou_threshold: IoU threshold for negative anchor matching. Default: 0.4.
        cls_weight: Weight for classification loss. Default: 1.0.
        box_weight: Weight for box regression loss. Default: 1.5.
        mask_weight: Weight for mask segmentation loss. Default: 6.125.
        focal_alpha: Alpha parameter for focal loss. Default: 0.25.
        focal_gamma: Gamma parameter for focal loss. Default: 2.0.
        neg_pos_ratio: Maximum ratio of negatives to positives for hard
                       negative mining. Default: 3.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pos_iou_threshold: float = 0.5,
        neg_iou_threshold: float = 0.4,
        cls_weight: float = 1.0,
        box_weight: float = 1.5,
        mask_weight: float = 6.125,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        neg_pos_ratio: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.mask_weight = mask_weight
        self.neg_pos_ratio = neg_pos_ratio

        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            num_classes=num_classes,
        )

    def match_anchors(
        self,
        anchors_xyxy: torch.Tensor,
        anchors_cxcywh: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match anchors to ground truth boxes using IoU-based assignment.

        Each anchor is assigned to at most one GT box. Positives have IoU > 0.5,
        negatives have IoU < 0.4. Additionally, each GT box is guaranteed to
        match its best anchor (bipartite matching).

        Hard negative mining is applied to limit the neg:pos ratio to 3:1.

        Args:
            anchors_xyxy: (A, 4) anchors in [x1, y1, x2, y2] format.
            anchors_cxcywh: (A, 4) anchors in [cx, cy, w, h] format.
            gt_boxes: (G, 4) ground truth boxes in [x1, y1, x2, y2] format.
            gt_labels: (G,) ground truth class labels.

        Returns:
            matched_gt_boxes: (A, 4) GT box assigned to each anchor (in xyxy).
            matched_labels: (A,) class label for each anchor (0=background).
            pos_mask: (A,) boolean mask for positive anchors.
            neg_mask: (A,) boolean mask for negative anchors.
        """
        num_anchors = anchors_xyxy.size(0)
        device = anchors_xyxy.device

        # Default: all anchors are background
        matched_gt_boxes = torch.zeros(num_anchors, 4, device=device)
        matched_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)

        if gt_boxes.size(0) == 0:
            # No GT boxes: all anchors are negative
            pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
            neg_mask = torch.ones(num_anchors, dtype=torch.bool, device=device)
            return matched_gt_boxes, matched_labels, pos_mask, neg_mask

        # Compute IoU matrix: (A, G)
        iou_matrix = compute_iou_matrix(anchors_xyxy, gt_boxes)

        # For each anchor, find the best-matching GT box
        best_gt_iou, best_gt_idx = iou_matrix.max(dim=1)  # (A,)

        # For each GT box, find the best-matching anchor (bipartite)
        best_anchor_iou, best_anchor_idx = iou_matrix.max(dim=0)  # (G,)

        # Assign labels based on IoU thresholds
        # Positive anchors: IoU > pos_threshold
        pos_mask = best_gt_iou >= self.pos_iou_threshold

        # Ensure each GT box has at least one positive anchor
        for gt_idx in range(gt_boxes.size(0)):
            best_anchor = best_anchor_idx[gt_idx]
            pos_mask[best_anchor] = True
            best_gt_idx[best_anchor] = gt_idx

        # Negative anchors: max IoU < neg_threshold
        neg_mask = best_gt_iou < self.neg_iou_threshold
        # Exclude positives from negatives (bipartite matches may have low IoU)
        neg_mask = neg_mask & ~pos_mask

        # Assign matched GT boxes and labels for positive anchors
        matched_gt_boxes[pos_mask] = gt_boxes[best_gt_idx[pos_mask]]
        matched_labels[pos_mask] = gt_labels[best_gt_idx[pos_mask]]

        # Hard negative mining: limit neg:pos ratio
        num_pos = pos_mask.sum().item()
        if num_pos > 0:
            max_neg = num_pos * self.neg_pos_ratio
            num_neg = neg_mask.sum().item()
            if num_neg > max_neg:
                # Keep only the hardest negatives (highest classification loss)
                # For simplicity, randomly sample max_neg negatives
                neg_indices = neg_mask.nonzero(as_tuple=False).squeeze(1)
                perm = torch.randperm(neg_indices.size(0), device=device)[:int(max_neg)]
                selected_neg = neg_indices[perm]
                neg_mask = torch.zeros_like(neg_mask)
                neg_mask[selected_neg] = True

        return matched_gt_boxes, matched_labels, pos_mask, neg_mask

    def forward(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute combined YOLACT loss.

        Args:
            predictions: Tuple of model training outputs:
                - class_preds: (B, num_anchors, num_classes)
                - box_preds: (B, num_anchors, 4) encoded offsets
                - mask_coeffs: (B, num_anchors, num_prototypes)
                - prototypes: (B, num_prototypes, proto_H, proto_W)
                - anchors: (num_anchors, 4) in [cx, cy, w, h] format (normalized)
            targets: List of B target dicts, each with:
                - 'boxes': (G, 4) in [x1, y1, x2, y2] format (absolute pixels)
                - 'labels': (G,) class labels
                - 'masks': (G, H, W) binary ground truth masks

        Returns:
            Dictionary with loss components:
                - 'total': Weighted sum of all losses
                - 'cls': Classification loss
                - 'box': Box regression loss
                - 'mask': Mask segmentation loss
        """
        class_preds, box_preds, mask_coeffs, prototypes, anchors = predictions
        batch_size = class_preds.size(0)
        device = class_preds.device

        # Anchors from the model are in normalized [0,1] [cx,cy,w,h] format.
        # Target boxes are in absolute pixel coordinates [x1,y1,x2,y2].
        # We need to normalize target boxes to [0,1] to match anchor space.
        # Input size is inferred from the prototype spatial dimensions:
        # proto is 138x138 which is input_size/4 -> input_size = 550 approx.
        # But more robustly, we work in normalized coordinates throughout.
        # The YOLACT model normalizes anchors by input_size in _generate_anchors.
        # We normalize GT boxes by input_size (550) to match.
        input_size = 550.0  # Default input size

        # Convert anchors from [cx,cy,w,h] to [x1,y1,x2,y2] for IoU computation
        anchors_xyxy = anchors_to_xyxy(anchors)  # (A, 4) normalized [0,1]

        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_mask_loss = torch.tensor(0.0, device=device)
        total_pos = 0

        for b in range(batch_size):
            gt_boxes = targets[b]['boxes'].to(device)    # (G, 4) absolute pixels
            gt_labels = targets[b]['labels'].to(device)   # (G,)
            gt_masks = targets[b]['masks'].to(device)     # (G, H, W) or (G, 550, 550)

            # Normalize GT boxes to [0, 1] to match anchor coordinate system
            gt_boxes_norm = gt_boxes / input_size

            # Match anchors to GT boxes
            matched_gt_boxes, matched_labels, pos_mask, neg_mask = self.match_anchors(
                anchors_xyxy, anchors, gt_boxes_norm, gt_labels
            )

            num_pos = pos_mask.sum().item()
            total_pos += num_pos

            # --- Classification Loss ---
            # Use both positive and negative anchors
            cls_mask = pos_mask | neg_mask
            if cls_mask.sum() > 0:
                cls_preds_selected = class_preds[b][cls_mask]  # (N_sel, num_classes)
                cls_targets_selected = matched_labels[cls_mask]  # (N_sel,)
                cls_loss = self.focal_loss(cls_preds_selected, cls_targets_selected)
                total_cls_loss = total_cls_loss + cls_loss

            if num_pos == 0:
                continue

            # --- Box Regression Loss ---
            # Encode matched GT boxes as offsets relative to anchors
            pos_anchors = anchors[pos_mask]       # (P, 4) [cx, cy, w, h] normalized
            pos_gt_boxes = matched_gt_boxes[pos_mask]  # (P, 4) [x1,y1,x2,y2] normalized

            # encode_boxes expects gt in [x1,y1,x2,y2] and anchors in [cx,cy,w,h]
            gt_encoded = encode_boxes(pos_gt_boxes, pos_anchors)  # (P, 4)
            pred_encoded = box_preds[b][pos_mask]  # (P, 4)

            box_loss = F.smooth_l1_loss(pred_encoded, gt_encoded, reduction='sum')
            total_box_loss = total_box_loss + box_loss

            # --- Mask Segmentation Loss ---
            # For each positive anchor, assemble the predicted mask from prototypes
            # and compare against the GT mask
            pos_coeffs = mask_coeffs[b][pos_mask]  # (P, num_prototypes)
            proto = prototypes[b]  # (num_prototypes, proto_H, proto_W)
            proto_h, proto_w = proto.shape[1], proto.shape[2]

            # Assemble predicted masks: (P, num_prototypes) @ (num_prototypes, H*W) -> (P, H*W)
            pred_masks = pos_coeffs @ proto.view(proto.size(0), -1)
            pred_masks = pred_masks.view(-1, proto_h, proto_w)  # (P, proto_H, proto_W)

            # Prepare GT masks at prototype resolution
            # GT masks are at input resolution (550x550), resize to proto resolution (138x138)
            if gt_masks.dim() == 2:
                gt_masks = gt_masks.unsqueeze(0)

            # Get the GT index for each positive anchor
            # We need to find which GT each positive anchor was matched to
            best_gt_iou_all, best_gt_idx_all = compute_iou_matrix(
                anchors_xyxy, gt_boxes_norm
            ).max(dim=1)
            pos_gt_indices = best_gt_idx_all[pos_mask]  # (P,)

            if gt_masks.size(0) > 0 and gt_masks.dim() == 3:
                # Resize GT masks to prototype resolution
                gt_masks_resized = F.interpolate(
                    gt_masks.unsqueeze(1).float(),
                    size=(proto_h, proto_w),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1)  # (G, proto_H, proto_W)

                # Select the GT mask for each positive anchor
                pos_gt_masks = gt_masks_resized[pos_gt_indices]  # (P, proto_H, proto_W)

                # Crop both predicted and GT masks by the GT box region
                # This focuses the mask loss on the relevant area
                pos_boxes_norm = pos_gt_boxes  # Already in normalized [0,1]

                # Create crop mask for each positive anchor
                x_grid = torch.arange(proto_w, device=device).float() / proto_w
                y_grid = torch.arange(proto_h, device=device).float() / proto_h
                y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
                x_grid = x_grid.unsqueeze(0).expand(num_pos, -1, -1)  # (P, H, W)
                y_grid = y_grid.unsqueeze(0).expand(num_pos, -1, -1)

                x1 = pos_boxes_norm[:, 0].view(-1, 1, 1)
                y1 = pos_boxes_norm[:, 1].view(-1, 1, 1)
                x2 = pos_boxes_norm[:, 2].view(-1, 1, 1)
                y2 = pos_boxes_norm[:, 3].view(-1, 1, 1)

                crop = (x_grid >= x1) & (x_grid <= x2) & (y_grid >= y1) & (y_grid <= y2)
                crop = crop.float()  # (P, proto_H, proto_W)

                # Apply crop and compute BCE loss
                pred_masks_cropped = pred_masks * crop
                gt_masks_cropped = pos_gt_masks * crop

                # Binary cross-entropy with logits (pred_masks are pre-sigmoid)
                mask_loss = F.binary_cross_entropy_with_logits(
                    pred_masks_cropped,
                    gt_masks_cropped,
                    reduction='none',
                )
                # Average loss within each mask's crop region, then sum over positives
                # Avoid division by zero for empty crops
                crop_areas = crop.sum(dim=(1, 2)).clamp(min=1.0)  # (P,)
                mask_loss_per_instance = (mask_loss * crop).sum(dim=(1, 2)) / crop_areas
                total_mask_loss = total_mask_loss + mask_loss_per_instance.sum()

        # Normalize by total number of positives
        num_pos_safe = max(total_pos, 1)
        cls_loss_norm = total_cls_loss / num_pos_safe
        box_loss_norm = total_box_loss / num_pos_safe
        mask_loss_norm = total_mask_loss / num_pos_safe

        # Weighted total loss
        total_loss = (
            self.cls_weight * cls_loss_norm
            + self.box_weight * box_loss_norm
            + self.mask_weight * mask_loss_norm
        )

        return {
            'total': total_loss,
            'cls': cls_loss_norm.detach(),
            'box': box_loss_norm.detach(),
            'mask': mask_loss_norm.detach(),
        }


if __name__ == '__main__':
    """Test loss computation with synthetic data."""
    print("=== YOLACT Loss Test ===\n")

    device = torch.device('cpu')

    # Simulate model outputs
    batch_size = 2
    num_anchors = 100
    num_classes = 2
    num_prototypes = 32
    proto_h, proto_w = 138, 138

    class_preds = torch.randn(batch_size, num_anchors, num_classes, device=device)
    box_preds = torch.randn(batch_size, num_anchors, 4, device=device) * 0.1
    mask_coeffs = torch.randn(batch_size, num_anchors, num_prototypes, device=device) * 0.1
    prototypes = torch.randn(batch_size, num_prototypes, proto_h, proto_w, device=device)

    # Anchors in normalized [cx, cy, w, h]
    anchors = torch.rand(num_anchors, 4, device=device)
    anchors[:, 2:] = anchors[:, 2:] * 0.3 + 0.05  # Reasonable widths/heights

    predictions = (class_preds, box_preds, mask_coeffs, prototypes, anchors)

    # Simulate targets
    targets = []
    for _ in range(batch_size):
        num_gt = 5
        gt_boxes = torch.rand(num_gt, 4, device=device) * 550
        # Ensure x2 > x1 and y2 > y1
        gt_boxes[:, 2] = gt_boxes[:, 0] + torch.rand(num_gt, device=device) * 100 + 20
        gt_boxes[:, 3] = gt_boxes[:, 1] + torch.rand(num_gt, device=device) * 100 + 20
        gt_boxes = gt_boxes.clamp(0, 550)
        gt_labels = torch.ones(num_gt, dtype=torch.long, device=device)
        gt_masks = torch.zeros(num_gt, 550, 550, dtype=torch.uint8, device=device)
        for i in range(num_gt):
            x1 = int(gt_boxes[i, 0].item())
            y1 = int(gt_boxes[i, 1].item())
            x2 = int(gt_boxes[i, 2].item())
            y2 = int(gt_boxes[i, 3].item())
            gt_masks[i, y1:y2, x1:x2] = 1

        targets.append({
            'boxes': gt_boxes,
            'labels': gt_labels,
            'masks': gt_masks,
        })

    # Compute loss
    criterion = YOLACTLoss(num_classes=2)
    losses = criterion(predictions, targets)

    print(f"Total loss: {losses['total']:.4f}")
    print(f"  Classification: {losses['cls']:.4f}")
    print(f"  Box regression: {losses['box']:.4f}")
    print(f"  Mask:           {losses['mask']:.4f}")
    print("\n=== Test Complete ===")
