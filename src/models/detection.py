"""Post-processing module for YOLACT detection outputs.

Handles decoding of raw network predictions into final detections:
    1. Decode bounding box offsets from anchor-relative to absolute coordinates
    2. Apply confidence thresholding to filter low-confidence predictions
    3. Apply Soft-NMS to suppress duplicate detections
    4. Assemble instance masks from prototype masks and mask coefficients
    5. Crop masks by predicted bounding boxes
    6. Apply top-k filtering and maximum detection limits
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from src.utils.soft_nms import soft_nms


class Detect:
    """Post-processing for YOLACT predictions.

    Decodes raw predictions into final detection results with boxes,
    scores, class labels, and instance masks.

    Args:
        num_classes: Number of classes including background.
        conf_threshold: Minimum confidence score to keep a detection.
        nms_sigma: Sigma parameter for Gaussian Soft-NMS.
        top_k: Maximum number of detections to consider before NMS.
        max_detections: Maximum number of final detections per image.
    """

    def __init__(
        self,
        num_classes: int = 2,
        conf_threshold: float = 0.05,
        nms_sigma: float = 0.5,
        top_k: int = 200,
        max_detections: int = 100,
    ) -> None:
        """Initialize Detect post-processor.

        Args:
            num_classes: Total classes including background.
            conf_threshold: Confidence threshold for filtering.
            nms_sigma: Gaussian Soft-NMS sigma.
            top_k: Pre-NMS top-k filtering count.
            max_detections: Post-NMS maximum detections per image.
        """
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_sigma = nms_sigma
        self.top_k = top_k
        self.max_detections = max_detections

    @staticmethod
    def decode_boxes(
        box_preds: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Decode predicted box offsets relative to anchors into absolute coordinates.

        Uses the standard box encoding scheme:
            tx = (x - xa) / wa,  ty = (y - ya) / ha
            tw = log(w / wa),    th = log(h / ha)

        Decodes as:
            x = tx * wa + xa,    y = ty * ha + ya
            w = exp(tw) * wa,    h = exp(th) * ha

        Args:
            box_preds: Predicted offsets (B, N, 4) as [tx, ty, tw, th].
            anchors: Prior anchor boxes (N, 4) as [cx, cy, w, h].

        Returns:
            Decoded boxes (B, N, 4) as [x1, y1, x2, y2].
        """
        anchor_cx = anchors[:, 0]
        anchor_cy = anchors[:, 1]
        anchor_w = anchors[:, 2]
        anchor_h = anchors[:, 3]

        # Decode center and size
        pred_cx = box_preds[..., 0] * anchor_w + anchor_cx
        pred_cy = box_preds[..., 1] * anchor_h + anchor_cy
        pred_w = torch.exp(box_preds[..., 2].clamp(max=4.0)) * anchor_w
        pred_h = torch.exp(box_preds[..., 3].clamp(max=4.0)) * anchor_h

        # Convert center-size to corner format [x1, y1, x2, y2]
        x1 = pred_cx - pred_w / 2
        y1 = pred_cy - pred_h / 2
        x2 = pred_cx + pred_w / 2
        y2 = pred_cy + pred_h / 2

        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def assemble_masks(
        prototypes: torch.Tensor,
        mask_coeffs: torch.Tensor,
        boxes: torch.Tensor,
        mask_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Assemble instance masks from prototypes and coefficients.

        Computes M = sigmoid(mask_coefficients @ prototypes) and crops
        each mask by its corresponding bounding box.

        Args:
            prototypes: Prototype masks (num_prototypes, H, W).
            mask_coeffs: Mask coefficients (N, num_prototypes) for N detections.
            boxes: Detected boxes (N, 4) as [x1, y1, x2, y2] in normalized [0,1].
            mask_size: If provided, resize masks to (mask_size, mask_size).

        Returns:
            Instance masks (N, H, W) with values in [0, 1].
        """
        num_dets = mask_coeffs.size(0)
        if num_dets == 0:
            h, w = prototypes.shape[1], prototypes.shape[2]
            return torch.zeros(0, h, w, device=prototypes.device)

        proto_h, proto_w = prototypes.shape[1], prototypes.shape[2]

        # mask_coeffs: (N, P), prototypes: (P, H, W) -> reshape to (P, H*W)
        # (N, P) @ (P, H*W) -> (N, H*W) -> (N, H, W)
        masks = mask_coeffs @ prototypes.view(prototypes.size(0), -1)
        masks = masks.view(num_dets, proto_h, proto_w)
        masks = torch.sigmoid(masks)

        # Crop masks by bounding boxes
        masks = Detect.crop_masks(masks, boxes, proto_h, proto_w)

        if mask_size is not None:
            masks = F.interpolate(
                masks.unsqueeze(1),
                size=(mask_size, mask_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)

        return masks

    @staticmethod
    def crop_masks(
        masks: torch.Tensor,
        boxes: torch.Tensor,
        mask_h: int,
        mask_w: int,
    ) -> torch.Tensor:
        """Crop masks to the area within their corresponding bounding boxes.

        Regions of the mask outside the bounding box are zeroed out.

        Args:
            masks: Instance masks (N, H, W).
            boxes: Bounding boxes (N, 4) as [x1, y1, x2, y2] in normalized [0,1].
            mask_h: Height of the mask.
            mask_w: Width of the mask.

        Returns:
            Cropped masks (N, H, W) with regions outside the box zeroed.
        """
        num_dets = masks.size(0)
        if num_dets == 0:
            return masks

        device = masks.device

        # Create normalized coordinate grids
        x = torch.arange(mask_w, device=device).float() / mask_w
        y = torch.arange(mask_h, device=device).float() / mask_h

        # Expand for broadcasting: (1, 1, W) and (1, H, 1)
        x = x.view(1, 1, -1).expand(num_dets, mask_h, mask_w)
        y = y.view(1, -1, 1).expand(num_dets, mask_h, mask_w)

        # Extract box coordinates: (N, 1, 1)
        x1 = boxes[:, 0].view(-1, 1, 1)
        y1 = boxes[:, 1].view(-1, 1, 1)
        x2 = boxes[:, 2].view(-1, 1, 1)
        y2 = boxes[:, 3].view(-1, 1, 1)

        # Create binary crop mask: 1 inside box, 0 outside
        crop = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)

        return masks * crop.float()

    def __call__(
        self,
        class_preds: torch.Tensor,
        box_preds: torch.Tensor,
        mask_coeffs: torch.Tensor,
        prototypes: torch.Tensor,
        anchors: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Process raw predictions into final detections.

        Args:
            class_preds: Classification logits (B, N, num_classes).
            box_preds: Box regression offsets (B, N, 4).
            mask_coeffs: Mask coefficients (B, N, num_prototypes).
            prototypes: Prototype masks (B, num_prototypes, H, W).
            anchors: Prior anchors (N, 4) as [cx, cy, w, h].

        Returns:
            List of detection dictionaries, one per image in the batch.
            Each dict contains:
                - 'boxes': (K, 4) detected boxes as [x1, y1, x2, y2]
                - 'scores': (K,) confidence scores
                - 'labels': (K,) class labels (0-indexed foreground)
                - 'masks': (K, H, W) instance masks
        """
        batch_size = class_preds.size(0)

        # Apply softmax to get class probabilities
        class_probs = F.softmax(class_preds, dim=-1)

        # Decode box predictions from anchor offsets to absolute coords
        decoded_boxes = self.decode_boxes(box_preds, anchors)

        # Clamp boxes to [0, 1] normalized range
        decoded_boxes = decoded_boxes.clamp(min=0.0, max=1.0)

        results = []
        for b in range(batch_size):
            result = self._process_single_image(
                class_probs[b],
                decoded_boxes[b],
                mask_coeffs[b],
                prototypes[b],
            )
            results.append(result)

        return results

    def _process_single_image(
        self,
        class_probs: torch.Tensor,
        boxes: torch.Tensor,
        mask_coeffs: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Process predictions for a single image.

        Args:
            class_probs: (N, num_classes) class probabilities after softmax.
            boxes: (N, 4) decoded boxes in [x1, y1, x2, y2] format.
            mask_coeffs: (N, num_prototypes) mask coefficients.
            prototypes: (num_prototypes, H, W) prototype masks.

        Returns:
            Detection dictionary with 'boxes', 'scores', 'labels', 'masks'.
        """
        device = class_probs.device
        proto_h, proto_w = prototypes.shape[1], prototypes.shape[2]

        empty_result = {
            'boxes': torch.zeros(0, 4, device=device),
            'scores': torch.zeros(0, device=device),
            'labels': torch.zeros(0, dtype=torch.long, device=device),
            'masks': torch.zeros(0, proto_h, proto_w, device=device),
        }

        # Get foreground class scores (skip background at index 0)
        fg_probs = class_probs[:, 1:]  # (N, num_fg_classes)
        max_scores, max_classes = fg_probs.max(dim=-1)  # (N,), (N,)

        # Confidence thresholding
        keep_mask = max_scores > self.conf_threshold
        if keep_mask.sum() == 0:
            return empty_result

        scores = max_scores[keep_mask]
        labels = max_classes[keep_mask]  # 0-indexed foreground classes
        det_boxes = boxes[keep_mask]
        det_coeffs = mask_coeffs[keep_mask]

        # Top-k filtering before NMS
        if scores.size(0) > self.top_k:
            top_k_scores, top_k_idx = scores.topk(self.top_k)
            scores = top_k_scores
            labels = labels[top_k_idx]
            det_boxes = det_boxes[top_k_idx]
            det_coeffs = det_coeffs[top_k_idx]

        # Apply Soft-NMS per class
        all_boxes = []
        all_scores = []
        all_labels = []
        all_coeffs = []

        unique_labels = labels.unique()
        for cls in unique_labels:
            cls_mask = labels == cls
            cls_boxes = det_boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_coeffs = det_coeffs[cls_mask]

            kept_boxes, kept_scores, keep_idx = soft_nms(
                cls_boxes, cls_scores,
                sigma=self.nms_sigma,
                score_threshold=self.conf_threshold,
            )

            if kept_boxes.size(0) > 0:
                all_boxes.append(kept_boxes)
                all_scores.append(kept_scores)
                all_labels.append(
                    torch.full((kept_boxes.size(0),), cls.item(),
                               dtype=torch.long, device=device)
                )
                all_coeffs.append(cls_coeffs[keep_idx])

        if len(all_boxes) == 0:
            return empty_result

        final_boxes = torch.cat(all_boxes, dim=0)
        final_scores = torch.cat(all_scores, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        final_coeffs = torch.cat(all_coeffs, dim=0)

        # Max detections limit (keep highest scoring)
        if final_scores.size(0) > self.max_detections:
            top_idx = final_scores.topk(self.max_detections).indices
            final_boxes = final_boxes[top_idx]
            final_scores = final_scores[top_idx]
            final_labels = final_labels[top_idx]
            final_coeffs = final_coeffs[top_idx]

        # Assemble instance masks from prototypes and coefficients
        final_masks = self.assemble_masks(prototypes, final_coeffs, final_boxes)

        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels,
            'masks': final_masks,
        }


if __name__ == '__main__':
    # Quick test of core components
    print("Testing Detect post-processor components...")

    # Test box decoding
    print("\n--- Box Decoding ---")
    anchors = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],  # cx, cy, w, h
        [0.3, 0.3, 0.1, 0.1],
    ])
    box_preds = torch.zeros(1, 2, 4)  # Zero offsets -> anchor positions
    decoded = Detect.decode_boxes(box_preds, anchors)
    print(f"Anchors (cx,cy,w,h):\n  {anchors}")
    print(f"Decoded (x1,y1,x2,y2):\n  {decoded[0]}")
    # Expected: [0.4, 0.4, 0.6, 0.6] and [0.25, 0.25, 0.35, 0.35]

    # Test mask assembly
    print("\n--- Mask Assembly ---")
    prototypes = torch.randn(32, 138, 138)
    coeffs = torch.randn(3, 32)
    boxes = torch.tensor([
        [0.1, 0.1, 0.5, 0.5],
        [0.3, 0.3, 0.7, 0.7],
        [0.5, 0.5, 0.9, 0.9],
    ])
    masks = Detect.assemble_masks(prototypes, coeffs, boxes)
    print(f"Prototype shape: {prototypes.shape}")
    print(f"Coefficients shape: {coeffs.shape}")
    print(f"Output mask shape: {masks.shape}")
    print(f"Mask value range: [{masks.min():.3f}, {masks.max():.3f}]")

    # Test empty case
    print("\n--- Empty Input ---")
    empty_masks = Detect.assemble_masks(
        prototypes, torch.zeros(0, 32), torch.zeros(0, 4)
    )
    print(f"Empty result shape: {empty_masks.shape}")
