"""YOLACT Prediction Head for classification, box regression, and mask coefficients.

A shared prediction head applied identically to each FPN level. Uses shared
convolution weights across all levels to produce per-anchor predictions:
    - Classification scores (num_anchors * num_classes per spatial location)
    - Bounding box offsets (num_anchors * 4 per spatial location)
    - Mask coefficients (num_anchors * num_prototypes per spatial location)

The mask coefficient branch uses tanh activation to allow negative
coefficients, enabling subtractive combinations of prototype masks.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """Multi-task prediction head for YOLACT.

    Shared-weight head applied to each FPN level to produce classification
    scores, bounding box regressions, and mask coefficients for each anchor.

    Args:
        in_channels: Number of input channels from FPN (typically 256).
        num_classes: Number of classes including background (2 for single-class).
        num_anchors: Number of anchors per spatial location (9 = 3 ratios x 3 scales).
        num_prototypes: Number of prototype masks (32).
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 2,
        num_anchors: int = 9,
        num_prototypes: int = 32,
    ) -> None:
        """Initialize PredictionHead.

        Args:
            in_channels: Input channels from FPN levels.
            num_classes: Total number of classes (background + object classes).
            num_anchors: Anchors per spatial location.
            num_prototypes: Number of prototype masks for mask coefficient output.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_prototypes = num_prototypes

        # Shared feature extraction layer
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Classification branch
        self.class_conv = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )

        # Bounding box regression branch
        self.box_conv = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, padding=1
        )

        # Mask coefficient branch (tanh activation applied in forward)
        self.mask_conv = nn.Conv2d(
            in_channels, num_anchors * num_prototypes, kernel_size=3, padding=1
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights with Xavier uniform for convolutions."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize classification bias with prior probability for better
        # initial training stability (focal loss style initialization)
        import math
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_conv.bias, bias_value)

    def _forward_single_level(
        self, feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a single FPN level through the prediction head.

        Args:
            feature: Feature tensor of shape (B, C, H, W) from one FPN level.

        Returns:
            Tuple of:
                - class_pred: (B, H*W*num_anchors, num_classes)
                - box_pred: (B, H*W*num_anchors, 4)
                - mask_coeff_pred: (B, H*W*num_anchors, num_prototypes)
        """
        batch_size = feature.size(0)

        # Shared feature extraction
        shared = self.shared_conv(feature)

        # Classification
        class_pred = self.class_conv(shared)
        # Reshape: (B, A*C, H, W) -> (B, H, W, A*C) -> (B, H*W*A, C)
        class_pred = class_pred.permute(0, 2, 3, 1).contiguous()
        class_pred = class_pred.view(batch_size, -1, self.num_classes)

        # Box regression
        box_pred = self.box_conv(shared)
        box_pred = box_pred.permute(0, 2, 3, 1).contiguous()
        box_pred = box_pred.view(batch_size, -1, 4)

        # Mask coefficients with tanh activation
        mask_coeff = self.mask_conv(shared)
        mask_coeff = mask_coeff.permute(0, 2, 3, 1).contiguous()
        mask_coeff = mask_coeff.view(batch_size, -1, self.num_prototypes)
        mask_coeff = torch.tanh(mask_coeff)

        return class_pred, box_pred, mask_coeff

    def forward(
        self, fpn_features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process all FPN levels and concatenate predictions.

        Args:
            fpn_features: List of FPN feature tensors [P3, P4, P5, P6, P7],
                each of shape (B, C, H_i, W_i).

        Returns:
            Tuple of concatenated predictions across all FPN levels:
                - class_preds: (B, total_anchors, num_classes)
                - box_preds: (B, total_anchors, 4)
                - mask_coeff_preds: (B, total_anchors, num_prototypes)
            where total_anchors = sum(H_i * W_i * num_anchors) for all levels.
        """
        all_class_preds = []
        all_box_preds = []
        all_mask_coeffs = []

        for feature in fpn_features:
            class_pred, box_pred, mask_coeff = self._forward_single_level(feature)
            all_class_preds.append(class_pred)
            all_box_preds.append(box_pred)
            all_mask_coeffs.append(mask_coeff)

        # Concatenate predictions from all FPN levels along the anchor dimension
        class_preds = torch.cat(all_class_preds, dim=1)
        box_preds = torch.cat(all_box_preds, dim=1)
        mask_coeff_preds = torch.cat(all_mask_coeffs, dim=1)

        return class_preds, box_preds, mask_coeff_preds


if __name__ == '__main__':
    # Verify PredictionHead output shapes
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    head = PredictionHead(
        in_channels=256,
        num_classes=2,
        num_anchors=9,
        num_prototypes=32,
    ).to(device)

    # Simulate FPN outputs for 550x550 input
    fpn_sizes = [(69, 69), (35, 35), (18, 18), (9, 9), (5, 5)]
    fpn_features = [
        torch.randn(1, 256, h, w, device=device) for h, w in fpn_sizes
    ]

    with torch.no_grad():
        class_preds, box_preds, mask_coeff_preds = head(fpn_features)

    # Calculate expected total anchors
    total_anchors = sum(h * w * 9 for h, w in fpn_sizes)
    print(f"\nFPN spatial sizes: {fpn_sizes}")
    print(f"Expected total anchors: {total_anchors}")

    print(f"\nPrediction shapes:")
    print(f"  Class predictions:  {class_preds.shape}  (expected: (1, {total_anchors}, 2))")
    print(f"  Box predictions:    {box_preds.shape}  (expected: (1, {total_anchors}, 4))")
    print(f"  Mask coefficients:  {mask_coeff_preds.shape}  (expected: (1, {total_anchors}, 32))")

    total_params = sum(p.numel() for p in head.parameters())
    print(f"\nPredictionHead parameters: {total_params:,}")
