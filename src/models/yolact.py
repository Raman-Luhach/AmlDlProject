"""YOLACT: Real-time Instance Segmentation with MobileNetV3-Large backbone.

Full model architecture combining:
    - MobileNetV3-Large backbone (ImageNet pretrained)
    - Feature Pyramid Network (FPN) for multi-scale features
    - ProtoNet for prototype mask generation
    - PredictionHead for class, box, and mask coefficient predictions
    - Detect for post-processing during inference

Input: (B, 3, 550, 550) RGB images
Training output: (class_preds, box_preds, mask_coeffs, prototypes, anchors)
Inference output: List of detection dicts with boxes, scores, masks, labels
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.backbone import MobileNetV3Backbone
from src.models.fpn import FPN
from src.models.protonet import ProtoNet
from src.models.prediction_head import PredictionHead
from src.models.detection import Detect


# Default model configuration
DEFAULT_CONFIG = {
    # Input
    'input_size': 550,

    # Backbone
    'pretrained_backbone': True,
    'freeze_backbone_layers': 0,

    # FPN
    'fpn_out_channels': 256,

    # ProtoNet
    'num_prototypes': 32,

    # Detection
    'num_classes': 2,  # 1 foreground + 1 background
    'num_anchors': 9,  # 3 aspect ratios x 3 scales

    # Anchors
    'anchor_scales': [24, 48, 96, 192, 384],   # Base sizes for P3-P7
    'anchor_ratios': [0.5, 1.0, 2.0],           # Aspect ratios
    'anchor_scale_factors': [1.0, 1.26, 1.587],  # ~2^(0/3), 2^(1/3), 2^(2/3)

    # Post-processing
    'conf_threshold': 0.05,
    'nms_sigma': 0.5,
    'top_k': 200,
    'max_detections': 100,
}


class YOLACT(nn.Module):
    """YOLACT model for real-time instance segmentation.

    Combines a MobileNetV3-Large backbone with FPN, ProtoNet, and
    multi-task prediction head for single-class instance segmentation.
    Uses Soft-NMS for post-processing.

    Args:
        config: Configuration dictionary. Missing keys use DEFAULT_CONFIG values.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize YOLACT model.

        Args:
            config: Optional configuration dict. Merged with DEFAULT_CONFIG.
        """
        super().__init__()

        # Merge provided config with defaults
        self.config = {**DEFAULT_CONFIG}
        if config is not None:
            self.config.update(config)

        cfg = self.config

        # Build backbone
        self.backbone = MobileNetV3Backbone(pretrained=cfg['pretrained_backbone'])
        if cfg['freeze_backbone_layers'] > 0:
            self.backbone.freeze_layers(cfg['freeze_backbone_layers'])

        # Build FPN
        self.fpn = FPN(
            in_channels_list=self.backbone.out_channels,
            out_channels=cfg['fpn_out_channels'],
        )

        # Build ProtoNet (operates on P3, the finest FPN level)
        self.protonet = ProtoNet(
            in_channels=cfg['fpn_out_channels'],
            num_prototypes=cfg['num_prototypes'],
        )

        # Build prediction head (shared across all FPN levels)
        self.prediction_head = PredictionHead(
            in_channels=cfg['fpn_out_channels'],
            num_classes=cfg['num_classes'],
            num_anchors=cfg['num_anchors'],
            num_prototypes=cfg['num_prototypes'],
        )

        # Post-processing (not an nn.Module, no parameters)
        self.detect = Detect(
            num_classes=cfg['num_classes'],
            conf_threshold=cfg['conf_threshold'],
            nms_sigma=cfg['nms_sigma'],
            top_k=cfg['top_k'],
            max_detections=cfg['max_detections'],
        )

        # Pre-generate anchors (registered as buffer, not a parameter)
        self._anchors = None
        self._anchor_input_size = None

    @property
    def device(self) -> torch.device:
        """Return the device of the model parameters.

        Returns:
            torch.device of the first model parameter.
        """
        return next(self.parameters()).device

    def _generate_anchors(
        self, fpn_shapes: List[Tuple[int, int]], input_size: int
    ) -> torch.Tensor:
        """Generate anchor boxes for all FPN levels.

        Creates a grid of anchors at each FPN level with multiple scales
        and aspect ratios. Anchors are in [cx, cy, w, h] format, normalized
        to [0, 1] relative to input image size.

        Args:
            fpn_shapes: List of (H, W) spatial dimensions for each FPN level.
            input_size: Input image size (assumes square input).

        Returns:
            Anchors tensor (total_anchors, 4) as [cx, cy, w, h] in [0, 1].
        """
        cfg = self.config
        anchor_scales = cfg['anchor_scales']
        anchor_ratios = cfg['anchor_ratios']
        scale_factors = cfg['anchor_scale_factors']

        all_anchors = []

        for level_idx, (fh, fw) in enumerate(fpn_shapes):
            base_size = anchor_scales[level_idx]

            # Generate anchor centers on the grid
            # Center of each cell in normalized coordinates
            cy = (torch.arange(fh, dtype=torch.float32) + 0.5) / fh
            cx = (torch.arange(fw, dtype=torch.float32) + 0.5) / fw
            cy, cx = torch.meshgrid(cy, cx, indexing='ij')
            centers = torch.stack([cx, cy], dim=-1).view(-1, 2)  # (fh*fw, 2)

            # Generate anchor sizes (width, height) for each ratio and scale
            anchors_wh = []
            for sf in scale_factors:
                for ratio in anchor_ratios:
                    # w = base_size * sf * sqrt(ratio)
                    # h = base_size * sf / sqrt(ratio)
                    w = base_size * sf * math.sqrt(ratio) / input_size
                    h = base_size * sf / math.sqrt(ratio) / input_size
                    anchors_wh.append([w, h])

            anchors_wh = torch.tensor(anchors_wh, dtype=torch.float32)  # (num_anchors, 2)

            # Combine centers with sizes
            num_centers = centers.size(0)
            num_anchors = anchors_wh.size(0)

            # Expand: centers (fh*fw, 1, 2) x sizes (1, num_anchors, 2)
            centers_exp = centers.unsqueeze(1).expand(num_centers, num_anchors, 2)
            sizes_exp = anchors_wh.unsqueeze(0).expand(num_centers, num_anchors, 2)

            # Concatenate to [cx, cy, w, h]
            level_anchors = torch.cat([centers_exp, sizes_exp], dim=-1)
            level_anchors = level_anchors.view(-1, 4)  # (fh*fw*num_anchors, 4)

            all_anchors.append(level_anchors)

        return torch.cat(all_anchors, dim=0)

    def forward(
        self, images: torch.Tensor
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        List[Dict[str, torch.Tensor]],
    ]:
        """Forward pass through the full YOLACT model.

        Args:
            images: Input images (B, 3, H, W). Expected H=W=550.

        Returns:
            If training:
                Tuple of (class_preds, box_preds, mask_coeffs, prototypes, anchors)
                    - class_preds: (B, total_anchors, num_classes)
                    - box_preds: (B, total_anchors, 4)
                    - mask_coeffs: (B, total_anchors, num_prototypes)
                    - prototypes: (B, num_prototypes, proto_H, proto_W)
                    - anchors: (total_anchors, 4) as [cx, cy, w, h]

            If eval:
                List of detection dicts, one per image, each containing:
                    - 'boxes': (K, 4) as [x1, y1, x2, y2]
                    - 'scores': (K,)
                    - 'labels': (K,)
                    - 'masks': (K, proto_H, proto_W)
        """
        batch_size = images.size(0)
        input_size = images.size(2)

        # 1. Extract multi-scale backbone features
        backbone_features = self.backbone(images)

        # 2. Build feature pyramid
        fpn_features = self.fpn(backbone_features)
        # fpn_features: [P3, P4, P5, P6, P7]

        # 3. Generate prototype masks from P3 (finest level)
        prototypes = self.protonet(fpn_features[0])
        # prototypes: (B, num_prototypes, proto_H, proto_W)

        # 4. Predict class scores, box offsets, mask coefficients at all levels
        class_preds, box_preds, mask_coeffs = self.prediction_head(fpn_features)

        # 5. Generate or retrieve anchors
        fpn_shapes = [(f.size(2), f.size(3)) for f in fpn_features]
        if self._anchors is None or self._anchor_input_size != input_size:
            self._anchors = self._generate_anchors(fpn_shapes, input_size)
            self._anchor_input_size = input_size
        anchors = self._anchors.to(images.device)

        if self.training:
            return class_preds, box_preds, mask_coeffs, prototypes, anchors
        else:
            # Post-processing for inference
            with torch.no_grad():
                detections = self.detect(
                    class_preds, box_preds, mask_coeffs, prototypes, anchors
                )
            return detections

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters, broken down by component.

        Returns:
            Dictionary with parameter counts for each model component
            and total/trainable counts.
        """
        def _count(module: nn.Module) -> Tuple[int, int]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        backbone_total, backbone_train = _count(self.backbone)
        fpn_total, fpn_train = _count(self.fpn)
        proto_total, proto_train = _count(self.protonet)
        head_total, head_train = _count(self.prediction_head)

        total = backbone_total + fpn_total + proto_total + head_total
        trainable = backbone_train + fpn_train + proto_train + head_train

        return {
            'backbone': {'total': backbone_total, 'trainable': backbone_train},
            'fpn': {'total': fpn_total, 'trainable': fpn_train},
            'protonet': {'total': proto_total, 'trainable': proto_train},
            'prediction_head': {'total': head_total, 'trainable': head_train},
            'total': total,
            'trainable': trainable,
        }


if __name__ == '__main__':
    from src.utils.helpers import get_device, format_params

    device = get_device()
    print(f"Using device: {device}")

    # Create model with default config
    model = YOLACT().to(device)

    # Print parameter counts
    param_info = model.count_parameters()
    print("\nModel Parameter Counts:")
    print(f"  Backbone:        {format_params(param_info['backbone']['total']):>8s} total, "
          f"{format_params(param_info['backbone']['trainable']):>8s} trainable")
    print(f"  FPN:             {format_params(param_info['fpn']['total']):>8s} total, "
          f"{format_params(param_info['fpn']['trainable']):>8s} trainable")
    print(f"  ProtoNet:        {format_params(param_info['protonet']['total']):>8s} total, "
          f"{format_params(param_info['protonet']['trainable']):>8s} trainable")
    print(f"  PredictionHead:  {format_params(param_info['prediction_head']['total']):>8s} total, "
          f"{format_params(param_info['prediction_head']['trainable']):>8s} trainable")
    print(f"  -------")
    print(f"  Total:           {format_params(param_info['total']):>8s} total, "
          f"{format_params(param_info['trainable']):>8s} trainable")

    # Test training forward pass
    print(f"\nTesting training forward pass (input: 1x3x550x550)...")
    model.train()
    dummy_input = torch.randn(1, 3, 550, 550, device=device)

    with torch.no_grad():
        class_preds, box_preds, mask_coeffs, prototypes, anchors = model(dummy_input)

    print(f"  Class predictions:  {class_preds.shape}")
    print(f"  Box predictions:    {box_preds.shape}")
    print(f"  Mask coefficients:  {mask_coeffs.shape}")
    print(f"  Prototypes:         {prototypes.shape}")
    print(f"  Anchors:            {anchors.shape}")

    # Test inference forward pass
    print(f"\nTesting inference forward pass...")
    model.eval()
    with torch.no_grad():
        detections = model(dummy_input)

    print(f"  Number of images processed: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  Image {i}:")
        print(f"    Boxes:  {det['boxes'].shape}")
        print(f"    Scores: {det['scores'].shape}")
        print(f"    Labels: {det['labels'].shape}")
        print(f"    Masks:  {det['masks'].shape}")

    print(f"\nModel device: {model.device}")
