"""MobileNetV3-Large backbone with FPN tap points for YOLACT.

Extracts multi-scale features at stride 8, 16, and ~32 from MobileNetV3-Large
for use with a Feature Pyramid Network. Uses ImageNet-pretrained weights.

Architecture tap points (verified with 550x550 input):
    C3: After features[6]  -> (B, 40,  69, 69)  stride 8
    C4: After features[12] -> (B, 112, 35, 35)  stride 16
    C5: After features[16] -> (B, 960, 18, 18)  stride ~32
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV3Backbone(nn.Module):
    """MobileNetV3-Large backbone that extracts multi-scale features.

    This backbone wraps torchvision's MobileNetV3-Large and provides
    feature extraction at three scales suitable for FPN integration.

    Attributes:
        C3_CHANNELS: Number of output channels at stride-8 level (40).
        C4_CHANNELS: Number of output channels at stride-16 level (112).
        C5_CHANNELS: Number of output channels at stride-32 level (960).
    """

    # Layer indices for feature extraction (0-indexed into model.features)
    _C3_END = 6   # After InvertedResidual block 6: 40ch, stride 8
    _C4_END = 12  # After InvertedResidual block 12: 112ch, stride 16
    _C5_END = 16  # After Conv2dNormActivation (final): 960ch, stride ~32

    C3_CHANNELS = 40
    C4_CHANNELS = 112
    C5_CHANNELS = 960

    def __init__(self, pretrained: bool = True) -> None:
        """Initialize MobileNetV3-Large backbone.

        Args:
            pretrained: If True, load ImageNet-pretrained weights (IMAGENET1K_V2).
        """
        super().__init__()

        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        mobilenet = models.mobilenet_v3_large(weights=weights)

        # Split features into three sequential stages for multi-scale extraction
        self.stage1 = nn.Sequential(*list(mobilenet.features[:self._C3_END + 1]))  # layers 0-6
        self.stage2 = nn.Sequential(*list(mobilenet.features[self._C3_END + 1:self._C4_END + 1]))  # layers 7-12
        self.stage3 = nn.Sequential(*list(mobilenet.features[self._C4_END + 1:self._C5_END + 1]))  # layers 13-16

    @property
    def out_channels(self) -> List[int]:
        """Return output channel counts for [C3, C4, C5].

        Returns:
            List of channel dimensions at each extraction level.
        """
        return [self.C3_CHANNELS, self.C4_CHANNELS, self.C5_CHANNELS]

    def freeze_layers(self, n: int) -> None:
        """Freeze the first n layers of the backbone.

        Freezes parameters in the underlying MobileNetV3 features sequentially.
        Layers are counted across all three stages continuously (0 through 16).

        Args:
            n: Number of layers to freeze from the start. Must be >= 0 and <= 17.
        """
        all_layers = list(self.stage1) + list(self.stage2) + list(self.stage3)
        n = min(n, len(all_layers))
        for i in range(n):
            for param in all_layers[i].parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from the input image tensor.

        Args:
            x: Input tensor of shape (B, 3, H, W). Expected H=W=550.

        Returns:
            Dictionary with keys 'C3', 'C4', 'C5' mapping to feature tensors:
                - C3: (B, 40,  H/8,  W/8)   e.g. (B, 40,  69, 69)
                - C4: (B, 112, H/16, W/16)  e.g. (B, 112, 35, 35)
                - C5: (B, 960, H/32, W/32)  e.g. (B, 960, 18, 18)
        """
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)

        return {'C3': c3, 'C4': c4, 'C5': c5}


if __name__ == '__main__':
    # Verify backbone output shapes and parameter count
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    backbone = MobileNetV3Backbone(pretrained=False).to(device)
    dummy_input = torch.randn(1, 3, 550, 550, device=device)

    with torch.no_grad():
        features = backbone(dummy_input)

    print("\nBackbone output shapes:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")

    print(f"\nExpected out_channels: {backbone.out_channels}")

    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test freeze
    backbone.freeze_layers(4)
    trainable_after_freeze = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Trainable after freezing 4 layers: {trainable_after_freeze:,}")
