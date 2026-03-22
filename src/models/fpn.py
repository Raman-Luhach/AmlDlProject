"""Feature Pyramid Network (FPN) for multi-scale feature fusion.

Builds a feature pyramid from backbone outputs (C3, C4, C5) by applying
lateral connections and top-down pathway with element-wise addition.
Produces 5 output levels (P3-P7) all with the same channel dimension.

Output levels for 550x550 input:
    P3: (B, 256, 69, 69)   stride 8
    P4: (B, 256, 35, 35)   stride 16
    P5: (B, 256, 18, 18)   stride ~32
    P6: (B, 256, 9, 9)     stride ~64
    P7: (B, 256, 5, 5)     stride ~128
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion.

    Takes backbone features at multiple scales and produces a feature pyramid
    with a uniform channel dimension. Uses top-down pathway with lateral
    connections, plus extra downsampling levels for detection.

    Args:
        in_channels_list: List of input channel counts from backbone [C3, C4, C5].
        out_channels: Number of output channels for all pyramid levels.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
    ) -> None:
        """Initialize FPN.

        Args:
            in_channels_list: Channel dimensions from backbone, e.g. [40, 112, 960].
            out_channels: Uniform output channel dimension for all pyramid levels.
        """
        super().__init__()
        assert len(in_channels_list) == 3, "Expected 3 input feature levels (C3, C4, C5)"

        self.out_channels = out_channels

        # Lateral connections: 1x1 convolutions to reduce channels
        self.lateral_c3 = nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1)

        # Smoothing convolutions: 3x3 conv after merging to reduce aliasing
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Extra downsampling levels for small object detection
        self.downsample_p6 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.downsample_p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize convolution weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Build feature pyramid from backbone features.

        Top-down pathway: Start from the coarsest level (C5), apply lateral
        connection, then upsample and add to the next finer level. Apply
        smoothing convolutions to reduce aliasing artifacts.

        Args:
            features: Dictionary with keys 'C3', 'C4', 'C5' containing
                backbone feature tensors.

        Returns:
            List of 5 feature tensors [P3, P4, P5, P6, P7], all with
            `out_channels` channels, at decreasing spatial resolutions.
        """
        c3 = features['C3']
        c4 = features['C4']
        c5 = features['C5']

        # Lateral connections
        lat_c5 = self.lateral_c5(c5)
        lat_c4 = self.lateral_c4(c4)
        lat_c3 = self.lateral_c3(c3)

        # Top-down pathway with element-wise addition
        # Upsample lat_c5 to match lat_c4 spatial size
        p4 = lat_c4 + F.interpolate(lat_c5, size=lat_c4.shape[2:], mode='bilinear', align_corners=False)
        # Upsample merged p4 to match lat_c3 spatial size
        p3 = lat_c3 + F.interpolate(p4, size=lat_c3.shape[2:], mode='bilinear', align_corners=False)

        # Apply smoothing convolutions
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(lat_c5)

        # Extra downsampling levels from P5
        p6 = self.downsample_p6(p5)
        p7 = self.downsample_p7(F.relu(p6))

        return [p3, p4, p5, p6, p7]


if __name__ == '__main__':
    # Verify FPN output shapes
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Simulate backbone outputs for 550x550 input
    in_channels = [40, 112, 960]
    features = {
        'C3': torch.randn(1, 40, 69, 69, device=device),
        'C4': torch.randn(1, 112, 35, 35, device=device),
        'C5': torch.randn(1, 960, 18, 18, device=device),
    }

    fpn = FPN(in_channels_list=in_channels, out_channels=256).to(device)

    with torch.no_grad():
        pyramid = fpn(features)

    print("\nFPN output shapes:")
    for i, p in enumerate(pyramid):
        level = i + 3
        print(f"  P{level}: {p.shape}")

    total_params = sum(p.numel() for p in fpn.parameters())
    print(f"\nFPN parameters: {total_params:,}")
