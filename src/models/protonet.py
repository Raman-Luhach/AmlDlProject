"""Prototype Mask Network (ProtoNet) for YOLACT.

Generates a set of prototype masks from the finest FPN level (P3).
These prototypes are linearly combined using per-instance mask coefficients
from the prediction head to produce final instance masks.

Architecture:
    3x3 conv(256->256), BN, ReLU
    3x3 conv(256->256), BN, ReLU
    3x3 conv(256->256), BN, ReLU
    Bilinear upsample 2x
    3x3 conv(256->256), BN, ReLU
    1x1 conv(256->32), ReLU  (non-negative prototypes)

For 550x550 input with P3 at 69x69:
    Output: (B, 32, 138, 138) ~ input_size / 4
"""

import torch
import torch.nn as nn


class ProtoNet(nn.Module):
    """Prototype mask generation network.

    Takes the P3 feature map from FPN and produces a set of prototype masks
    that serve as a basis for instance-level mask generation. The final
    instance masks are computed as a linear combination of these prototypes,
    weighted by per-detection mask coefficients.

    Args:
        in_channels: Number of input channels from P3 feature map.
        hidden_channels: Number of channels in intermediate layers.
        num_prototypes: Number of prototype masks to generate.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        num_prototypes: int = 32,
    ) -> None:
        """Initialize ProtoNet.

        Args:
            in_channels: Input channels from P3 (typically 256 from FPN).
            hidden_channels: Hidden layer channels (256).
            num_prototypes: Number of output prototype masks (32).
        """
        super().__init__()
        self.num_prototypes = num_prototypes

        self.proto_net = nn.Sequential(
            # Block 1: 3x3 conv + BN + ReLU
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # Block 2: 3x3 conv + BN + ReLU
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # Block 3: 3x3 conv + BN + ReLU
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # Bilinear upsample 2x
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # Block 4: 3x3 conv + BN + ReLU (after upsample)
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            # Final 1x1 conv to produce prototypes + ReLU for non-negativity
            nn.Conv2d(hidden_channels, num_prototypes, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming normal for ReLU layers."""
        for module in self.proto_net.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, p3: torch.Tensor) -> torch.Tensor:
        """Generate prototype masks from P3 features.

        Args:
            p3: P3 feature tensor of shape (B, in_channels, H, W).
                For 550x550 input, this is (B, 256, 69, 69).

        Returns:
            Prototype masks of shape (B, num_prototypes, 2*H, 2*W).
            For 550x550 input: (B, 32, 138, 138).
        """
        return self.proto_net(p3)


if __name__ == '__main__':
    # Verify ProtoNet output shapes
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    protonet = ProtoNet(in_channels=256, num_prototypes=32).to(device)

    # Simulate P3 from FPN (550x550 input -> 69x69 at stride 8)
    p3 = torch.randn(1, 256, 69, 69, device=device)

    with torch.no_grad():
        prototypes = protonet(p3)

    print(f"\nInput P3 shape:  {p3.shape}")
    print(f"Output prototypes shape: {prototypes.shape}")
    print(f"Expected: (1, 32, 138, 138)")

    total_params = sum(p.numel() for p in protonet.parameters())
    print(f"\nProtoNet parameters: {total_params:,}")
