"""ONNX export for the YOLACT model.

Exports the full model (backbone + FPN + prediction heads) or, if the full
model export fails (e.g., due to unsupported ops in post-processing),
falls back to exporting only the feature-extraction portion.

ONNX export is done **without** NMS / post-processing so the graph stays
ONNX-friendly. Post-processing should be applied at inference time on the
ONNX runtime side.

Usage:
    python -m src.deployment.export_onnx                       # defaults
    python -m src.deployment.export_onnx --output model.onnx   # custom path
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class YOLACTExportWrapper(nn.Module):
    """Wrapper that exposes YOLACT raw outputs for ONNX export.

    Runs the model in *training* mode (bypassing post-processing) so
    that the graph contains only standard ops and avoids Soft-NMS /
    dynamic slicing that may fail during tracing.

    Outputs:
        class_preds  (B, N, num_classes)
        box_preds    (B, N, 4)
        mask_coeffs  (B, N, num_prototypes)
        prototypes   (B, num_prototypes, H, W)
    """

    def __init__(self, model: nn.Module) -> None:
        """Wrap a YOLACT model.

        Args:
            model: Full YOLACT model instance.
        """
        super().__init__()
        self.backbone = model.backbone
        self.fpn = model.fpn
        self.protonet = model.protonet
        self.prediction_head = model.prediction_head

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass producing raw predictions.

        Args:
            images: (B, 3, H, W) input tensor.

        Returns:
            Tuple of (class_preds, box_preds, mask_coeffs, prototypes).
        """
        backbone_features = self.backbone(images)
        fpn_features = self.fpn(backbone_features)
        prototypes = self.protonet(fpn_features[0])
        class_preds, box_preds, mask_coeffs = self.prediction_head(fpn_features)
        return class_preds, box_preds, mask_coeffs, prototypes


class BackboneFPNWrapper(nn.Module):
    """Minimal wrapper that exports only backbone + FPN.

    Used as a fallback when the full model cannot be exported.
    """

    def __init__(self, model: nn.Module) -> None:
        """Wrap backbone and FPN.

        Args:
            model: Full YOLACT model instance.
        """
        super().__init__()
        self.backbone = model.backbone
        self.fpn = model.fpn

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass producing FPN features.

        Args:
            images: (B, 3, H, W) input tensor.

        Returns:
            Tuple of FPN feature maps (P3, P4, P5, P6, P7).
        """
        backbone_features = self.backbone(images)
        fpn_features = self.fpn(backbone_features)
        return tuple(fpn_features)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: int = 550,
    opset: int = 11,
    verify: bool = True,
) -> bool:
    """Export YOLACT to ONNX format.

    Attempts full-model export (backbone + FPN + heads) first. If that
    fails, falls back to backbone + FPN only.

    Args:
        model: YOLACT model instance (on CPU).
        output_path: Destination path for the .onnx file.
        input_size: Spatial input size (assumed square).
        opset: ONNX opset version.
        verify: If True, verify the exported model with onnx.checker.

    Returns:
        True if export succeeded, False otherwise.
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Move model to CPU for export
    model_cpu = model.cpu()
    model_cpu.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    # ---- Attempt 1: full model (without post-processing) ----
    logger.info("Attempting full YOLACT ONNX export (backbone + FPN + heads)...")
    wrapper = YOLACTExportWrapper(model_cpu)
    wrapper.eval()

    success = _try_export(
        wrapper,
        dummy_input,
        output_path,
        opset=opset,
        input_names=['images'],
        output_names=['class_preds', 'box_preds', 'mask_coeffs', 'prototypes'],
        dynamic_axes={
            'images': {0: 'batch'},
            'class_preds': {0: 'batch'},
            'box_preds': {0: 'batch'},
            'mask_coeffs': {0: 'batch'},
            'prototypes': {0: 'batch'},
        },
    )

    if success:
        logger.info(f"Full model exported successfully to {output_path}")
        if verify:
            _verify_onnx(output_path)
        _print_model_size(output_path)
        return True

    # ---- Attempt 2: backbone + FPN only ----
    fallback_path = output_path.replace('.onnx', '_backbone_fpn.onnx')
    logger.warning("Full model export failed. Trying backbone+FPN fallback...")
    fallback = BackboneFPNWrapper(model_cpu)
    fallback.eval()

    output_names_fpn = [f'P{i}' for i in range(3, 8)]
    dynamic_axes_fpn = {'images': {0: 'batch'}}
    for name in output_names_fpn:
        dynamic_axes_fpn[name] = {0: 'batch'}

    success = _try_export(
        fallback,
        dummy_input,
        fallback_path,
        opset=opset,
        input_names=['images'],
        output_names=output_names_fpn,
        dynamic_axes=dynamic_axes_fpn,
    )

    if success:
        logger.info(f"Backbone+FPN exported to {fallback_path}")
        if verify:
            _verify_onnx(fallback_path)
        _print_model_size(fallback_path)
        return True

    logger.error("Both export attempts failed.")
    return False


def _try_export(
    module: nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    opset: int,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
) -> bool:
    """Attempt ONNX export and return success status.

    Args:
        module: Module to export.
        dummy_input: Example input tensor.
        output_path: Path for the exported file.
        opset: ONNX opset version.
        input_names: Names of input tensors.
        output_names: Names of output tensors.
        dynamic_axes: Dynamic axis specification.

    Returns:
        True if export succeeded, False otherwise.
    """
    try:
        torch.onnx.export(
            module,
            dummy_input,
            output_path,
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
        )
        return True
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return False


def _verify_onnx(path: str) -> bool:
    """Verify an ONNX model with onnx.checker.

    Args:
        path: Path to the .onnx file.

    Returns:
        True if verification passes, False otherwise.
    """
    try:
        import onnx
        model = onnx.load(path)
        onnx.checker.check_model(model)
        logger.info(f"ONNX model verification passed: {path}")
        return True
    except ImportError:
        logger.warning("onnx package not installed; skipping verification.")
        return False
    except Exception as e:
        logger.warning(f"ONNX verification failed: {e}")
        return False


def _print_model_size(path: str) -> None:
    """Log the file size of an ONNX model.

    Args:
        path: Path to the .onnx file.
    """
    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"Model size: {size_mb:.2f} MB ({path})")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    parser = argparse.ArgumentParser(description='Export YOLACT to ONNX')
    parser.add_argument('--output', type=str, default='results/deployment/yolact.onnx')
    parser.add_argument('--input-size', type=int, default=550)
    parser.add_argument('--opset', type=int, default=11)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    from src.models.yolact import YOLACT

    model = YOLACT(config={'pretrained_backbone': True})

    if args.checkpoint and os.path.isfile(args.checkpoint):
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)

    export_to_onnx(model, args.output, input_size=args.input_size, opset=args.opset)
