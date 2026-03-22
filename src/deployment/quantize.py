"""ONNX model quantization to INT8.

Applies post-training dynamic quantization to reduce model size
and (on supported hardware) improve inference throughput.

Usage:
    python -m src.deployment.quantize                        # defaults
    python -m src.deployment.quantize --input model.onnx     # custom path
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def quantize_model(
    input_path: str,
    output_path: Optional[str] = None,
    method: str = 'dynamic',
) -> Optional[str]:
    """Quantize an ONNX model to INT8.

    Supports dynamic quantization via onnxruntime.quantization.

    Args:
        input_path: Path to the FP32 ONNX model.
        output_path: Path for the quantized model. If None, derived from
            input_path by appending '_int8'.
        method: Quantization method. Currently supports 'dynamic'.

    Returns:
        Path to the quantized model on success, None on failure.
    """
    if not os.path.isfile(input_path):
        logger.error(f"Input model not found: {input_path}")
        return None

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_int8{ext}"

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.error(
            "onnxruntime.quantization not available. "
            "Install with: pip install onnxruntime"
        )
        return None

    logger.info(f"Quantizing {input_path} -> {output_path}")
    logger.info(f"Method: {method}")

    try:
        if method == 'dynamic':
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=QuantType.QUInt8,
            )
        else:
            logger.error(f"Unsupported quantization method: {method}")
            return None
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return None

    # Print size comparison
    fp32_size = os.path.getsize(input_path) / (1024 * 1024)
    int8_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1.0 - int8_size / fp32_size) * 100 if fp32_size > 0 else 0.0

    logger.info(f"FP32 model size : {fp32_size:.2f} MB")
    logger.info(f"INT8 model size : {int8_size:.2f} MB")
    logger.info(f"Size reduction  : {reduction:.1f}%")

    print("\n" + "=" * 50)
    print("Quantization Results")
    print("=" * 50)
    print(f"  Input  (FP32) : {fp32_size:.2f} MB  ({input_path})")
    print(f"  Output (INT8) : {int8_size:.2f} MB  ({output_path})")
    print(f"  Reduction     : {reduction:.1f}%")
    print("=" * 50)

    return output_path


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    parser = argparse.ArgumentParser(description='Quantize ONNX model to INT8')
    parser.add_argument(
        '--input', type=str,
        default='results/deployment/yolact.onnx',
        help='Path to the FP32 ONNX model.',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path for quantized model. Auto-derived if omitted.',
    )
    parser.add_argument(
        '--method', type=str, default='dynamic',
        choices=['dynamic'],
        help='Quantization method.',
    )
    args = parser.parse_args()

    result = quantize_model(args.input, args.output, args.method)
    if result:
        print(f"\nQuantized model saved to: {result}")
    else:
        print("\nQuantization failed.")
        sys.exit(1)
