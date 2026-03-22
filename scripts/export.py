#!/usr/bin/env python3
"""Orchestration script: ONNX export, INT8 quantization, and benchmarking.

This script runs the full deployment pipeline:
    1. Load or create the YOLACT model
    2. Export to ONNX format (FP32)
    3. Quantize to INT8
    4. Run inference benchmarks (PyTorch FP32, ONNX FP32, ONNX INT8)
    5. Save all results to results/deployment/

Usage:
    python scripts/export.py                         # auto-find checkpoint
    python scripts/export.py --checkpoint best.pth   # specific checkpoint
    python scripts/export.py --untrained             # random weights
    python scripts/export.py --skip-benchmark        # export only
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.yolact import YOLACT, DEFAULT_CONFIG
from src.deployment.export_onnx import export_to_onnx
from src.deployment.quantize import quantize_model
from src.deployment.benchmark import benchmark_inference
from src.utils.helpers import get_device

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Export YOLACT: ONNX export, quantization, and benchmarking.'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to model checkpoint (.pth). Auto-detected if omitted.',
    )
    parser.add_argument(
        '--untrained', action='store_true',
        help='Use untrained model (random weights) for testing.',
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/deployment',
        help='Directory for saving deployment artifacts.',
    )
    parser.add_argument(
        '--input-size', type=int, default=550,
        help='Input spatial dimension.',
    )
    parser.add_argument(
        '--opset', type=int, default=11,
        help='ONNX opset version.',
    )
    parser.add_argument(
        '--skip-quantize', action='store_true',
        help='Skip INT8 quantization step.',
    )
    parser.add_argument(
        '--skip-benchmark', action='store_true',
        help='Skip benchmarking step.',
    )
    parser.add_argument(
        '--num-warmup', type=int, default=10,
        help='Number of warmup iterations for benchmarking.',
    )
    parser.add_argument(
        '--num-runs', type=int, default=50,
        help='Number of timed iterations for benchmarking.',
    )
    return parser.parse_args()


def find_checkpoint() -> str:
    """Search common locations for a model checkpoint.

    Returns:
        Path to the checkpoint file, or empty string if not found.
    """
    search_dirs = [
        str(PROJECT_ROOT / 'weights'),
        str(PROJECT_ROOT / 'checkpoints'),
        str(PROJECT_ROOT / 'results' / 'training'),
    ]
    preferred = ['best.pth', 'best_model.pth', 'latest.pth', 'checkpoint.pth']

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for name in preferred:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path
        for fname in sorted(os.listdir(d)):
            if fname.endswith('.pth'):
                return os.path.join(d, fname)
    return ''


def main() -> None:
    """Run the full deployment pipeline."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pipeline_start = time.time()

    # ========================================================
    # Step 1: Load or create the model
    # ========================================================
    print("\n" + "=" * 60)
    print("Step 1/4: Loading model")
    print("=" * 60)

    checkpoint_path = args.checkpoint
    if not checkpoint_path and not args.untrained:
        checkpoint_path = find_checkpoint()
        if checkpoint_path:
            logger.info(f"Auto-detected checkpoint: {checkpoint_path}")

    config = {**DEFAULT_CONFIG, 'pretrained_backbone': not args.untrained}
    model = YOLACT(config=config)

    if checkpoint_path and os.path.isfile(checkpoint_path) and not args.untrained:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        logger.info("Checkpoint loaded.")
    elif args.untrained:
        logger.info("Using untrained model (random weights).")
    else:
        logger.info("No checkpoint found. Using ImageNet-pretrained backbone.")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # ========================================================
    # Step 2: ONNX export
    # ========================================================
    print("\n" + "=" * 60)
    print("Step 2/4: ONNX Export")
    print("=" * 60)

    onnx_fp32_path = os.path.join(args.output_dir, 'yolact.onnx')
    export_success = export_to_onnx(
        model,
        output_path=onnx_fp32_path,
        input_size=args.input_size,
        opset=args.opset,
    )

    if not export_success:
        logger.warning("ONNX export failed. Checking for fallback file...")
        fallback = onnx_fp32_path.replace('.onnx', '_backbone_fpn.onnx')
        if os.path.isfile(fallback):
            onnx_fp32_path = fallback
            export_success = True

    # ========================================================
    # Step 3: INT8 Quantization
    # ========================================================
    onnx_int8_path = None
    if not args.skip_quantize:
        print("\n" + "=" * 60)
        print("Step 3/4: INT8 Quantization")
        print("=" * 60)

        if export_success and os.path.isfile(onnx_fp32_path):
            onnx_int8_path = quantize_model(
                input_path=onnx_fp32_path,
                method='dynamic',
            )
            if onnx_int8_path is None:
                logger.warning("Quantization failed.")
        else:
            logger.warning("Skipping quantization: no ONNX model available.")
    else:
        print("\n  [Skipping quantization as requested]")

    # ========================================================
    # Step 4: Benchmarking
    # ========================================================
    benchmark_results = []
    if not args.skip_benchmark:
        print("\n" + "=" * 60)
        print("Step 4/4: Benchmarking")
        print("=" * 60)

        # Rebuild model on device for PyTorch benchmarking
        # (export_to_onnx moves it to CPU)
        device = get_device()
        model = model.to(device)
        model.eval()

        onnx_models = {}
        if export_success and os.path.isfile(onnx_fp32_path):
            onnx_models['ONNX FP32'] = onnx_fp32_path
        if onnx_int8_path and os.path.isfile(onnx_int8_path):
            onnx_models['ONNX INT8'] = onnx_int8_path

        benchmark_output = os.path.join(args.output_dir, 'benchmark.json')
        benchmark_results = benchmark_inference(
            model_paths=onnx_models,
            pytorch_model=model,
            input_size=args.input_size,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            output_path=benchmark_output,
        )
    else:
        print("\n  [Skipping benchmark as requested]")

    # ========================================================
    # Summary
    # ========================================================
    pipeline_time = time.time() - pipeline_start

    summary = {
        'checkpoint': checkpoint_path or 'none',
        'untrained': args.untrained,
        'total_params': total_params,
        'onnx_export': export_success,
        'onnx_fp32_path': onnx_fp32_path if export_success else None,
        'onnx_int8_path': onnx_int8_path,
        'benchmark_results': benchmark_results,
        'pipeline_time_s': round(pipeline_time, 1),
    }

    summary_path = os.path.join(args.output_dir, 'deployment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("DEPLOYMENT PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  ONNX export       : {'OK' if export_success else 'FAILED'}")
    if export_success:
        onnx_size = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
        print(f"  ONNX FP32 size    : {onnx_size:.1f} MB")
    if onnx_int8_path and os.path.isfile(onnx_int8_path):
        int8_size = os.path.getsize(onnx_int8_path) / (1024 * 1024)
        print(f"  ONNX INT8 size    : {int8_size:.1f} MB")
    else:
        print(f"  ONNX INT8         : {'skipped' if args.skip_quantize else 'not available'}")
    if benchmark_results:
        for r in benchmark_results:
            print(f"  {r['backend']:<20s}: {r['avg_latency_ms']:.1f} ms, {r['fps']:.1f} FPS")
    print(f"  Pipeline time     : {pipeline_time:.1f}s")
    print(f"  Output directory  : {args.output_dir}")
    print(f"  Summary           : {summary_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
