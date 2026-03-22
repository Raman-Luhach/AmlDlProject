"""Inference benchmarking for PyTorch and ONNX models.

Measures latency, throughput, and model size for:
  - PyTorch FP32
  - ONNX FP32  (via onnxruntime)
  - ONNX INT8  (via onnxruntime, if quantized model exists)

Results are saved to a JSON file for comparison.

Usage:
    python -m src.deployment.benchmark
    python -m src.deployment.benchmark --num-runs 100
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def _get_model_size_mb(path: str) -> float:
    """Return file size in megabytes.

    Args:
        path: File path.

    Returns:
        Size in MB, or 0 if file does not exist.
    """
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def _pytorch_model_size_mb(model: torch.nn.Module) -> float:
    """Estimate PyTorch model size from its state dict.

    Args:
        model: PyTorch model.

    Returns:
        Estimated size in MB.
    """
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    for buf in model.buffers():
        total_bytes += buf.nelement() * buf.element_size()
    return total_bytes / (1024 * 1024)


def benchmark_pytorch(
    model: torch.nn.Module,
    input_size: int = 550,
    num_warmup: int = 10,
    num_runs: int = 50,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Benchmark PyTorch FP32 inference.

    Args:
        model: YOLACT model instance.
        input_size: Spatial size of the square input.
        num_warmup: Number of warmup iterations.
        num_runs: Number of timed iterations.
        device: Device for benchmarking. Auto-detected if None.

    Returns:
        Dictionary with latency_ms, fps, model_size_mb.
    """
    from src.utils.helpers import get_device

    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size, device=device)

    # Warmup
    logger.info(f"PyTorch warmup ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)

    # Synchronise before timing on MPS/CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

    # Timed runs
    logger.info(f"PyTorch benchmarking ({num_runs} iterations)...")
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    model_size = _pytorch_model_size_mb(model)

    result = {
        'backend': 'PyTorch FP32',
        'device': str(device),
        'avg_latency_ms': round(avg_latency, 2),
        'std_latency_ms': round(std_latency, 2),
        'fps': round(fps, 1),
        'model_size_mb': round(model_size, 2),
        'num_runs': num_runs,
    }

    logger.info(
        f"PyTorch FP32 | {avg_latency:.1f} +/- {std_latency:.1f} ms | "
        f"{fps:.1f} FPS | {model_size:.1f} MB"
    )

    return result


def benchmark_onnx(
    model_path: str,
    label: str = 'ONNX FP32',
    input_size: int = 550,
    num_warmup: int = 10,
    num_runs: int = 50,
) -> Optional[Dict[str, Any]]:
    """Benchmark an ONNX model using ONNX Runtime.

    Args:
        model_path: Path to the .onnx model.
        label: Human-readable label for the result.
        input_size: Spatial size of the square input.
        num_warmup: Number of warmup iterations.
        num_runs: Number of timed iterations.

    Returns:
        Dictionary with benchmark results, or None if model not found
        or ONNX Runtime is unavailable.
    """
    if not os.path.isfile(model_path):
        logger.warning(f"ONNX model not found: {model_path}")
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed; skipping ONNX benchmark.")
        return None

    # Use CPU for consistent benchmarking (CoreML provider may not support
    # all ops from quantized models)
    providers = ['CPUExecutionProvider']
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        return None

    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    # Warmup
    logger.info(f"{label} warmup ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy})

    # Timed runs
    logger.info(f"{label} benchmarking ({num_runs} iterations)...")
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    model_size = _get_model_size_mb(model_path)

    result = {
        'backend': label,
        'device': 'CPU (ONNX Runtime)',
        'avg_latency_ms': round(avg_latency, 2),
        'std_latency_ms': round(std_latency, 2),
        'fps': round(fps, 1),
        'model_size_mb': round(model_size, 2),
        'num_runs': num_runs,
        'model_path': model_path,
    }

    logger.info(
        f"{label} | {avg_latency:.1f} +/- {std_latency:.1f} ms | "
        f"{fps:.1f} FPS | {model_size:.1f} MB"
    )

    return result


def benchmark_inference(
    model_paths: Dict[str, str],
    pytorch_model: Optional[torch.nn.Module] = None,
    input_size: int = 550,
    num_warmup: int = 10,
    num_runs: int = 50,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Benchmark inference for PyTorch and ONNX models.

    Args:
        model_paths: Dictionary mapping labels to ONNX model paths, e.g.:
            {'ONNX FP32': 'model.onnx', 'ONNX INT8': 'model_int8.onnx'}
        pytorch_model: Optional PyTorch model for comparison.
        input_size: Spatial input dimension.
        num_warmup: Warmup iterations.
        num_runs: Timed iterations.
        output_path: If provided, save results JSON here.

    Returns:
        List of benchmark result dictionaries.
    """
    results: List[Dict[str, Any]] = []

    # PyTorch benchmark
    if pytorch_model is not None:
        pt_result = benchmark_pytorch(
            pytorch_model,
            input_size=input_size,
            num_warmup=num_warmup,
            num_runs=num_runs,
        )
        results.append(pt_result)

    # ONNX benchmarks
    for label, path in model_paths.items():
        onnx_result = benchmark_onnx(
            path,
            label=label,
            input_size=input_size,
            num_warmup=num_warmup,
            num_runs=num_runs,
        )
        if onnx_result is not None:
            results.append(onnx_result)

    # Print comparison table
    _print_benchmark_table(results)

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Benchmark results saved to {output_path}")

    return results


def _print_benchmark_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table of benchmark results.

    Args:
        results: List of benchmark result dictionaries.
    """
    if not results:
        print("No benchmark results to display.")
        return

    print("\n" + "=" * 78)
    print("INFERENCE BENCHMARK RESULTS")
    print("=" * 78)
    header = f"{'Backend':<22s} {'Device':<22s} {'Latency (ms)':<16s} {'FPS':<10s} {'Size (MB)':<10s}"
    print(header)
    print("-" * 78)

    for r in results:
        latency_str = f"{r['avg_latency_ms']:.1f} +/- {r.get('std_latency_ms', 0):.1f}"
        print(
            f"{r['backend']:<22s} "
            f"{r['device']:<22s} "
            f"{latency_str:<16s} "
            f"{r['fps']:<10.1f} "
            f"{r['model_size_mb']:<10.1f}"
        )

    print("=" * 78)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    parser = argparse.ArgumentParser(description='Benchmark YOLACT inference')
    parser.add_argument('--input-size', type=int, default=550)
    parser.add_argument('--num-warmup', type=int, default=10)
    parser.add_argument('--num-runs', type=int, default=50)
    parser.add_argument('--onnx-fp32', type=str, default='results/deployment/yolact.onnx')
    parser.add_argument('--onnx-int8', type=str, default='results/deployment/yolact_int8.onnx')
    parser.add_argument('--output', type=str, default='results/deployment/benchmark.json')
    parser.add_argument('--skip-pytorch', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    # Load PyTorch model
    pytorch_model = None
    if not args.skip_pytorch:
        from src.models.yolact import YOLACT
        pytorch_model = YOLACT(config={'pretrained_backbone': True})
        if args.checkpoint and os.path.isfile(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            pytorch_model.load_state_dict(state, strict=False)

    model_paths = {}
    if os.path.isfile(args.onnx_fp32):
        model_paths['ONNX FP32'] = args.onnx_fp32
    if os.path.isfile(args.onnx_int8):
        model_paths['ONNX INT8'] = args.onnx_int8

    # Also check for backbone-only fallback
    fallback_path = args.onnx_fp32.replace('.onnx', '_backbone_fpn.onnx')
    if os.path.isfile(fallback_path) and 'ONNX FP32' not in model_paths:
        model_paths['ONNX FP32 (backbone+FPN)'] = fallback_path

    benchmark_inference(
        model_paths=model_paths,
        pytorch_model=pytorch_model,
        input_size=args.input_size,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        output_path=args.output,
    )
