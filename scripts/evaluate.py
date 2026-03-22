#!/usr/bin/env python3
"""Evaluate the YOLACT model on the SKU-110K validation set.

Usage:
    python scripts/evaluate.py                          # use trained checkpoint
    python scripts/evaluate.py --checkpoint best.pth    # specific checkpoint
    python scripts/evaluate.py --untrained              # evaluate untrained model
    python scripts/evaluate.py --max-images 50          # quick test on subset

Outputs (saved to results/eval/):
    - metrics.json              : all COCO-style metrics
    - detection_samples.png     : grid of 8 sample detections
    - precision_recall.png      : PR curves (if enough data)
    - density_analysis.png      : per-density-bucket analysis (if enough data)
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

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.yolact import YOLACT, DEFAULT_CONFIG
from src.evaluation.evaluator import COCOEvaluator
from src.evaluation.metrics import compute_detection_metrics
from src.utils.helpers import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate YOLACT on SKU-110K validation set.'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to model checkpoint (.pth). Auto-detected if omitted.',
    )
    parser.add_argument(
        '--untrained', action='store_true',
        help='Evaluate an untrained model (random weights) for testing.',
    )
    parser.add_argument(
        '--data-dir', type=str, default='data',
        help='Root data directory containing SKU110K_fixed/.',
    )
    parser.add_argument(
        '--max-images', type=int, default=None,
        help='Maximum number of validation images to evaluate.',
    )
    parser.add_argument(
        '--batch-size', type=int, default=4,
        help='Batch size for inference.',
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/eval',
        help='Directory for saving evaluation outputs.',
    )
    parser.add_argument(
        '--score-threshold', type=float, default=0.05,
        help='Minimum score threshold for detections.',
    )
    parser.add_argument(
        '--num-samples', type=int, default=8,
        help='Number of sample images for the detection grid.',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility.',
    )
    return parser.parse_args()


def find_checkpoint(search_dirs: Optional[List[str]] = None) -> Optional[str]:
    """Auto-detect the best available checkpoint.

    Searches in common locations for checkpoint files, preferring
    'best.pth' over 'latest.pth' over any other .pth file.

    Args:
        search_dirs: List of directories to search.

    Returns:
        Path to the checkpoint file, or None if not found.
    """
    if search_dirs is None:
        search_dirs = [
            str(PROJECT_ROOT / 'weights'),
            str(PROJECT_ROOT / 'checkpoints'),
            str(PROJECT_ROOT / 'results' / 'training'),
        ]

    preferred = ['best.pth', 'best_model.pth', 'latest.pth', 'checkpoint.pth']

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        # Try preferred names first
        for name in preferred:
            path = os.path.join(search_dir, name)
            if os.path.isfile(path):
                return path
        # Fall back to any .pth file
        for fname in sorted(os.listdir(search_dir)):
            if fname.endswith('.pth'):
                return os.path.join(search_dir, fname)

    return None


def load_model(
    checkpoint_path: Optional[str],
    device: torch.device,
    untrained: bool = False,
) -> YOLACT:
    """Load the YOLACT model, optionally from a checkpoint.

    Args:
        checkpoint_path: Path to a .pth checkpoint.
        device: Device to load the model onto.
        untrained: If True, skip loading checkpoint weights.

    Returns:
        YOLACT model in eval mode.
    """
    config = {**DEFAULT_CONFIG, 'pretrained_backbone': not untrained}
    model = YOLACT(config=config)

    if checkpoint_path and not untrained:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False
        )
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Checkpoint loaded successfully.")
    elif untrained:
        logger.info("Using untrained model (random weights) for testing.")
    else:
        logger.info("No checkpoint found. Using ImageNet-pretrained backbone only.")

    model = model.to(device)
    model.eval()
    return model


def create_synthetic_data(
    num_images: int = 20,
    input_size: int = 550,
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Create synthetic validation data for testing without the real dataset.

    Args:
        num_images: Number of synthetic images.
        input_size: Image spatial dimension.

    Returns:
        images: List of (3, H, W) tensors.
        targets: List of target dicts with 'boxes', 'labels', 'image_id'.
    """
    rng = np.random.RandomState(42)
    images = []
    targets = []

    for i in range(num_images):
        img = torch.randn(3, input_size, input_size)
        images.append(img)

        n_gt = rng.randint(3, 30)
        boxes = np.zeros((n_gt, 4), dtype=np.float32)
        boxes[:, 0] = rng.uniform(0, input_size * 0.7, n_gt)
        boxes[:, 1] = rng.uniform(0, input_size * 0.7, n_gt)
        boxes[:, 2] = boxes[:, 0] + rng.uniform(20, input_size * 0.3, n_gt)
        boxes[:, 3] = boxes[:, 1] + rng.uniform(20, input_size * 0.3, n_gt)
        # Clip to image
        boxes[:, 2] = np.minimum(boxes[:, 2], input_size)
        boxes[:, 3] = np.minimum(boxes[:, 3], input_size)

        target = {
            'boxes': torch.from_numpy(boxes),
            'labels': torch.ones(n_gt, dtype=torch.int64),
            'image_id': torch.tensor(i, dtype=torch.int64),
        }
        targets.append(target)

    return images, targets


@torch.no_grad()
def run_inference(
    model: YOLACT,
    images: List[torch.Tensor],
    device: torch.device,
    batch_size: int = 4,
    input_size: int = 550,
) -> List[Dict[str, torch.Tensor]]:
    """Run model inference on a list of images.

    Args:
        model: YOLACT model in eval mode.
        images: List of (3, H, W) image tensors.
        device: Compute device.
        batch_size: Batch size for inference.
        input_size: Expected spatial size.

    Returns:
        List of detection dicts, one per image.
    """
    all_detections: List[Dict[str, torch.Tensor]] = []
    num_images = len(images)

    for start in range(0, num_images, batch_size):
        end = min(start + batch_size, num_images)
        batch_imgs = images[start:end]

        # Stack into batch
        batch = torch.stack(batch_imgs).to(device)

        try:
            detections = model(batch)
        except Exception as e:
            logger.warning(f"Inference failed on batch {start}-{end}: {e}")
            # Return empty detections for this batch
            for _ in range(end - start):
                all_detections.append({
                    'boxes': torch.zeros(0, 4),
                    'scores': torch.zeros(0),
                    'labels': torch.zeros(0, dtype=torch.long),
                    'masks': torch.zeros(0, 138, 138),
                })
            continue

        # Move results to CPU
        for det in detections:
            cpu_det = {}
            for k, v in det.items():
                cpu_det[k] = v.detach().cpu()
            all_detections.append(cpu_det)

        if (start // batch_size + 1) % 10 == 0 or end == num_images:
            logger.info(f"  Inference: {end}/{num_images} images")

    return all_detections


def prepare_eval_data(
    detections: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    input_size: int = 550,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """Convert model outputs and targets to numpy for evaluation.

    Model outputs boxes in normalized [0,1] coordinates; targets may be
    in absolute pixel coordinates. This function normalizes both to
    pixel coordinates at input_size scale for consistent evaluation.

    Args:
        detections: Model output dicts.
        targets: Ground-truth target dicts.
        input_size: Image size used for coordinate scaling.

    Returns:
        predictions_np: List of numpy prediction dicts.
        ground_truths_np: List of numpy ground-truth dicts.
    """
    predictions_np = []
    ground_truths_np = []

    for det, tgt in zip(detections, targets):
        # Predicted boxes
        pred_boxes = det['boxes'].numpy().astype(np.float64)
        pred_scores = det['scores'].numpy().astype(np.float64)
        pred_labels = det['labels'].numpy().astype(np.int64)

        # If boxes are normalized [0,1], scale to pixel coords
        if len(pred_boxes) > 0 and pred_boxes.max() <= 1.0:
            pred_boxes = pred_boxes * input_size

        predictions_np.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': pred_labels,
        })

        # Ground-truth boxes
        gt_boxes = tgt['boxes'].numpy().astype(np.float64)
        gt_labels = tgt['labels'].numpy().astype(np.int64)

        # If boxes are normalized, scale them too
        if len(gt_boxes) > 0 and gt_boxes.max() <= 1.0:
            gt_boxes = gt_boxes * input_size

        ground_truths_np.append({
            'boxes': gt_boxes,
            'labels': gt_labels,
        })

    return predictions_np, ground_truths_np


def plot_detection_samples(
    images: List[torch.Tensor],
    detections: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    save_path: str,
    num_samples: int = 8,
    input_size: int = 550,
    score_threshold: float = 0.3,
) -> None:
    """Plot a grid of sample detections alongside ground truth.

    Args:
        images: List of (3, H, W) image tensors.
        detections: Model detections.
        targets: Ground-truth targets.
        save_path: Path to save the figure.
        num_samples: Number of samples to plot.
        input_size: Image spatial size.
        score_threshold: Minimum score for displaying detections.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available; skipping detection grid.")
        return

    num_samples = min(num_samples, len(images))
    if num_samples == 0:
        return

    # Each sample gets 2 columns: GT and Pred
    ncols = 4
    nrows = max(1, (num_samples * 2 + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    # ImageNet denormalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(num_samples):
        # Denormalize image
        img = images[i].numpy().transpose(1, 2, 0)  # (H, W, 3)
        img = img * std + mean
        img = np.clip(img, 0, 1)

        # Position in grid
        pair_idx = i
        row_gt = (pair_idx * 2) // ncols
        col_gt = (pair_idx * 2) % ncols
        row_pred = ((pair_idx * 2) + 1) // ncols
        col_pred = ((pair_idx * 2) + 1) % ncols

        # Ground truth
        ax_gt = axes[row_gt, col_gt]
        ax_gt.imshow(img)
        gt_boxes = targets[i]['boxes'].numpy()
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            # Scale if normalized
            if max(x1, y1, x2, y2) <= 1.0:
                x1, y1, x2, y2 = x1 * input_size, y1 * input_size, x2 * input_size, y2 * input_size
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor='lime', facecolor='none',
            )
            ax_gt.add_patch(rect)
        ax_gt.set_title(f'GT #{i} ({len(gt_boxes)} obj)', fontsize=9)
        ax_gt.axis('off')

        # Predictions
        ax_pred = axes[row_pred, col_pred]
        ax_pred.imshow(img)
        det = detections[i]
        pred_boxes = det['boxes'].numpy()
        pred_scores = det['scores'].numpy()
        keep = pred_scores >= score_threshold
        for j in range(len(pred_boxes)):
            if not keep[j]:
                continue
            x1, y1, x2, y2 = pred_boxes[j]
            if max(x1, y1, x2, y2) <= 1.0:
                x1, y1, x2, y2 = x1 * input_size, y1 * input_size, x2 * input_size, y2 * input_size
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor='red', facecolor='none',
            )
            ax_pred.add_patch(rect)
        n_shown = int(keep.sum()) if len(keep) > 0 else 0
        ax_pred.set_title(f'Pred #{i} ({n_shown} det)', fontsize=9)
        ax_pred.axis('off')

    # Hide unused axes
    for r in range(axes.shape[0]):
        for c in range(axes.shape[1]):
            idx = r * ncols + c
            if idx >= num_samples * 2:
                axes[r, c].set_visible(False)

    plt.suptitle('Detection Samples: Ground Truth vs Predictions', fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved detection samples to {save_path}")


def plot_precision_recall(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, np.ndarray]],
    save_path: str,
    iou_thresholds: Optional[List[float]] = None,
) -> None:
    """Plot precision-recall curves at multiple IoU thresholds.

    Args:
        predictions: Numpy prediction dicts.
        ground_truths: Numpy ground-truth dicts.
        save_path: Path to save the figure.
        iou_thresholds: IoU thresholds to plot.
    """
    if not HAS_MATPLOTLIB:
        return

    from src.evaluation.metrics import (
        match_predictions_single_image,
        precision_recall_curve,
    )

    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for iou_thresh in iou_thresholds:
        all_scores = []
        all_tp_fp = []
        total_gt = 0

        for pred, gt in zip(predictions, ground_truths):
            total_gt += len(gt['boxes'])
            tp_fp = match_predictions_single_image(
                pred['boxes'], pred['scores'], gt['boxes'], iou_thresh
            )
            all_scores.append(pred['scores'])
            all_tp_fp.append(tp_fp)

        if total_gt == 0:
            continue

        scores = np.concatenate(all_scores)
        tp_fp = np.concatenate(all_tp_fp)
        idx = np.argsort(-scores)
        scores, tp_fp = scores[idx], tp_fp[idx]

        recalls, precisions = precision_recall_curve(scores, tp_fp, total_gt)
        ax.plot(recalls, precisions, label=f'IoU={iou_thresh:.2f}', linewidth=1.5)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved PR curves to {save_path}")


def plot_density_analysis(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, np.ndarray]],
    save_path: str,
) -> None:
    """Analyse detection performance as a function of object density.

    Buckets images by the number of ground-truth objects and computes
    AP@0.5 within each bucket.

    Args:
        predictions: Numpy prediction dicts.
        ground_truths: Numpy ground-truth dicts.
        save_path: Path to save the figure.
    """
    if not HAS_MATPLOTLIB:
        return

    # Determine density buckets
    gt_counts = [len(gt['boxes']) for gt in ground_truths]
    if len(gt_counts) == 0 or max(gt_counts) == 0:
        logger.info("No GT objects; skipping density analysis.")
        return

    # Create buckets
    bucket_edges = [0, 10, 30, 60, 100, 200, float('inf')]
    bucket_labels = ['0-10', '10-30', '30-60', '60-100', '100-200', '200+']

    bucket_aps = []
    bucket_counts = []

    for b_idx in range(len(bucket_labels)):
        lo = bucket_edges[b_idx]
        hi = bucket_edges[b_idx + 1]

        bucket_preds = []
        bucket_gts = []
        for pred, gt, count in zip(predictions, ground_truths, gt_counts):
            if lo <= count < hi:
                bucket_preds.append(pred)
                bucket_gts.append(gt)

        if len(bucket_preds) == 0:
            bucket_aps.append(0.0)
            bucket_counts.append(0)
            continue

        metrics = compute_detection_metrics(bucket_preds, bucket_gts, iou_thresholds=[0.5])
        bucket_aps.append(metrics.get('AP@0.50', 0.0))
        bucket_counts.append(len(bucket_preds))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # AP vs density
    x = np.arange(len(bucket_labels))
    bars = ax1.bar(x, bucket_aps, color='steelblue', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bucket_labels, fontsize=9)
    ax1.set_xlabel('Objects per Image', fontsize=11)
    ax1.set_ylabel('AP@0.5', fontsize=11)
    ax1.set_title('AP@0.5 by Object Density', fontsize=13)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, bucket_aps):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontsize=8)

    # Image count histogram
    ax2.bar(x, bucket_counts, color='coral', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bucket_labels, fontsize=9)
    ax2.set_xlabel('Objects per Image', fontsize=11)
    ax2.set_ylabel('Number of Images', fontsize=11)
    ax2.set_title('Image Distribution by Density', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Density Analysis', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved density analysis to {save_path}")


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    logger.info(f"Device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load model ----
    checkpoint_path = args.checkpoint
    if checkpoint_path is None and not args.untrained:
        checkpoint_path = find_checkpoint()
        if checkpoint_path:
            logger.info(f"Auto-detected checkpoint: {checkpoint_path}")

    model = load_model(checkpoint_path, device, untrained=args.untrained)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # ---- Load data ----
    use_real_data = False
    try:
        from src.data.dataset import SKU110KDataset
        val_dataset = SKU110KDataset(
            data_dir=args.data_dir,
            split='val',
            max_images=args.max_images,
            input_size=550,
        )
        if len(val_dataset) > 0:
            use_real_data = True
            logger.info(f"Loaded {len(val_dataset)} validation images.")
    except Exception as e:
        logger.warning(f"Could not load real dataset: {e}")

    if use_real_data:
        images = []
        targets = []
        for i in range(len(val_dataset)):
            img, tgt = val_dataset[i]
            images.append(img)
            targets.append(tgt)
    else:
        logger.info("Using synthetic data for evaluation (dataset not available).")
        n_synthetic = args.max_images if args.max_images else 20
        images, targets = create_synthetic_data(num_images=n_synthetic)

    # ---- Run inference ----
    logger.info("Running inference...")
    t0 = time.time()
    detections = run_inference(model, images, device, batch_size=args.batch_size)
    inference_time = time.time() - t0
    fps = len(images) / inference_time if inference_time > 0 else 0
    logger.info(f"Inference complete: {len(images)} images in {inference_time:.1f}s ({fps:.1f} FPS)")

    # ---- Prepare data for evaluation ----
    predictions_np, ground_truths_np = prepare_eval_data(detections, targets)

    # ---- Evaluate ----
    logger.info("Computing COCO-style metrics...")
    evaluator = COCOEvaluator()
    results = evaluator.evaluate(predictions_np, ground_truths_np)
    evaluator.print_results(results)

    # Add metadata
    results['_meta'] = {
        'num_images': len(images),
        'inference_time_s': round(inference_time, 2),
        'fps': round(fps, 1),
        'checkpoint': checkpoint_path or 'none',
        'untrained': args.untrained,
        'synthetic_data': not use_real_data,
        'device': str(device),
        'total_params': total_params,
    }

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    # Convert any numpy types for JSON serialization
    serializable = {}
    for k, v in results.items():
        if isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        elif isinstance(v, dict):
            serializable[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v.items()}
        else:
            serializable[k] = v

    with open(metrics_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"Saved metrics to {metrics_path}")

    # ---- Generate visualizations ----
    if HAS_MATPLOTLIB:
        # Detection samples grid
        samples_path = os.path.join(args.output_dir, 'detection_samples.png')
        plot_detection_samples(
            images, detections, targets, samples_path,
            num_samples=args.num_samples,
            score_threshold=args.score_threshold,
        )

        # PR curves (only useful with enough data)
        if len(images) >= 5:
            pr_path = os.path.join(args.output_dir, 'precision_recall.png')
            plot_precision_recall(predictions_np, ground_truths_np, pr_path)

            # Density analysis
            density_path = os.path.join(args.output_dir, 'density_analysis.png')
            plot_density_analysis(predictions_np, ground_truths_np, density_path)
    else:
        logger.warning("matplotlib not available; skipping visualizations.")

    # ---- Print summary table ----
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Images evaluated  : {len(images)}")
    print(f"  Data source       : {'real' if use_real_data else 'synthetic'}")
    print(f"  Device            : {device}")
    print(f"  Inference FPS     : {fps:.1f}")
    print(f"  AP@0.50           : {results.get('AP@0.50', 0):.4f}")
    print(f"  AP@0.75           : {results.get('AP@0.75', 0):.4f}")
    print(f"  AP@[.50:.95]      : {results.get('AP@[.50:.95]', 0):.4f}")
    print(f"  AR@100            : {results.get('AR@100', 0):.4f}")
    print(f"  AR@300            : {results.get('AR@300', 0):.4f}")
    print(f"  Output dir        : {args.output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()
