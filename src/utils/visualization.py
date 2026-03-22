"""Visualization utilities for YOLACT detection results.

Provides functions for drawing detections on images, plotting training
curves, and creating side-by-side comparison grids for evaluation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Color palette for visualization (RGB, 0-1 range)
COLORS = [
    (0.122, 0.467, 0.706),   # Blue
    (1.000, 0.498, 0.055),   # Orange
    (0.173, 0.627, 0.173),   # Green
    (0.839, 0.153, 0.157),   # Red
    (0.580, 0.404, 0.741),   # Purple
    (0.549, 0.337, 0.294),   # Brown
    (0.890, 0.467, 0.761),   # Pink
    (0.498, 0.498, 0.498),   # Gray
    (0.737, 0.741, 0.133),   # Yellow-green
    (0.090, 0.745, 0.812),   # Cyan
]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    masks: Optional[np.ndarray] = None,
    threshold: float = 0.3,
    class_names: Optional[List[str]] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """Draw detection results (boxes, labels, scores, masks) on an image.

    Args:
        image: Input image as numpy array (H, W, 3) with values in [0, 255] uint8
            or [0, 1] float.
        boxes: Bounding boxes (N, 4) as [x1, y1, x2, y2] in pixel coordinates.
        scores: Confidence scores (N,).
        labels: Class labels (N,) as integers.
        masks: Optional instance masks (N, H, W) with values in [0, 1].
        threshold: Minimum score threshold for drawing.
        class_names: Optional list of class name strings indexed by label.
        alpha: Transparency for mask overlay.

    Returns:
        Annotated image as numpy array (H, W, 3) with uint8 values.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

    # Ensure image is float for overlay operations
    img = image.copy().astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    h, w = img.shape[:2]

    # Filter by threshold
    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if masks is not None:
        masks = masks[keep]

    for i in range(len(boxes)):
        color = COLORS[int(labels[i]) % len(COLORS)]
        x1, y1, x2, y2 = boxes[i].astype(int)

        # Draw mask overlay
        if masks is not None and i < len(masks):
            mask = masks[i]
            # Resize mask to image size if needed
            if mask.shape != (h, w):
                mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
                mask = torch.nn.functional.interpolate(
                    mask_tensor, size=(h, w), mode='bilinear', align_corners=False
                ).squeeze().numpy()

            # Apply mask overlay
            mask_bool = mask > 0.5
            for c in range(3):
                img[:, :, c] = np.where(
                    mask_bool,
                    img[:, :, c] * (1 - alpha) + color[c] * alpha,
                    img[:, :, c],
                )

        # Draw bounding box
        thickness = max(1, int(min(h, w) / 300))
        # Top
        img[max(y1, 0):min(y1 + thickness, h), max(x1, 0):min(x2, w)] = color
        # Bottom
        img[max(y2 - thickness, 0):min(y2, h), max(x1, 0):min(x2, w)] = color
        # Left
        img[max(y1, 0):min(y2, h), max(x1, 0):min(x1 + thickness, w)] = color
        # Right
        img[max(y1, 0):min(y2, h), max(x2 - thickness, 0):min(x2, w)] = color

        # Build label text
        if class_names is not None and int(labels[i]) < len(class_names):
            label_text = f"{class_names[int(labels[i])]}: {scores[i]:.2f}"
        else:
            label_text = f"Class {int(labels[i])}: {scores[i]:.2f}"

        # Draw label background and text using matplotlib later in figure mode
        # For now, we store the text position for figure-based rendering

    # Convert back to uint8
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def draw_detections_figure(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    masks: Optional[np.ndarray] = None,
    threshold: float = 0.3,
    class_names: Optional[List[str]] = None,
    alpha: float = 0.4,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Draw detections using matplotlib for better text rendering.

    Args:
        image: Input image (H, W, 3), uint8 or float.
        boxes: Bounding boxes (N, 4) as [x1, y1, x2, y2] in pixel coords.
        scores: Confidence scores (N,).
        labels: Class labels (N,).
        masks: Optional instance masks (N, H, W) in [0, 1].
        threshold: Minimum score to draw.
        class_names: Optional class name list.
        alpha: Mask overlay transparency.
        figsize: Figure size in inches.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure, or None if save_path is provided.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization.")

    # First apply mask overlays to the image
    annotated = draw_detections(image, boxes, scores, labels, masks, threshold, class_names, alpha)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(annotated)

    # Filter by threshold
    keep = scores >= threshold
    boxes_f = boxes[keep]
    scores_f = scores[keep]
    labels_f = labels[keep]

    for i in range(len(boxes_f)):
        color = COLORS[int(labels_f[i]) % len(COLORS)]
        x1, y1, x2, y2 = boxes_f[i]

        # Draw box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none',
        )
        ax.add_patch(rect)

        # Draw label
        if class_names is not None and int(labels_f[i]) < len(class_names):
            label_text = f"{class_names[int(labels_f[i])]}: {scores_f[i]:.2f}"
        else:
            label_text = f"Class {int(labels_f[i])}: {scores_f[i]:.2f}"

        ax.text(
            x1, y1 - 5, label_text,
            fontsize=10, color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8),
        )

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def plot_training_curves(
    log_file: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> Optional[plt.Figure]:
    """Plot training loss and mAP curves from a log file.

    Expects a CSV or JSON-lines log file with columns/keys:
        epoch, train_loss, val_loss, mAP (optional), lr (optional)

    Args:
        log_file: Path to the training log file (CSV or JSONL).
        save_path: If provided, save figure to this path.
        figsize: Figure size in inches.

    Returns:
        Matplotlib figure, or None if save_path is provided.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization.")

    log_path = Path(log_file)

    # Parse log file
    if log_path.suffix == '.csv':
        import csv
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    elif log_path.suffix in ('.jsonl', '.json'):
        import json
        data = []
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported log format: {log_path.suffix}. Use .csv or .jsonl")

    if not data:
        raise ValueError(f"Empty log file: {log_file}")

    # Extract columns
    epochs = [float(d.get('epoch', i)) for i, d in enumerate(data)]
    train_loss = [float(d['train_loss']) for d in data if 'train_loss' in d]
    val_loss = [float(d['val_loss']) for d in data if 'val_loss' in d]
    has_map = 'mAP' in data[0]
    has_lr = 'lr' in data[0]

    num_plots = 2
    if has_map:
        num_plots += 1
    if has_lr:
        num_plots += 1

    fig, axes = plt.subplots(
        (num_plots + 1) // 2, 2, figsize=figsize
    )
    axes = axes.flatten()

    plot_idx = 0

    # Training loss
    if train_loss:
        ax = axes[plot_idx]
        ax.plot(epochs[:len(train_loss)], train_loss, 'b-', label='Train Loss', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Validation loss
    if val_loss:
        ax = axes[plot_idx]
        ax.plot(epochs[:len(val_loss)], val_loss, 'r-', label='Val Loss', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Combined loss
    if train_loss and val_loss:
        ax = axes[plot_idx]
        ax.plot(epochs[:len(train_loss)], train_loss, 'b-', label='Train', linewidth=1.5)
        ax.plot(epochs[:len(val_loss)], val_loss, 'r-', label='Val', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Train vs Val Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # mAP
    if has_map:
        mAP_vals = [float(d['mAP']) for d in data]
        ax = axes[plot_idx]
        ax.plot(epochs, mAP_vals, 'g-o', label='mAP', linewidth=1.5, markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('Mean Average Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Training Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def create_comparison_grid(
    images: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    pred_boxes: List[np.ndarray],
    pred_scores: Optional[List[np.ndarray]] = None,
    pred_labels: Optional[List[np.ndarray]] = None,
    gt_labels: Optional[List[np.ndarray]] = None,
    pred_masks: Optional[List[np.ndarray]] = None,
    class_names: Optional[List[str]] = None,
    num_cols: int = 4,
    figsize_per_image: Tuple[float, float] = (4, 4),
    threshold: float = 0.3,
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Create a side-by-side comparison grid of ground truth and predictions.

    Displays pairs of images: ground truth annotations on the left,
    predicted detections on the right.

    Args:
        images: List of input images (H, W, 3).
        gt_boxes: List of ground truth box arrays (M, 4).
        pred_boxes: List of predicted box arrays (N, 4).
        pred_scores: Optional list of prediction score arrays (N,).
        pred_labels: Optional list of prediction label arrays (N,).
        gt_labels: Optional list of ground truth label arrays (M,).
        pred_masks: Optional list of predicted mask arrays (N, H, W).
        class_names: Optional class name list.
        num_cols: Number of image pairs per row.
        figsize_per_image: (width, height) per image in inches.
        threshold: Score threshold for drawing predictions.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib figure, or None if save_path is provided.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization.")

    n = len(images)
    # Each sample takes 2 columns (GT + Pred)
    cols = min(num_cols * 2, n * 2)
    rows = int(np.ceil(n * 2 / cols))

    fig_w = figsize_per_image[0] * cols
    fig_h = figsize_per_image[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(n):
        row = (i * 2) // cols
        col_gt = (i * 2) % cols
        col_pred = col_gt + 1

        img = images[i].copy()
        if img.max() > 1.0:
            img_display = img.astype(np.float32) / 255.0
        else:
            img_display = img.astype(np.float32)

        # Ground truth
        ax_gt = axes[row, col_gt]
        ax_gt.imshow(img_display)
        if gt_boxes[i] is not None and len(gt_boxes[i]) > 0:
            for j, box in enumerate(gt_boxes[i]):
                x1, y1, x2, y2 = box
                label = int(gt_labels[i][j]) if gt_labels is not None else 0
                color = COLORS[label % len(COLORS)]
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor='none',
                )
                ax_gt.add_patch(rect)
        ax_gt.set_title(f'GT #{i}', fontsize=10)
        ax_gt.axis('off')

        # Predictions
        ax_pred = axes[row, col_pred]
        p_scores = pred_scores[i] if pred_scores is not None else np.ones(len(pred_boxes[i]))
        p_labels = pred_labels[i] if pred_labels is not None else np.zeros(len(pred_boxes[i]), dtype=int)
        p_masks = pred_masks[i] if pred_masks is not None else None

        annotated = draw_detections(
            img, pred_boxes[i], p_scores, p_labels,
            masks=p_masks, threshold=threshold, class_names=class_names,
        )
        ax_pred.imshow(annotated.astype(np.float32) / 255.0 if annotated.max() > 1 else annotated)
        ax_pred.set_title(f'Pred #{i}', fontsize=10)
        ax_pred.axis('off')

    # Hide unused axes
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            if row * cols + col >= n * 2:
                axes[row, col].set_visible(False)

    plt.suptitle('Ground Truth vs Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig
