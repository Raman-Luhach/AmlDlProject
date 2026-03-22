#!/usr/bin/env python3
"""Live demo: run YOLACT inference on a single image and visualize detections.

Usage:
    python scripts/demo.py --image path/to/image.jpg
    python scripts/demo.py --image path/to/image.jpg --checkpoint results/training/checkpoints/best_model.pth
    python scripts/demo.py --image path/to/image.jpg --output results/demo_output.png
    python scripts/demo.py --test-images 5   # run on 5 random test images

The script loads the trained YOLACT model, runs inference, draws bounding
boxes with confidence scores, and saves/displays the result.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.yolact import YOLACT, DEFAULT_CONFIG
from src.utils.helpers import get_device

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Colors for drawing (BGR)
COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (128, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (128, 0, 255),
]

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_path: str, input_size: int = 550):
    """Load and preprocess an image for YOLACT inference.

    Returns:
        tensor: (1, 3, H, W) normalized tensor
        orig_image: original BGR image for visualization
        scale: (scale_x, scale_y) for mapping detections back
    """
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    orig_h, orig_w = orig_image.shape[:2]
    logger.info(f"Loaded image: {image_path} ({orig_w}x{orig_h})")

    # Resize to model input size
    resized = cv2.resize(orig_image, (input_size, input_size))

    # Convert BGR -> RGB, normalize
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD

    # HWC -> CHW -> BCHW
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

    scale_x = orig_w / input_size
    scale_y = orig_h / input_size

    return tensor, orig_image, (scale_x, scale_y)


def draw_detections(image, detections, scale, score_threshold=0.01):
    """Draw bounding boxes and scores on the image.

    Args:
        image: BGR image (numpy array)
        detections: dict with 'boxes', 'scores', 'labels' tensors
        scale: (scale_x, scale_y) to map from model coords to original
        score_threshold: minimum score to draw

    Returns:
        annotated image, number of detections drawn
    """
    vis = image.copy()
    boxes = detections.get('boxes', torch.tensor([]))
    scores = detections.get('scores', torch.tensor([]))

    if len(boxes) == 0:
        logger.info("No detections found.")
        # Add text saying no detections
        cv2.putText(vis, "No detections (model needs more training)",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return vis, 0

    # Move to CPU numpy
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()

    # Filter by threshold
    mask = scores_np >= score_threshold
    boxes_np = boxes_np[mask]
    scores_np = scores_np[mask]

    # Scale boxes to original image size
    scale_x, scale_y = scale
    boxes_np[:, 0] *= scale_x
    boxes_np[:, 2] *= scale_x
    boxes_np[:, 1] *= scale_y
    boxes_np[:, 3] *= scale_y

    num_drawn = len(boxes_np)
    logger.info(f"Drawing {num_drawn} detections (threshold={score_threshold:.2f})")

    for i, (box, score) in enumerate(zip(boxes_np, scores_np)):
        x1, y1, x2, y2 = box.astype(int)
        color = COLORS[i % len(COLORS)]

        # Draw box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Draw score label
        label = f"{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add summary text
    cv2.putText(vis, f"Detections: {num_drawn}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return vis, num_drawn


def load_model(checkpoint_path, device):
    """Load YOLACT model from checkpoint."""
    config = {**DEFAULT_CONFIG, 'pretrained_backbone': False}
    model = YOLACT(config=config)

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Checkpoint loaded.")
    else:
        logger.warning("No checkpoint found — using untrained model.")

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    return model


def find_test_images(data_dir="data", num_images=5):
    """Find random test images from the dataset."""
    image_dir = Path(data_dir) / "SKU110K_fixed" / "images"
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return []

    # Look for test images
    test_images = sorted([f for f in image_dir.glob("test_*.jpg")])
    if not test_images:
        # Fall back to any images
        test_images = sorted(image_dir.glob("*.jpg"))

    # Sample randomly
    rng = np.random.RandomState(42)
    if len(test_images) > num_images:
        indices = rng.choice(len(test_images), num_images, replace=False)
        test_images = [test_images[i] for i in sorted(indices)]

    return [str(p) for p in test_images[:num_images]]


def create_comparison_grid(results, output_path):
    """Create a grid of detection results for multiple images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (img, num_det, img_path) in enumerate(results):
        ax = axes[i]
        # Convert BGR -> RGB for matplotlib
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{Path(img_path).name}\n{num_det} detections", fontsize=10)
        ax.axis('off')

    # Hide empty axes
    for i in range(len(results), len(axes)):
        axes[i].axis('off')

    plt.suptitle("YOLACT Detection Results (MobileNetV3-Large + Soft-NMS)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved grid to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLACT Live Demo')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image')
    parser.add_argument('--test-images', type=int, default=None,
                        help='Run on N random test images from the dataset')
    parser.add_argument('--checkpoint', type=str,
                        default='results/training/checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path (default: results/demo_output.png)')
    parser.add_argument('--score-threshold', type=float, default=0.01,
                        help='Minimum detection score to display')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory (for --test-images mode)')
    args = parser.parse_args()

    if args.image is None and args.test_images is None:
        # Default: run on 5 test images
        args.test_images = 5

    device = get_device()
    logger.info(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    if args.image:
        # Single image mode
        output_path = args.output or 'results/demo_output.png'
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        tensor, orig_image, scale = preprocess_image(args.image)
        tensor = tensor.to(device)

        logger.info("Running inference...")
        with torch.no_grad():
            detections = model(tensor)

        det = detections[0] if isinstance(detections, list) else detections
        vis, num_det = draw_detections(orig_image, det, scale, args.score_threshold)

        cv2.imwrite(output_path, vis)
        logger.info(f"Saved output to: {output_path}")
        logger.info(f"Total detections: {num_det}")

    elif args.test_images:
        # Multi-image mode
        output_path = args.output or 'results/demo_grid.png'
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        image_paths = find_test_images(args.data_dir, args.test_images)
        if not image_paths:
            logger.error("No test images found. Provide --image path instead.")
            sys.exit(1)

        logger.info(f"Running inference on {len(image_paths)} images...")
        results = []

        for img_path in image_paths:
            tensor, orig_image, scale = preprocess_image(img_path)
            tensor = tensor.to(device)

            with torch.no_grad():
                detections = model(tensor)

            det = detections[0] if isinstance(detections, list) else detections
            vis, num_det = draw_detections(orig_image, det, scale, args.score_threshold)

            # Save individual result
            individual_path = str(Path(output_path).parent / f"demo_{Path(img_path).stem}.png")
            cv2.imwrite(individual_path, vis)

            results.append((vis, num_det, img_path))

        # Create grid
        create_comparison_grid(results, output_path)

        total_det = sum(r[1] for r in results)
        logger.info(f"\nDemo complete: {len(results)} images, {total_det} total detections")
        logger.info(f"Grid saved to: {output_path}")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
