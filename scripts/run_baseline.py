#!/usr/bin/env python
"""
Run the HOG + SVM baseline pipeline.

Usage:
    python scripts/run_baseline.py

The script will:
1. Try to load real SKU-110K data from data/SKU-110K/.
2. Fall back to procedurally generated synthetic shelf images.
3. Train a HOG + Linear SVM detector.
4. Evaluate on a held-out validation split.
5. Save metrics, detection visualisations, and HOG feature plots under
   results/baseline/.

Expected outcome: mAP@0.5 ~ 5-12 %, demonstrating that classic CV methods
fail on dense retail scenes.

Dependencies: numpy, opencv-python, scikit-learn, scikit-image, matplotlib
"""

import json
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root so the script works regardless of cwd
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.baseline.hog_svm import HOGSVMBaseline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "SKU-110K")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "baseline")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===================================================================
# 1.  DATA LOADING / SYNTHETIC FALLBACK
# ===================================================================

def try_load_sku110k(max_train=300, max_val=100):
    """Attempt to load real SKU-110K images and annotations.

    Expected layout:
        data/SKU-110K/images/train/*.jpg
        data/SKU-110K/annotations/annotations_train.csv

    CSV columns (per the original dataset):
        image_name, x1, y1, x2, y2, class, image_width, image_height

    Returns
    -------
    (train_images, train_annots, val_images, val_annots) or None
    """
    img_dir = os.path.join(DATA_DIR, "images", "train")
    ann_file = os.path.join(DATA_DIR, "annotations", "annotations_train.csv")

    if not os.path.isdir(img_dir) or not os.path.isfile(ann_file):
        return None

    print("[INFO] Found SKU-110K data. Loading ...")

    # Parse annotations
    from collections import defaultdict
    ann_map = defaultdict(list)
    with open(ann_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            name = parts[0]
            try:
                x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                ann_map[name].append([x1, y1, x2, y2])
            except ValueError:
                continue

    all_names = sorted(ann_map.keys())
    if len(all_names) < max_train + max_val:
        return None

    rng = np.random.RandomState(42)
    rng.shuffle(all_names)
    train_names = all_names[:max_train]
    val_names = all_names[max_train : max_train + max_val]

    def _load(names):
        imgs, anns = [], []
        for n in names:
            path = os.path.join(img_dir, n)
            img = cv2.imread(path)
            if img is None:
                continue
            # Resize large images to manageable size
            h, w = img.shape[:2]
            scale = 1.0
            if max(h, w) > 600:
                scale = 600.0 / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            boxes = np.array(ann_map[n]) * scale
            imgs.append(img)
            anns.append(boxes)
        return imgs, anns

    ti, ta = _load(train_names)
    vi, va = _load(val_names)
    if len(ti) < 50 or len(vi) < 10:
        return None

    print(f"  Loaded {len(ti)} train / {len(vi)} val images from SKU-110K.")
    return ti, ta, vi, va


# -------------------------------------------------------------------
# Synthetic data generator
# -------------------------------------------------------------------

def generate_synthetic_dataset(
    num_train=300,
    num_val=80,
    img_size=300,
    min_products=10,
    max_products=30,
    seed=42,
):
    """Generate synthetic shelf images with coloured rectangles as products.

    Each image has a shelf-like striped background with `min_products` to
    `max_products` axis-aligned rectangles placed in a grid-ish layout to
    mimic dense retail shelves.

    Returns
    -------
    train_images, train_annots, val_images, val_annots
    """
    rng = np.random.RandomState(seed)

    def _make_image():
        # Shelf background -- horizontal bands of slightly different greys
        img = np.full((img_size, img_size, 3), 200, dtype=np.uint8)
        num_shelves = rng.randint(3, 6)
        shelf_h = img_size // num_shelves
        for s in range(num_shelves):
            y0 = s * shelf_h
            y1 = min((s + 1) * shelf_h, img_size)
            shade = rng.randint(170, 230)
            img[y0:y1, :] = shade
            # Dark shelf edge
            cv2.line(img, (0, y1 - 1), (img_size, y1 - 1), (100, 90, 80), 2)

        # Add slight noise to background
        noise = rng.randint(-10, 11, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Place products
        num_products = rng.randint(min_products, max_products + 1)
        boxes = []

        # Try grid-like placement with jitter
        cols = rng.randint(4, 8)
        rows = num_shelves
        cell_w = img_size // cols
        cell_h = shelf_h

        placed = 0
        for r in range(rows):
            for c in range(cols):
                if placed >= num_products:
                    break
                # Product dimensions with randomness
                pw = rng.randint(int(cell_w * 0.4), int(cell_w * 0.9) + 1)
                ph = rng.randint(int(cell_h * 0.5), int(cell_h * 0.9) + 1)

                # Position with jitter
                cx = c * cell_w + cell_w // 2 + rng.randint(-cell_w // 6, cell_w // 6 + 1)
                cy = r * cell_h + cell_h // 2 + rng.randint(-cell_h // 6, cell_h // 6 + 1)

                x1 = max(0, cx - pw // 2)
                y1 = max(0, cy - ph // 2)
                x2 = min(img_size, x1 + pw)
                y2 = min(img_size, y1 + ph)

                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue

                # Random product colour
                colour = tuple(int(v) for v in rng.randint(40, 240, 3))
                cv2.rectangle(img, (x1, y1), (x2, y2), colour, -1)

                # Border
                border_col = tuple(int(v) for v in (np.array(colour) * 0.6).astype(int))
                cv2.rectangle(img, (x1, y1), (x2, y2), border_col, 1)

                # Optional: small label rectangle on the product
                if rng.rand() > 0.4:
                    lw = max(4, (x2 - x1) // 2)
                    lh = max(3, (y2 - y1) // 4)
                    lx = x1 + (x2 - x1 - lw) // 2
                    ly = y1 + (y2 - y1) // 2 - lh // 2
                    label_col = tuple(int(v) for v in rng.randint(200, 256, 3))
                    cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), label_col, -1)

                boxes.append([x1, y1, x2, y2])
                placed += 1

        return img, np.array(boxes, dtype=np.float32).reshape(-1, 4)

    print("[INFO] Generating synthetic shelf images ...")
    train_images, train_annots = [], []
    for _ in range(num_train):
        img, boxes = _make_image()
        train_images.append(img)
        train_annots.append(boxes)

    val_images, val_annots = [], []
    for _ in range(num_val):
        img, boxes = _make_image()
        val_images.append(img)
        val_annots.append(boxes)

    avg_products = np.mean([len(a) for a in train_annots + val_annots])
    print(f"  Generated {num_train} train / {num_val} val images "
          f"({avg_products:.1f} products/image avg).")
    return train_images, train_annots, val_images, val_annots


# ===================================================================
# 2.  VISUALISATION HELPERS
# ===================================================================

def draw_detections(image, boxes, scores, gt_boxes=None, max_det=80):
    """Draw detection boxes (green) and optional GT boxes (blue) on image."""
    vis = image.copy()

    # Draw GT in blue
    if gt_boxes is not None:
        for b in gt_boxes:
            x1, y1, x2, y2 = b.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 180, 0), 1)

    # Draw detections in green
    order = np.argsort(-scores)[:max_det]
    for i in order:
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{scores[i]:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    return vis


def save_detection_grid(images, annotations, model, path, n=4):
    """Run detection on *n* images and save a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(min(n, len(images))):
        boxes, scores = model.detect(images[i], score_threshold=0.3)
        vis = draw_detections(images[i], boxes, scores, gt_boxes=annotations[i])
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        axes[i].imshow(vis_rgb)
        axes[i].set_title(f"Image {i}: {len(boxes)} dets / {len(annotations[i])} GT")
        axes[i].axis("off")

    plt.suptitle("HOG+SVM Baseline Detections (green=det, blue=GT)", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved detection grid -> {path}")


def save_hog_visualization(images, model, path, n=4):
    """Visualise HOG features on sample crops."""
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    rng = np.random.RandomState(0)
    for i in range(n):
        img = images[i % len(images)]
        h, w = img.shape[:2]
        cx = rng.randint(0, max(1, w - 64))
        cy = rng.randint(0, max(1, h - 64))
        patch = img[cy:cy + 64, cx:cx + 64]
        if patch.shape[0] < 10 or patch.shape[1] < 10:
            patch = cv2.resize(img, (64, 64))

        _, hog_img = model.extract_hog_features(patch, visualize=True)

        patch_rgb = cv2.cvtColor(
            cv2.resize(patch, model.window_size), cv2.COLOR_BGR2RGB
        )
        axes[0, i].imshow(patch_rgb)
        axes[0, i].set_title(f"Patch {i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(hog_img, cmap="gray")
        axes[1, i].set_title("HOG features")
        axes[1, i].axis("off")

    plt.suptitle("HOG Feature Visualisation", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved HOG visualisation -> {path}")


# ===================================================================
# 3.  MAIN
# ===================================================================

def main():
    print("=" * 65)
    print("  HOG + SVM Baseline for Dense Object Detection")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    result = try_load_sku110k(max_train=300, max_val=80)
    if result is not None:
        train_images, train_annots, val_images, val_annots = result
        data_source = "SKU-110K"
    else:
        print("[INFO] SKU-110K data not found -- using synthetic fallback.")
        train_images, train_annots, val_images, val_annots = generate_synthetic_dataset(
            num_train=300, num_val=80, img_size=300,
            min_products=10, max_products=30,
        )
        data_source = "synthetic"

    print(f"\n  Data source : {data_source}")
    print(f"  Train images: {len(train_images)}")
    print(f"  Val images  : {len(val_images)}")
    avg_gt_train = np.mean([len(a) for a in train_annots])
    avg_gt_val = np.mean([len(a) for a in val_annots])
    print(f"  Avg GT/image: {avg_gt_train:.1f} (train), {avg_gt_val:.1f} (val)")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = HOGSVMBaseline(
        window_size=(64, 64),
        cell_size=(8, 8),
        block_size=(16, 16),
        nbins=9,
    )

    # ------------------------------------------------------------------
    # Prepare training data
    # ------------------------------------------------------------------
    print("\n--- Preparing training data ---")
    t0 = time.time()
    X_train, y_train = model.prepare_training_data(
        train_images, train_annots, num_pos=5000, num_neg=10000
    )
    prep_time = time.time() - t0
    print(f"  Feature matrix shape: {X_train.shape}")
    print(f"  Preparation time    : {prep_time:.1f}s")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\n--- Training Linear SVM ---")
    t0 = time.time()
    train_acc = model.train(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # ------------------------------------------------------------------
    # HOG feature visualisation (save before long eval)
    # ------------------------------------------------------------------
    print("\n--- Generating HOG visualisation ---")
    hog_vis_path = os.path.join(RESULTS_DIR, "hog_features_visualization.png")
    save_hog_visualization(val_images, model, hog_vis_path)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    num_eval = min(50, len(val_images))
    print(f"\n--- Evaluating on {num_eval} validation images ---")
    t0 = time.time()
    metrics = model.evaluate(val_images, val_annots, num_images=num_eval, score_threshold=0.3)
    eval_time = time.time() - t0
    print(f"  Evaluation time: {eval_time:.1f}s")

    # Add extra info to metrics
    metrics["data_source"] = data_source
    metrics["num_train_images"] = len(train_images)
    metrics["num_val_images"] = len(val_images)
    metrics["train_accuracy"] = round(train_acc * 100, 2)
    metrics["feature_dim"] = int(X_train.shape[1])
    metrics["training_time_sec"] = round(train_time, 2)
    metrics["data_prep_time_sec"] = round(prep_time, 2)
    metrics["evaluation_time_sec"] = round(eval_time, 2)
    metrics["window_size"] = list(model.window_size)
    metrics["cell_size"] = list(model.cell_size)
    metrics["block_size"] = list(model.block_size)
    metrics["nbins"] = model.nbins

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    metrics_path = os.path.join(RESULTS_DIR, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved metrics -> {metrics_path}")

    # ------------------------------------------------------------------
    # Save detection visualisation grid
    # ------------------------------------------------------------------
    det_vis_path = os.path.join(RESULTS_DIR, "baseline_detections.png")
    print("\n--- Saving detection visualisation ---")
    save_detection_grid(val_images, val_annots, model, det_vis_path, n=4)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"  HOG+SVM Baseline: mAP@0.5 = {metrics['mAP@0.5']:.2f}%")
    print(f"  Precision        : {metrics['precision']:.2f}%")
    print(f"  Recall           : {metrics['recall']:.2f}%")
    print(f"  Avg time / image : {metrics['avg_time_per_image_sec']:.3f}s")
    print(f"  Total GT boxes   : {metrics['total_gt_boxes']}")
    print(f"  Total detections : {metrics['total_detections']}")
    print("=" * 65)
    print("\n  Conclusion: Classic HOG+SVM achieves very low mAP on dense")
    print("  retail scenes, confirming the need for deep learning detectors.")
    print()

    return metrics


if __name__ == "__main__":
    main()
