#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for SKU-110K Dataset.

Generates 8 publication-quality visualizations and computes key statistics.
Falls back to realistic synthetic data if the real dataset is unavailable.

Usage:
    python scripts/run_eda.py

Outputs:
    results/eda/*.png          -- 8 visualization plots
    configs/custom_anchors.yaml -- K-means anchor analysis results
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.spatial.distance import cdist
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "SKU110K_fixed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
IMAGES_DIR = DATA_DIR / "images"
OUTPUT_DIR = PROJECT_ROOT / "results" / "eda"
CONFIGS_DIR = PROJECT_ROOT / "configs"

SPLITS = ["train", "val", "test"]
EXPECTED_COUNTS = {"train": 8233, "val": 588, "test": 2941}

# Plot style
DPI = 200
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_WIDE = (12, 6)
FIGSIZE_GRID = (16, 12)

PALETTE = sns.color_palette("Set2", 8)
ACCENT = "#2c7fb8"
ACCENT2 = "#d95f02"

np.random.seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_plot_style():
    """Set a consistent professional plot style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("ggplot")

    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def save_fig(fig, name):
    """Save figure and close."""
    path = OUTPUT_DIR / name
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Data loading / synthetic generation
# ---------------------------------------------------------------------------
def try_load_real_data():
    """Attempt to load real SKU-110K annotation CSVs.

    Expected columns: image_name, x1, y1, x2, y2, class, image_width, image_height
    """
    dfs = {}
    for split in SPLITS:
        csv_path = ANNOTATIONS_DIR / f"annotations_{split}.csv"
        if not csv_path.exists():
            return None
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["image_name", "x1", "y1", "x2", "y2", "class",
                   "image_width", "image_height"],
        )
        df["split"] = split
        dfs[split] = df

    combined = pd.concat(dfs.values(), ignore_index=True)
    print(f"Loaded REAL dataset: {len(combined):,} annotations across "
          f"{combined['image_name'].nunique():,} images")
    return combined


def generate_synthetic_data():
    """Generate synthetic data mimicking SKU-110K statistics."""
    print("Real dataset not found. Generating synthetic data ...")
    records = []

    for split, n_images in EXPECTED_COUNTS.items():
        for img_idx in range(n_images):
            image_name = f"synthetic_{split}_{img_idx:06d}.jpg"
            image_width = np.random.choice([1024, 1280, 1920])
            image_height = np.random.choice([768, 1024, 1280])

            n_objects = int(np.clip(
                np.random.normal(147, 40), 5, 400
            ))

            # Generate box dimensions using lognormal
            widths = np.random.lognormal(
                mean=np.log(45), sigma=0.4, size=n_objects
            ).clip(8, image_width * 0.3)
            heights = np.random.lognormal(
                mean=np.log(60), sigma=0.4, size=n_objects
            ).clip(10, image_height * 0.4)

            # Place boxes with some structure (grid-like with noise)
            cols = int(np.sqrt(n_objects * image_width / image_height)) + 1
            rows = n_objects // cols + 1
            cell_w = image_width / cols
            cell_h = image_height / rows

            for k in range(n_objects):
                r, c = divmod(k, cols)
                cx = c * cell_w + cell_w / 2 + np.random.normal(0, cell_w * 0.15)
                cy = r * cell_h + cell_h / 2 + np.random.normal(0, cell_h * 0.15)
                w, h = widths[k], heights[k]
                x1 = np.clip(cx - w / 2, 0, image_width - w)
                y1 = np.clip(cy - h / 2, 0, image_height - h)
                x2 = x1 + w
                y2 = y1 + h

                records.append({
                    "image_name": image_name,
                    "x1": round(float(x1), 1),
                    "y1": round(float(y1), 1),
                    "x2": round(float(x2), 1),
                    "y2": round(float(y2), 1),
                    "class": 1,
                    "image_width": int(image_width),
                    "image_height": int(image_height),
                    "split": split,
                })

    df = pd.DataFrame(records)
    print(f"Generated SYNTHETIC dataset: {len(df):,} annotations across "
          f"{df['image_name'].nunique():,} images")
    return df


def load_data():
    """Load real data or fall back to synthetic."""
    df = try_load_real_data()
    if df is None:
        df = generate_synthetic_data()
    # Derived columns
    df["box_w"] = df["x2"] - df["x1"]
    df["box_h"] = df["y2"] - df["y1"]
    df["norm_w"] = df["box_w"] / df["image_width"]
    df["norm_h"] = df["box_h"] / df["image_height"]
    df["area"] = df["box_w"] * df["box_h"]
    df["norm_area"] = df["norm_w"] * df["norm_h"]
    df["aspect_ratio"] = df["box_w"] / df["box_h"].clip(lower=1e-6)
    return df


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------
def compute_iou_matrix(boxes):
    """Compute pairwise IoU for an Nx4 array of [x1, y1, x2, y2]."""
    x1 = np.maximum(boxes[:, 0][:, None], boxes[:, 0][None, :])
    y1 = np.maximum(boxes[:, 1][:, None], boxes[:, 1][None, :])
    x2 = np.minimum(boxes[:, 2][:, None], boxes[:, 2][None, :])
    y2 = np.minimum(boxes[:, 3][:, None], boxes[:, 3][None, :])

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_b = area_a.copy()
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / np.maximum(union, 1e-8)
    return iou


# ---------------------------------------------------------------------------
# K-means anchor analysis
# ---------------------------------------------------------------------------
def iou_distance(boxes, clusters):
    """1 - IoU between each box and each cluster center (treating as wh)."""
    n = boxes.shape[0]
    k = clusters.shape[0]

    box_area = boxes[:, 0] * boxes[:, 1]  # w*h
    cluster_area = clusters[:, 0] * clusters[:, 1]

    inter_w = np.minimum(boxes[:, 0][:, None], clusters[:, 0][None, :])
    inter_h = np.minimum(boxes[:, 1][:, None], clusters[:, 1][None, :])
    inter = inter_w * inter_h

    union = box_area[:, None] + cluster_area[None, :] - inter
    iou = inter / np.maximum(union, 1e-8)
    return 1.0 - iou


def kmeans_anchors(boxes_wh, k, max_iter=300):
    """K-means clustering using IoU distance on (w, h) pairs."""
    n = boxes_wh.shape[0]
    indices = np.random.choice(n, k, replace=False)
    clusters = boxes_wh[indices].copy()

    for _ in range(max_iter):
        dist = iou_distance(boxes_wh, clusters)
        assignments = np.argmin(dist, axis=1)

        new_clusters = np.zeros_like(clusters)
        for j in range(k):
            mask = assignments == j
            if mask.sum() > 0:
                new_clusters[j] = boxes_wh[mask].mean(axis=0)
            else:
                new_clusters[j] = boxes_wh[np.random.randint(n)]

        if np.allclose(clusters, new_clusters, atol=1e-6):
            break
        clusters = new_clusters

    dist = iou_distance(boxes_wh, clusters)
    mean_iou = 1.0 - dist.min(axis=1).mean()
    return clusters, mean_iou


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def plot_objects_per_image(df):
    """Plot 1: Distribution of object count per image."""
    counts = df.groupby("image_name").size()

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.hist(counts, bins=60, color=ACCENT, edgecolor="white", linewidth=0.5,
            alpha=0.85)
    ax.axvline(counts.mean(), color=ACCENT2, linestyle="--", linewidth=2,
               label=f"Mean = {counts.mean():.1f}")
    ax.axvline(counts.median(), color="#e7298a", linestyle=":", linewidth=2,
               label=f"Median = {counts.median():.1f}")
    ax.set_xlabel("Number of Objects per Image")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Object Count per Image (SKU-110K)")
    ax.legend(frameon=True, fancybox=True)
    fig.tight_layout()
    save_fig(fig, "objects_per_image_histogram.png")

    return counts


def plot_box_dimensions_scatter(df):
    """Plot 2: Normalized width vs height scatter."""
    sample = df.sample(n=min(50000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.scatter(sample["norm_w"], sample["norm_h"],
               s=1, alpha=0.15, color=ACCENT, rasterized=True)
    ax.set_xlabel("Normalized Box Width (w / image_width)")
    ax.set_ylabel("Normalized Box Height (h / image_height)")
    ax.set_title("Ground-Truth Box Dimensions (Normalized)")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)

    # Marginal statistics
    ax.axvline(sample["norm_w"].mean(), color=ACCENT2, linestyle="--",
               alpha=0.7, label=f"Mean W = {sample['norm_w'].mean():.3f}")
    ax.axhline(sample["norm_h"].mean(), color="#e7298a", linestyle="--",
               alpha=0.7, label=f"Mean H = {sample['norm_h'].mean():.3f}")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    save_fig(fig, "box_dimensions_scatter.png")


def plot_aspect_ratio_distribution(df):
    """Plot 3: Histogram of width/height aspect ratios."""
    ar = df["aspect_ratio"].clip(upper=5.0)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.hist(ar, bins=80, color=ACCENT, edgecolor="white", linewidth=0.5,
            alpha=0.85, density=True)
    ax.axvline(ar.median(), color=ACCENT2, linestyle="--", linewidth=2,
               label=f"Median = {ar.median():.2f}")
    ax.set_xlabel("Aspect Ratio (width / height)")
    ax.set_ylabel("Density")
    ax.set_title("Aspect Ratio Distribution of Ground-Truth Boxes")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_fig(fig, "aspect_ratio_distribution.png")


def plot_box_area_distribution(df):
    """Plot 4: Log-scale histogram of normalized box areas."""
    areas = df["norm_area"].clip(lower=1e-8)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    log_areas = np.log10(areas)
    ax.hist(log_areas, bins=80, color=ACCENT, edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.set_xlabel("log10(Normalized Box Area)")
    ax.set_ylabel("Frequency")
    ax.set_title("Box Area Distribution (Log Scale)")
    ax.axvline(np.log10(areas.median()), color=ACCENT2, linestyle="--",
               linewidth=2,
               label=f"Median area = {areas.median():.4f}")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_fig(fig, "box_area_distribution.png")


def plot_pairwise_iou(df, n_sample_images=200):
    """Plot 5: IoU between GT boxes in the same image."""
    print("  Computing pairwise IoU (sampling images) ...")
    images = df["image_name"].unique()
    sample_imgs = np.random.choice(images,
                                   size=min(n_sample_images, len(images)),
                                   replace=False)

    all_ious = []
    for img_name in sample_imgs:
        img_df = df[df["image_name"] == img_name]
        if len(img_df) < 2:
            continue
        boxes = img_df[["x1", "y1", "x2", "y2"]].values.astype(np.float64)

        # For efficiency: subsample if too many boxes
        if len(boxes) > 300:
            idx = np.random.choice(len(boxes), 300, replace=False)
            boxes = boxes[idx]

        iou_mat = compute_iou_matrix(boxes)
        # Upper triangle, exclude diagonal
        triu_idx = np.triu_indices(len(boxes), k=1)
        ious = iou_mat[triu_idx]
        # Keep non-zero IoUs (neighboring / overlapping boxes)
        nonzero = ious[ious > 0.001]
        if len(nonzero) > 0:
            all_ious.append(nonzero)

    if all_ious:
        all_ious = np.concatenate(all_ious)
        # Subsample for plotting
        if len(all_ious) > 500000:
            all_ious = np.random.choice(all_ious, 500000, replace=False)
    else:
        all_ious = np.array([0.0])

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.hist(all_ious, bins=100, color=ACCENT, edgecolor="white",
            linewidth=0.5, alpha=0.85, density=True)
    ax.set_xlabel("Pairwise IoU")
    ax.set_ylabel("Density")
    ax.set_title(f"Pairwise IoU Between GT Boxes (Sampled {len(sample_imgs)} Images)")
    ax.axvline(np.mean(all_ious), color=ACCENT2, linestyle="--", linewidth=2,
               label=f"Mean IoU = {np.mean(all_ious):.3f}")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_fig(fig, "pairwise_iou_histogram.png")


def plot_anchor_kmeans(df):
    """Plot 6: K-means anchor analysis — elbow plot + cluster visualization."""
    print("  Running K-means anchor analysis ...")
    # Prepare normalized wh
    wh = df[["norm_w", "norm_h"]].values.astype(np.float64)
    # Subsample for speed
    if len(wh) > 100000:
        idx = np.random.choice(len(wh), 100000, replace=False)
        wh = wh[idx]

    k_range = range(3, 13)
    mean_ious = []
    all_clusters = {}

    for k in k_range:
        clusters, mean_iou = kmeans_anchors(wh, k)
        mean_ious.append(mean_iou)
        all_clusters[k] = clusters
        print(f"    k={k:2d}  Mean IoU = {mean_iou:.4f}")

    # Best k=9 clusters (YOLO default)
    best_k = 9
    best_clusters = all_clusters[best_k]
    # Sort by area
    areas = best_clusters[:, 0] * best_clusters[:, 1]
    order = np.argsort(areas)
    best_clusters = best_clusters[order]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Elbow plot
    ax = axes[0]
    ax.plot(list(k_range), mean_ious, "o-", color=ACCENT, linewidth=2,
            markersize=8)
    ax.axvline(best_k, color=ACCENT2, linestyle="--", alpha=0.7,
               label=f"k = {best_k}")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Mean IoU with Nearest Anchor")
    ax.set_title("K-Means Anchor Analysis (Elbow Plot)")
    ax.legend(frameon=True)
    ax.set_xticks(list(k_range))

    # Cluster centers visualization
    ax = axes[1]
    colors = plt.cm.Set3(np.linspace(0, 1, best_k))
    for i, (w, h) in enumerate(best_clusters):
        rect = patches.Rectangle(
            (-w / 2, -h / 2), w, h,
            linewidth=2, edgecolor=colors[i], facecolor=colors[i],
            alpha=0.35, label=f"({w:.3f}, {h:.3f})"
        )
        ax.add_patch(rect)
    max_dim = max(best_clusters.max() * 1.2, 0.15)
    ax.set_xlim(-max_dim, max_dim)
    ax.set_ylim(-max_dim, max_dim)
    ax.set_aspect("equal")
    ax.set_xlabel("Normalized Width")
    ax.set_ylabel("Normalized Height")
    ax.set_title(f"Anchor Cluster Centers (k={best_k})")
    ax.legend(fontsize=8, loc="upper left", frameon=True,
              bbox_to_anchor=(1.02, 1.0))

    fig.tight_layout()
    save_fig(fig, "anchor_kmeans_analysis.png")

    return best_clusters, mean_ious, list(k_range)


def plot_sample_images(df):
    """Plot 7: Sample images with GT annotations or synthetic representations."""
    fig, axes = plt.subplots(2, 4, figsize=FIGSIZE_GRID)
    axes = axes.flatten()

    sample_images = df["image_name"].unique().tolist()
    np.random.shuffle(sample_images)
    sample_images = sample_images[:8]

    real_images_available = IMAGES_DIR.exists() and any(IMAGES_DIR.iterdir()) \
        if IMAGES_DIR.exists() else False

    for idx, img_name in enumerate(sample_images):
        ax = axes[idx]
        img_df = df[df["image_name"] == img_name].head(300)  # limit for display
        total_objs = len(df[df["image_name"] == img_name])
        iw = img_df["image_width"].iloc[0]
        ih = img_df["image_height"].iloc[0]

        if real_images_available:
            # Try loading real image
            img_path = IMAGES_DIR / img_name
            if img_path.exists():
                import matplotlib.image as mpimg
                try:
                    img = mpimg.imread(str(img_path))
                    ax.imshow(img)
                except Exception:
                    ax.set_facecolor("#f0f0f0")
                    ax.set_xlim(0, iw)
                    ax.set_ylim(ih, 0)
            else:
                ax.set_facecolor("#f0f0f0")
                ax.set_xlim(0, iw)
                ax.set_ylim(ih, 0)
        else:
            # Synthetic background
            ax.set_facecolor("#f5f5f5")
            ax.set_xlim(0, iw)
            ax.set_ylim(ih, 0)

        # Draw GT boxes
        colors_cycle = plt.cm.tab20(np.linspace(0, 1, 20))
        for j, (_, row) in enumerate(img_df.iterrows()):
            rect = patches.Rectangle(
                (row["x1"], row["y1"]),
                row["x2"] - row["x1"],
                row["y2"] - row["y1"],
                linewidth=0.5, edgecolor=colors_cycle[j % 20],
                facecolor="none", alpha=0.7,
            )
            ax.add_patch(rect)

        short_name = img_name[:20] + "..." if len(img_name) > 23 else img_name
        ax.set_title(f"{short_name}\n({total_objs} objects)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    data_label = "Real" if real_images_available else "Synthetic"
    fig.suptitle(f"Sample Images with Ground-Truth Boxes ({data_label} Data)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "sample_images_with_annotations.png")


def plot_data_split_statistics(df):
    """Plot 8: Bar chart of train/val/test counts."""
    split_img_counts = df.groupby("split")["image_name"].nunique()
    split_box_counts = df.groupby("split").size()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Images per split
    ax = axes[0]
    splits_ordered = ["train", "val", "test"]
    img_vals = [split_img_counts.get(s, 0) for s in splits_ordered]
    bars = ax.bar(splits_ordered, img_vals,
                  color=[PALETTE[0], PALETTE[1], PALETTE[2]],
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, img_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha="center", va="bottom", fontweight="bold",
                fontsize=11)
    ax.set_ylabel("Number of Images")
    ax.set_title("Images per Data Split")

    # Annotations per split
    ax = axes[1]
    box_vals = [split_box_counts.get(s, 0) for s in splits_ordered]
    bars = ax.bar(splits_ordered, box_vals,
                  color=[PALETTE[0], PALETTE[1], PALETTE[2]],
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, box_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                f"{val:,}", ha="center", va="bottom", fontweight="bold",
                fontsize=11)
    ax.set_ylabel("Number of Annotations")
    ax.set_title("Annotations per Data Split")

    fig.suptitle("Dataset Split Statistics (SKU-110K)", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "data_split_statistics.png")


# ---------------------------------------------------------------------------
# Save anchor config
# ---------------------------------------------------------------------------
def save_anchor_config(clusters, mean_ious, k_range):
    """Save K-means anchors to a YAML config file."""
    # Convert normalized anchors to pixel values at 640 input size
    input_size = 640
    pixel_anchors = (clusters * input_size).tolist()

    config = {
        "anchor_analysis": {
            "method": "kmeans_iou",
            "input_size": input_size,
            "best_k": len(clusters),
            "mean_iou_at_best_k": float(mean_ious[k_range.index(len(clusters))]),
            "elbow_data": {
                f"k{k}": float(miou) for k, miou in zip(k_range, mean_ious)
            },
        },
        "custom_anchors": {
            "normalized": [
                [round(float(w), 4), round(float(h), 4)]
                for w, h in clusters
            ],
            "pixel_640": [
                [round(w, 1), round(h, 1)]
                for w, h in pixel_anchors
            ],
        },
        "anchor_groups": {
            "small": [
                [round(float(w), 4), round(float(h), 4)]
                for w, h in clusters[:3]
            ],
            "medium": [
                [round(float(w), 4), round(float(h), 4)]
                for w, h in clusters[3:6]
            ],
            "large": [
                [round(float(w), 4), round(float(h), 4)]
                for w, h in clusters[6:]
            ],
        },
    }

    config_path = CONFIGS_DIR / "custom_anchors.yaml"
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved anchor config: {config_path}")


# ---------------------------------------------------------------------------
# Print statistics
# ---------------------------------------------------------------------------
def print_statistics(df):
    """Print key dataset statistics to console."""
    print("\n" + "=" * 70)
    print("  SKU-110K Dataset Statistics")
    print("=" * 70)

    n_images = df["image_name"].nunique()
    n_annotations = len(df)
    print(f"  Total images:       {n_images:>10,}")
    print(f"  Total annotations:  {n_annotations:>10,}")
    print(f"  Avg objects/image:  {n_annotations / n_images:>10.1f}")
    print()

    for split in SPLITS:
        sdf = df[df["split"] == split]
        n_img = sdf["image_name"].nunique()
        n_ann = len(sdf)
        avg = n_ann / max(n_img, 1)
        print(f"  {split:>5s}:  {n_img:>6,} images,  {n_ann:>10,} annotations  "
              f"(avg {avg:.1f} obj/img)")

    print()
    counts = df.groupby("image_name").size()
    print(f"  Objects/image  min: {counts.min()},  max: {counts.max()},  "
          f"mean: {counts.mean():.1f},  std: {counts.std():.1f}")

    print(f"\n  Normalized box width   mean: {df['norm_w'].mean():.4f},  "
          f"std: {df['norm_w'].std():.4f}")
    print(f"  Normalized box height  mean: {df['norm_h'].mean():.4f},  "
          f"std: {df['norm_h'].std():.4f}")
    print(f"  Aspect ratio (w/h)     mean: {df['aspect_ratio'].mean():.3f},  "
          f"median: {df['aspect_ratio'].median():.3f}")
    print(f"  Normalized area        mean: {df['norm_area'].mean():.5f},  "
          f"median: {df['norm_area'].median():.5f}")

    # Resolution distribution
    resolutions = df.groupby("image_name")[["image_width", "image_height"]].first()
    res_str = resolutions.apply(
        lambda r: f"{int(r['image_width'])}x{int(r['image_height'])}", axis=1
    )
    print(f"\n  Unique resolutions: {res_str.nunique()}")
    for res, cnt in res_str.value_counts().head(5).items():
        print(f"    {res}: {cnt:,} images")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("  SKU-110K Exploratory Data Analysis")
    print("=" * 70 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    # Load data
    df = load_data()

    # Print statistics
    print_statistics(df)

    # Generate all 8 plots
    print("Generating visualizations ...\n")

    print("[1/8] Objects per image histogram")
    plot_objects_per_image(df)

    print("[2/8] Box dimensions scatter")
    plot_box_dimensions_scatter(df)

    print("[3/8] Aspect ratio distribution")
    plot_aspect_ratio_distribution(df)

    print("[4/8] Box area distribution")
    plot_box_area_distribution(df)

    print("[5/8] Pairwise IoU histogram")
    plot_pairwise_iou(df)

    print("[6/8] Anchor K-means analysis")
    clusters, mean_ious, k_range = plot_anchor_kmeans(df)

    print("[7/8] Sample images with annotations")
    plot_sample_images(df)

    print("[8/8] Data split statistics")
    plot_data_split_statistics(df)

    # Save anchor config
    print("\nSaving anchor configuration ...")
    save_anchor_config(clusters, mean_ious, k_range)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("Done.\n")


if __name__ == "__main__":
    main()
