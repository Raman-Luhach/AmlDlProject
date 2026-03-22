# High-Density Object Segmentation using YOLACT + MobileNetV3 + Soft-NMS

> **B.Tech Final Year Project** -- Applied Machine Learning & Deep Learning
> **Authors:** Raman Luhach (230107), Rachit Kumar (230128)
> **Dataset:** [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) (dense retail shelf scenes)

---

## Overview

This project tackles **high-density object detection and instance segmentation** in cluttered retail shelf images using the **SKU-110K** dataset. Retail scenes contain hundreds of tightly packed, visually similar products per image, making standard NMS-based detectors fail due to aggressive suppression of valid overlapping detections.

We implement a lightweight **YOLACT** (You Only Look At CoefficienTs) architecture with a **MobileNetV3-Large** backbone (~10M parameters), a **Feature Pyramid Network (FPN)** for multi-scale detection, and **Soft-NMS** with Gaussian decay to preserve detections in dense regions. The model is exported to **ONNX** with **INT8 quantization** for edge deployment.

### Key Contributions

- Custom YOLACT architecture with MobileNetV3-Large backbone (10M params vs. 30M+ for ResNet-101)
- Feature Pyramid Network with 5 levels (P3-P7) for detecting objects across scales
- ProtoNet generating 32 prototype masks for instance segmentation
- Soft-NMS with Gaussian decay (sigma=0.5) replacing standard NMS for dense scenes
- Focal Loss (alpha=0.25, gamma=2.0) to handle extreme foreground-background imbalance
- K-means anchor optimization (IoU-based) derived from EDA on SKU-110K annotations
- ONNX export with INT8 quantization for edge/mobile deployment
- Apple M4 (MPS) backend support for training and inference

---

## Architecture

```
                         Input Image (3 x 550 x 550)
                                    |
                    +===============================+
                    |     MobileNetV3-Large          |
                    |       (ImageNet pretrained)     |
                    +===============================+
                      |              |              |
                  C3 (40ch)     C4 (112ch)     C5 (960ch)
                  stride 8      stride 16      stride ~32
                      |              |              |
                    +===============================+
                    |   Feature Pyramid Network      |
                    |       (256ch, 5 levels)         |
                    +===============================+
                      |    |    |    |    |
                     P3   P4   P5   P6   P7
                      |    |    |    |    |
          +-----------+    |    |    |    +----------+
          |                |    |    |               |
          v                v    v    v               v
    +============+   +=========================+
    |  ProtoNet  |   |   Prediction Head       |
    |  (from P3) |   |   (shared across all)   |
    +============+   +=========================+
          |                |       |       |
     32 Prototype     Class    Box Reg   Mask
       Masks          Scores   Offsets   Coeffs
          |                |       |       |
          +--------+-------+-------+-------+
                   |
          +==================+
          |  Assembly Layer  |
          | masks = proto @  |
          |   coeffs^T      |
          +==================+
                   |
          +==================+
          |    Soft-NMS      |
          | (Gaussian decay) |
          +==================+
                   |
            Final Detections
         (boxes, scores, masks)
```

### Parameter Breakdown

| Component        | Parameters |
|------------------|------------|
| MobileNetV3-Large (backbone) | ~4.2M |
| Feature Pyramid Network      | ~3.5M |
| ProtoNet (32 prototypes)     | ~0.5M |
| Prediction Head              | ~1.8M |
| **Total**                    | **~10M** |

---

## Project Structure

```
AmlDlProject/
|
|-- configs/
|   |-- default.yaml              # Full training configuration
|   +-- custom_anchors.yaml       # K-means optimized anchors from EDA
|
|-- notebooks/
|   |-- 01_EDA_and_Data_Analysis.ipynb
|   |-- 02_Classic_ML_Baseline.ipynb
|   |-- 03_DL_Training_and_Evaluation.ipynb
|   +-- 04_ONNX_Deployment.ipynb
|
|-- report/
|   |-- main.tex                  # LaTeX project report
|   +-- references.bib            # Bibliography
|
|-- results/
|   |-- eda/                      # 8 EDA plots (distributions, anchors, samples)
|   |-- baseline/                 # HOG + SVM detection results
|   |-- eval/                     # Detection metrics, PR curves, density analysis
|   +-- deployment/               # ONNX models, INT8 quantized, benchmarks
|
|-- scripts/
|   |-- download_data.sh          # Download and extract SKU-110K dataset
|   |-- run_eda.py                # Run exploratory data analysis
|   |-- run_baseline.py           # Run HOG + SVM baseline
|   |-- train.py                  # Train YOLACT model
|   |-- evaluate.py               # Evaluate trained model (COCO-style mAP)
|   +-- export.py                 # Export to ONNX + INT8 quantization
|
|-- src/
|   |-- __init__.py
|   |-- baseline/
|   |   +-- hog_svm.py            # HOG feature extractor + Linear SVM
|   |-- data/
|   |   |-- anchors.py            # Anchor generation + encoding/decoding
|   |   |-- augmentations.py      # Training augmentations (albumentations)
|   |   +-- dataset.py            # SKU-110K PyTorch Dataset + DataLoader
|   |-- deployment/
|   |   |-- benchmark.py          # Latency/throughput benchmarking
|   |   |-- export_onnx.py        # ONNX export (opset 11)
|   |   +-- quantize.py           # INT8 post-training quantization
|   |-- evaluation/
|   |   |-- evaluator.py          # COCO-style mAP evaluator
|   |   +-- metrics.py            # Precision, recall, AP computation
|   |-- models/
|   |   |-- backbone.py           # MobileNetV3-Large with C3/C4/C5 tap points
|   |   |-- detection.py          # Post-processing (decode + Soft-NMS)
|   |   |-- fpn.py                # Feature Pyramid Network (P3-P7)
|   |   |-- prediction_head.py    # Classification, box, mask coefficient heads
|   |   |-- protonet.py           # Prototype mask generator (32 masks)
|   |   +-- yolact.py             # Full YOLACT model assembly
|   |-- training/
|   |   |-- losses.py             # Focal Loss + YOLACT multi-task loss
|   |   +-- trainer.py            # Training loop (SGD, cosine LR, AMP)
|   +-- utils/
|       |-- helpers.py            # Device detection, checkpointing, formatting
|       |-- soft_nms.py           # Soft-NMS (Gaussian/Linear) + Hard-NMS
|       +-- visualization.py     # Detection and mask visualization
|
|-- tests/                        # Unit tests
|-- weights/                      # Model checkpoints (git-ignored)
|-- Makefile                      # Build targets for full pipeline
|-- requirements.txt              # Python dependencies
|-- setup.py                      # Package installation
|-- LICENSE                       # MIT License
+-- README.md                     # This file
```

---

## Installation

### Prerequisites

- Python >= 3.8
- macOS (Apple Silicon M-series with MPS) or Linux with CUDA
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/AmlDlProject.git
cd AmlDlProject

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Or use the Makefile
make install
```

### Dependencies

| Package         | Version  | Purpose                         |
|-----------------|----------|---------------------------------|
| torch           | >= 2.0.0 | Deep learning framework         |
| torchvision     | >= 0.15.0| Pretrained backbones, transforms|
| onnxruntime     | >= 1.16.0| ONNX model inference            |
| onnx            | >= 1.14.0| ONNX export format              |
| opencv-python   | >= 4.8.0 | Image I/O and processing        |
| scikit-learn    | >= 1.3.0 | SVM baseline, K-means anchors   |
| albumentations  | >= 1.3.0 | Training data augmentations     |
| pycocotools     | >= 2.0   | COCO-style evaluation metrics   |
| matplotlib      | >= 3.7.0 | Plotting and visualization      |

---

## Quick Start

The full pipeline can be run step-by-step using Make targets:

```bash
# 1. Download the SKU-110K dataset (~2.4 GB)
make download-data

# 2. Run Exploratory Data Analysis (generates 8 plots in results/eda/)
make eda

# 3. Run the HOG + Linear SVM baseline
make baseline

# 4. Train the YOLACT model
make train

# 5. Evaluate on the test set (COCO-style mAP)
make evaluate

# 6. Export to ONNX + INT8 quantization
make export

# 7. Run inference benchmarks
make benchmark

# 8. Compile the LaTeX report (requires pdflatex + bibtex)
make report

# Or run everything end-to-end
make all
```

### Training Configuration

All hyperparameters are in `configs/default.yaml`:

```yaml
training:
  epochs: 20
  batch_size: 8
  optimizer: sgd
  lr: 0.001
  momentum: 0.9
  weight_decay: 0.0005
  warmup_epochs: 3
  scheduler: cosine
  gradient_clip: 10.0
  amp: true               # Mixed precision (CUDA only)

loss:
  cls_weight: 1.0          # Focal loss weight
  box_weight: 1.5          # Smooth L1 weight
  mask_weight: 6.125       # Mask BCE weight
  focal_alpha: 0.25
  focal_gamma: 2.0

softnms:
  method: gaussian
  sigma: 0.5
  score_threshold: 0.001
```

---

## Results

### Exploratory Data Analysis

The EDA phase produces 8 analysis plots saved to `results/eda/`:

| Plot | Description |
|------|-------------|
| `objects_per_image_histogram.png` | Distribution of object counts per image (mean ~147) |
| `box_area_distribution.png` | Bounding box area distribution |
| `aspect_ratio_distribution.png` | Width/height aspect ratio analysis |
| `box_dimensions_scatter.png` | Width vs. height scatter plot |
| `pairwise_iou_histogram.png` | IoU overlap distribution (justifies Soft-NMS) |
| `sample_images_with_annotations.png` | Sample images with ground truth boxes |
| `data_split_statistics.png` | Train/val/test split statistics |
| `anchor_kmeans_analysis.png` | K-means anchor clustering (k=9, mean IoU=0.744) |

### Baseline: HOG + Linear SVM

| Metric          | Value       |
|-----------------|-------------|
| mAP@0.5         | _TBD_       |
| Precision       | _TBD_       |
| Recall          | _TBD_       |
| Inference Time  | _TBD_       |

### YOLACT + MobileNetV3 + Soft-NMS

| Metric            | Value       |
|-------------------|-------------|
| mAP@0.50          | _TBD_       |
| mAP@0.75          | _TBD_       |
| mAP@[.50:.95]     | _TBD_       |
| AP (small)        | _TBD_       |
| AP (medium)       | _TBD_       |
| AP (large)        | _TBD_       |
| Parameters        | ~10M        |
| Input Resolution  | 550 x 550   |

### Soft-NMS vs. Hard-NMS Ablation

| NMS Method        | mAP@0.50 | mAP@0.75 | Detections/Image |
|-------------------|----------|----------|------------------|
| Hard NMS (0.5)    | _TBD_    | _TBD_    | _TBD_            |
| Soft-NMS Gaussian | _TBD_    | _TBD_    | _TBD_            |
| Soft-NMS Linear   | _TBD_    | _TBD_    | _TBD_            |

### Deployment Benchmarks

| Model Variant     | Size (MB) | Latency (ms) | Throughput (FPS) |
|-------------------|-----------|--------------|------------------|
| PyTorch FP32      | _TBD_     | _TBD_        | _TBD_            |
| ONNX FP32         | _TBD_     | _TBD_        | _TBD_            |
| ONNX INT8         | _TBD_     | _TBD_        | _TBD_            |

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_EDA_and_Data_Analysis.ipynb` | Dataset exploration, annotation statistics, K-means anchor analysis |
| 2 | `02_Classic_ML_Baseline.ipynb` | HOG feature extraction + Linear SVM sliding window detector |
| 3 | `03_DL_Training_and_Evaluation.ipynb` | YOLACT model training, loss curves, COCO-style evaluation |
| 4 | `04_ONNX_Deployment.ipynb` | ONNX export, INT8 quantization, latency benchmarking |

---

## Technical Details

### Soft-NMS (Gaussian Decay)

Standard NMS hard-suppresses all detections overlapping above an IoU threshold, causing missed detections in dense scenes. Soft-NMS instead decays the confidence score:

```
score_i = score_i * exp(-IoU(M, b_i)^2 / sigma)
```

where `M` is the current highest-scoring box, `b_i` is a candidate box, and `sigma` controls the decay rate. This allows closely overlapping but distinct objects to survive post-processing.

### Focal Loss

With ~147 objects per image and thousands of anchors, most anchors are background (negative). Focal Loss addresses this imbalance:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

with `alpha=0.25` for the foreground class and `gamma=2.0` to down-weight easy negatives.

### Multi-Task Loss

The total YOLACT loss combines three objectives:

```
L = 1.0 * L_cls (Focal) + 1.5 * L_box (Smooth L1) + 6.125 * L_mask (BCE)
```

### Device Support

| Backend | Training | Inference | AMP  |
|---------|----------|-----------|------|
| CUDA    | Yes      | Yes       | Yes  |
| MPS (Apple Silicon) | Yes | Yes | No (FP32) |
| CPU     | Yes      | Yes       | No   |

---

## Citation

If you find this work useful, please cite:

```bibtex
@thesis{luhaj2026highdensity,
  title   = {High-Density Object Segmentation using YOLACT + MobileNetV3 + Soft-NMS
             on SKU-110K Dataset},
  author  = {Luhach, Raman and Kumar, Rachit},
  year    = {2026},
  type    = {B.Tech Final Year Project},
  note    = {Applied Machine Learning and Deep Learning}
}
```

### References

- **YOLACT:** Bolya et al., "YOLACT: Real-time Instance Segmentation," ICCV 2019. [arXiv:1904.02689](https://arxiv.org/abs/1904.02689)
- **MobileNetV3:** Howard et al., "Searching for MobileNetV3," ICCV 2019. [arXiv:1905.02244](https://arxiv.org/abs/1905.02244)
- **Soft-NMS:** Bodla et al., "Soft-NMS -- Improving Object Detection With One Line of Code," ICCV 2017. [arXiv:1704.04503](https://arxiv.org/abs/1704.04503)
- **SKU-110K:** Goldman et al., "Precise Detection in Densely Packed Scenes," CVPR 2019. [arXiv:1904.00853](https://arxiv.org/abs/1904.00853)
- **Focal Loss:** Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- **FPN:** Lin et al., "Feature Pyramid Networks for Object Detection," CVPR 2017. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)

---

## License

This project is licensed under the [MIT License](LICENSE).

```
MIT License
Copyright (c) 2026 Raman Luhach, Rachit Kumar
```
