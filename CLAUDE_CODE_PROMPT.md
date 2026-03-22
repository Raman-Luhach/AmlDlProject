# CLAUDE CODE MASTER PROMPT — High-Density Object Segmentation Project

> **CRITICAL**: You have exactly 5 hours. Read the `RESEARCH_REPORT.md` in this repo root for full context. Use `Task` tool to spawn parallel sub-agents aggressively. Never do sequentially what can be done in parallel.

## PROJECT OVERVIEW

Build a complete B.Tech final project: **High-Density Object Segmentation using YOLACT + MobileNetV3 + Soft-NMS on SKU-110K dataset**, with a classic ML baseline (HOG+SVM), deep learning pipeline, ONNX edge deployment, LaTeX report, and professional GitHub repo. Target: 9-10/10 on all rubric criteria.

## REPO STRUCTURE TO CREATE FIRST

```
high-density-object-segmentation/
├── configs/
│   ├── default.yaml
│   ├── mobilenetv3_yolact.yaml
│   └── training.yaml
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py          # MobileNetV3-Large backbone with FPN tap points
│   │   ├── fpn.py               # Feature Pyramid Network
│   │   ├── yolact.py            # Full YOLACT architecture
│   │   ├── protonet.py          # Prototype mask generation network
│   │   ├── prediction_head.py   # Detection + mask coefficient heads
│   │   └── detection.py         # Post-processing with Soft-NMS
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # SKU110K dataset loader (CSV → COCO format)
│   │   ├── augmentations.py     # SSD-style + density-aware augmentations
│   │   └── anchors.py           # K-means anchor generation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loop with AMP, cosine annealing
│   │   └── losses.py            # Focal loss, Smooth L1, mask BCE
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py         # COCO-style mAP evaluation
│   │   └── metrics.py           # AP, AR computation
│   ├── baseline/
│   │   ├── __init__.py
│   │   └── hog_svm.py           # Complete HOG+SVM classic ML pipeline
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── export_onnx.py       # PyTorch → ONNX export
│   │   ├── quantize.py          # INT8 quantization
│   │   └── benchmark.py         # Latency/FPS/size benchmarking
│   └── utils/
│       ├── __init__.py
│       ├── soft_nms.py          # Gaussian Soft-NMS implementation
│       ├── visualization.py     # Detection visualization, EDA plots
│       └── helpers.py           # Misc utilities
├── notebooks/
│   ├── 01_EDA_and_Data_Analysis.ipynb
│   ├── 02_Classic_ML_Baseline.ipynb
│   ├── 03_DL_Training_and_Evaluation.ipynb
│   └── 04_ONNX_Deployment.ipynb
├── scripts/
│   ├── download_data.sh
│   ├── train.py
│   ├── evaluate.py
│   ├── export.py
│   └── run_baseline.py
├── report/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
├── tests/
│   ├── test_model.py
│   ├── test_soft_nms.py
│   └── test_dataset.py
├── results/
│   └── .gitkeep
├── Makefile
├── requirements.txt
├── setup.py
├── README.md
├── .gitignore
├── RESEARCH_REPORT.md
└── LICENSE
```

---

## PHASE 1 — FOUNDATION (0:00–0:30) — Run 3 sub-agents in parallel

### Sub-Agent 1: Project Scaffolding + Git Init
```
Create the entire folder structure shown above. Initialize git repo.
Create .gitignore (Python, PyTorch, Jupyter, __pycache__, *.pyc, data/, weights/, *.onnx, wandb/, .ipynb_checkpoints/).
Create requirements.txt with:
torch>=2.0.0
torchvision>=0.15.0
onnxruntime>=1.16.0
onnx>=1.14.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
scikit-image>=0.21.0
scipy>=1.11.0
numpy>=1.24.0
pandas>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
pycocotools>=2.0
tqdm>=4.65.0
pyyaml>=6.0
Pillow>=10.0.0
albumentations>=1.3.0
torchmetrics>=1.0.0

Create Makefile with targets: install, download-data, eda, baseline, train, evaluate, export, benchmark, report, clean, all.
Create setup.py for the src package.
Create configs/default.yaml with ALL hyperparameters:
  - dataset: sku110k, data_dir, num_classes: 1, input_size: 550
  - backbone: mobilenetv3_large, pretrained: true
  - fpn: channels: 256, num_levels: 5
  - yolact: num_prototypes: 32, num_masks: 32
  - anchors: sizes, ratios, scales (to be updated after K-means)
  - training: lr: 0.001, momentum: 0.9, weight_decay: 5e-4, epochs: 20, batch_size: 8, warmup_epochs: 3, scheduler: cosine
  - loss: focal_alpha: 0.25, focal_gamma: 2.0, box_weight: 1.5, mask_weight: 6.125, cls_weight: 1.0
  - softnms: sigma: 0.5, score_threshold: 0.001, method: gaussian
  - eval: iou_thresholds: [0.5, 0.55, ..., 0.95], max_detections: [1, 10, 100, 300]
  - deployment: opset: 11, quantize: int8
Create LICENSE (MIT).
Make initial commit: "feat: project scaffolding and configuration"
```

### Sub-Agent 2: Data Download + Preprocessing Pipeline
```
Create scripts/download_data.sh that:
1. Downloads SKU-110K from the Ultralytics auto-download URL OR Google Drive mirror
   - Primary: Use ultralytics pip package to auto-download:
     python -c "from ultralytics.utils.downloads import download; download('https://huggingface.co/datasets/Ultralytics/SKU-110K/resolve/main/SKU110K_fixed.tar.gz', dir='data/')"
   - Fallback: wget/curl from available mirrors
   - If full dataset too slow (>15 min), use Roboflow SKU110K subset OR download only first 3000 train images
2. Extracts to data/SKU110K_fixed/

Create src/data/dataset.py that:
1. Parses SKU-110K CSV annotations (columns: image_name, x1, y1, x2, y2, class, image_width, image_height)
2. Converts to COCO JSON format and caches the conversion
3. Implements PyTorch Dataset class with __getitem__ returning:
   - image tensor (normalized, resized to 550x550)
   - target dict: {'boxes': tensor, 'labels': tensor, 'masks': tensor, 'image_id': int, 'area': tensor}
4. For masks (SKU-110K has NO segmentation masks, only boxes):
   - Generate pseudo-masks from bounding boxes (filled rectangles as binary masks)
   - This is a known limitation — document it in code comments
5. Data cleaning in __init__:
   - Skip corrupt images (try PIL.open, catch exceptions)
   - Remove zero-area boxes (x1==x2 or y1==y2)
   - Clip coordinates to image bounds
   - Remove duplicates
6. Supports train/val/test splits per official SKU-110K split
7. Collate function for variable-length targets

Create src/data/augmentations.py with:
1. SSD-style augmentation pipeline: random crop, random expand, random mirror, photometric distortion
2. Resize to 550x550
3. Normalize with ImageNet mean/std
4. Density-aware augmentation: random occlusion simulation (paste random product crops over others)
5. Use albumentations for composition

Create src/data/anchors.py with:
1. K-means anchor generation using 1-IoU distance metric
2. Run on ground-truth boxes from training set
3. Test k=5,7,9 and output elbow plot
4. Default anchor generation matching YOLACT specs (5 FPN levels, 3 aspect ratios per level)
5. Function to compute anchor-GT IoU statistics

Commit: "feat: data pipeline with SKU-110K loader, augmentations, and anchor generation"
```

### Sub-Agent 3: Core Model Architecture
```
Create src/models/backbone.py:
1. MobileNetV3LargeBackbone class wrapping torchvision.models.mobilenet_v3_large(weights='IMAGENET1K_V2')
2. Extract features at 3 stages mapping to C3/C4/C5:
   - C3: after layer index 6 (stride 8, ~40 channels) 
   - C4: after layer index 12 (stride 16, ~112 channels)
   - C5: after layer index 16 (stride 32, ~960 channels)
3. Return dict of {'C3': tensor, 'C4': tensor, 'C5': tensor}
4. Support freezing early layers for fine-tuning
5. Print parameter count and verify output shapes in __main__ block

IMPORTANT: The exact layer indices for MobileNetV3-Large features extraction:
- Access via model.features (Sequential of InvertedResidual blocks)
- Split into 3 groups and use forward hooks or manual splitting
- Verify stride and channel dims with a dummy forward pass of shape (1, 3, 550, 550)

Create src/models/fpn.py:
1. FPN class taking C3/C4/C5 feature maps
2. Lateral connections: 1x1 conv to reduce channels to 256
3. Top-down pathway with bilinear upsampling + element-wise addition
4. 3x3 conv smoothing on each merged level
5. 2 additional downsample levels (P6, P7) via stride-2 3x3 convs
6. Output: [P3, P4, P5, P6, P7] all with 256 channels

Create src/models/protonet.py:
1. ProtoNet class taking P3 (largest FPN feature map)
2. Architecture: 3x conv3x3(256→256) + bilinear upsample 2x + conv3x3(256→256) + conv1x1(256→32)
3. ReLU after each conv except the last (which uses ReLU to keep prototypes positive, per YOLACT paper)
4. Output shape: (batch, 32, H/4, W/4) where H,W is input resolution

Create src/models/prediction_head.py:
1. PredictionHead class applied to each FPN level
2. Shared conv layers (configurable depth, default 1 layer of 256 channels)
3. Three parallel output branches:
   - Classification: conv → (num_anchors * num_classes) channels
   - Box regression: conv → (num_anchors * 4) channels  
   - Mask coefficients: conv → (num_anchors * num_prototypes) channels with tanh activation
4. Weight sharing across FPN levels (same conv weights applied to P3-P7)

Create src/models/detection.py:
1. Detect class for post-processing
2. Decode boxes from anchor offsets
3. Apply Soft-NMS (import from src/utils/soft_nms.py)
4. Crop masks by predicted bounding boxes
5. Score thresholding and top-k filtering
6. Max detections: 300 (configurable, must be high for dense scenes)

Create src/models/yolact.py:
1. YOLACT class combining all components:
   - self.backbone = MobileNetV3LargeBackbone()
   - self.fpn = FPN(backbone_channels=[40, 112, 960])
   - self.protonet = ProtoNet()
   - self.prediction_head = PredictionHead(num_classes=1+1, num_prototypes=32) # +1 for background
   - self.detect = Detect()
2. Forward pass:
   a. features = backbone(image) → {C3, C4, C5}
   b. fpn_features = fpn(features) → [P3, P4, P5, P6, P7]
   c. prototypes = protonet(fpn_features[0]) → (B, 32, H, W)
   d. predictions = [prediction_head(p) for p in fpn_features] → per-level (cls, box, mask_coeff)
   e. Concatenate predictions across levels
   f. If training: return (predictions, prototypes) for loss
   g. If eval: return detect(predictions, prototypes, prior_boxes)
3. Mask assembly: M = sigmoid(prototypes @ mask_coefficients.T), then crop by box
4. Count and print total parameters
5. Test with dummy input in __main__

Create src/utils/soft_nms.py:
1. Implement Gaussian Soft-NMS:
   def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.001):
     - For each detection in score-descending order:
       - Compute IoU with all remaining detections
       - Decay scores: s_i = s_i * exp(-iou^2 / sigma)
       - Remove detections below score_threshold
     - Return filtered boxes, scores, indices
2. Also implement standard hard NMS for comparison
3. Support both CPU and GPU tensors
4. Include unit test in __main__

Commit: "feat: YOLACT architecture with MobileNetV3 backbone, FPN, ProtoNet, and Soft-NMS"
```

---

## PHASE 2 — PARALLEL EXECUTION (0:30–2:00) — Run 4 sub-agents in parallel

### Sub-Agent 4: EDA + Visualization (runs on CPU, parallel with everything)
```
Create notebooks/01_EDA_and_Data_Analysis.ipynb (as a Python script that generates all plots):
Actually, create src/utils/visualization.py AND a script scripts/run_eda.py that generates all EDA plots to results/ folder.

The EDA script MUST produce these 8 visualizations (save as PNG to results/eda/):
1. objects_per_image_histogram.png — Distribution of object count per image (expect peak ~147)
2. box_dimensions_scatter.png — Normalized width vs height scatter of all GT boxes
3. aspect_ratio_distribution.png — Histogram of width/height ratios
4. box_area_distribution.png — Log-scale histogram of box areas (normalized)
5. pairwise_iou_histogram.png — IoU between neighboring GT boxes (sample 1000 images, compute IoU for all pairs within each image, histogram the IoU values > 0)
6. anchor_kmeans_analysis.png — Elbow plot (k vs mean IoU) + visualization of cluster centers for k=9
7. sample_images_with_annotations.png — 4x2 grid showing 8 sample images with GT boxes drawn
8. data_split_statistics.png — Bar chart of train/val/test image counts and total annotation counts

Each plot must have: title, axis labels, proper font sizes, tight_layout, high DPI (150+).
Print key statistics to console: total images, total annotations, mean/median/min/max objects per image, mean box area, annotation format details.

Run K-means anchor analysis and save optimal anchors to configs/custom_anchors.yaml

Commit: "feat: comprehensive EDA with 8 publication-quality visualizations"
```

### Sub-Agent 5: Classic ML Baseline (runs on CPU)
```
Create src/baseline/hog_svm.py — COMPLETE standalone pipeline:

class HOGSVMBaseline:
    def __init__(self, window_size=(64,64), cell_size=(8,8), block_size=(16,16), nbins=9):
        # HOG parameters
        # LinearSVC with C=0.01
    
    def extract_hog_features(self, image_patch):
        # Use cv2.HOGDescriptor or skimage.feature.hog
        # Return feature vector of ~1764 dims
    
    def prepare_training_data(self, dataset, num_pos=5000, num_neg=10000):
        # Extract positive crops from GT boxes (resize to window_size)
        # Extract random negative patches from background regions
        # Apply HOG to all patches
        # Return X_train, y_train
    
    def train(self, X_train, y_train):
        # Train LinearSVC
        # Hard negative mining: run detector on training images,
        # collect false positives, add to training set, retrain
    
    def sliding_window(self, image, scales=[1.0, 0.75, 0.5, 0.35], step_size=16):
        # Multi-scale sliding window
        # Return list of (x, y, scale, score) detections
    
    def detect(self, image):
        # Run sliding window
        # Apply standard NMS (threshold=0.3)
        # Return boxes, scores
    
    def evaluate(self, dataset, num_images=100):
        # Compute mAP@0.5 using pycocotools
        # Return metrics dict

Create scripts/run_baseline.py:
1. Load SKU-110K validation subset (100-200 images for speed)
2. Train HOG+SVM on 500 training images (sample for speed)
3. Evaluate and save results to results/baseline/
4. Generate comparison visualizations:
   - baseline_detections.png — 4 images with HOG+SVM detections overlaid
   - baseline_metrics.json — mAP, precision, recall numbers
5. Print: "HOG+SVM Baseline: mAP@0.5 = X.XX% (expected 5-12%)"
6. Save timing info: training time, inference time per image

USE ONLY 200-500 images for training and 50-100 for eval to stay within 15-20 min total.

Commit: "feat: HOG+SVM classic ML baseline with evaluation"
```

### Sub-Agent 6: Training Infrastructure
```
Create src/training/losses.py:
1. FocalLoss class:
   def forward(self, predictions, targets):
     # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
     # alpha=0.25, gamma=2.0
     # Handle background class properly
   
2. YOLACTLoss class combining:
   - Focal loss for classification (weight: 1.0)
   - Smooth L1 loss for box regression (weight: 1.5)
   - Binary cross-entropy for masks (weight: 6.125/num_pos)
   - Match predictions to GT using IoU-based matching (threshold 0.5)
   
3. Anchor matching strategy:
   - Compute IoU between all anchors and all GT boxes
   - Positive: IoU > 0.5, Negative: IoU < 0.4
   - Hard negative mining: keep neg:pos ratio at 3:1

Create src/training/trainer.py:
1. Trainer class:
   def __init__(self, model, train_loader, val_loader, config):
     # SGD optimizer with momentum 0.9, weight_decay 5e-4
     # CosineAnnealingLR scheduler
     # GradScaler for mixed precision (AMP)
     # Best model checkpoint saving
   
   def train_epoch(self):
     # Standard training loop with AMP (torch.cuda.amp.autocast)
     # Gradient clipping: max_norm=10.0
     # Log losses every 50 iterations
     # Return avg losses
   
   def validate(self):
     # Run evaluation on val set
     # Return mAP metrics
   
   def fit(self, num_epochs):
     # Linear warmup for first 3 epochs (lr: 0 → base_lr)
     # Then cosine annealing
     # Save best model by val mAP
     # Early stopping patience: 5 epochs
     # Print epoch summary: loss, lr, mAP, time

Create scripts/train.py:
1. Parse config YAML
2. Create dataset and dataloaders (num_workers=2, pin_memory=True)
3. Create model, move to GPU
4. Create trainer and run training
5. Save final model + best model to weights/
6. Save training curves (loss, lr, mAP vs epoch) to results/training/
7. Print total training time and final metrics

IMPORTANT training settings for 5-hour constraint:
- 15-20 epochs maximum
- Batch size 8-12 (whatever fits in GPU memory with AMP)
- Use 3000-5000 training images (subset if needed)
- Mixed precision training is MANDATORY for speed
- DataLoader: num_workers=2, pin_memory=True, persistent_workers=True
- Log every 20 batches
- Validate every 2 epochs (not every epoch — too slow)

Commit: "feat: training pipeline with focal loss, AMP, and cosine annealing"
```

### Sub-Agent 7: Evaluation + Deployment Infrastructure
```
Create src/evaluation/evaluator.py:
1. COCOEvaluator class:
   - Convert model outputs to COCO detection format
   - Use pycocotools.COCOeval for standard metrics
   - Extend max_detections to [1, 10, 100, 300] for dense scenes
   - Compute: AP@0.5, AP@0.75, AP@[.5:.95], AR@1, AR@10, AR@100, AR@300
   - Per-density analysis: split images by object count into buckets (1-50, 51-100, 101-150, 150+)
   
2. Visualization functions:
   - draw_detections(image, boxes, scores, masks) — overlay predictions on image
   - plot_precision_recall_curve(eval_results) — PR curve at multiple IoU thresholds
   - create_comparison_grid(images, gt_boxes, pred_boxes) — side-by-side GT vs prediction
   - failure_analysis(model, dataset, num_images=20) — find worst predictions, visualize

Create scripts/evaluate.py:
1. Load trained model
2. Run evaluation on full val set
3. Generate results:
   - results/eval/metrics.json — all mAP numbers
   - results/eval/detection_samples.png — 8 sample detections
   - results/eval/precision_recall.png — PR curves
   - results/eval/failure_cases.png — worst 4 predictions with analysis
   - results/eval/softnms_comparison.png — same images with hard NMS vs Soft-NMS
   - results/eval/density_analysis.png — mAP breakdown by density bucket
4. Print results table

Create src/deployment/export_onnx.py:
1. Load PyTorch model
2. Create dummy input (1, 3, 550, 550)
3. Export with torch.onnx.export():
   - opset_version=11
   - dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
   - do_constant_folding=True
4. Verify: load with onnx, run onnx.checker.check_model
5. Save to weights/yolact_mobilenetv3.onnx
6. If ONNX export of full model fails (common with complex post-processing):
   - Export backbone+FPN+heads only (without NMS)
   - Handle NMS in Python post-processing
   - This is the standard practice

Create src/deployment/quantize.py:
1. Dynamic quantization: onnxruntime.quantization.quantize_dynamic
   - Weight type: QUInt8
   - Save to weights/yolact_mobilenetv3_int8.onnx
2. Print size comparison: FP32 vs INT8

Create src/deployment/benchmark.py:
1. Run inference with:
   - PyTorch FP32 (GPU if available, else CPU)
   - ONNX Runtime FP32 (CPUExecutionProvider)
   - ONNX Runtime INT8 (CPUExecutionProvider)
2. For each: warmup 10 runs, then time 50 runs
3. Report: avg latency (ms), FPS, model size (MB)
4. Save to results/deployment/benchmark.json and benchmark_table.png

Create scripts/export.py — orchestrates export → quantize → benchmark

Commit: "feat: evaluation suite with COCO metrics, ONNX export, INT8 quantization, and benchmarking"
```

---

## PHASE 3 — TRAINING + RESULTS (2:00–4:00)

This phase is mostly waiting for training. While GPU trains, run sub-agents for:

### Sub-Agent 8: Notebooks (runs while training)
```
Create 4 Jupyter notebooks in notebooks/:

01_EDA_and_Data_Analysis.ipynb:
- Import and display all EDA plots from results/eda/
- Add markdown cells explaining each finding
- Include code cells that reproduce key statistics
- Narrative: "The SKU-110K dataset presents [X] images with average [Y] objects per image..."
- Conclude with: "These findings motivate our choice of [custom anchors, Soft-NMS, focal loss]"

02_Classic_ML_Baseline.ipynb:
- Load and demonstrate HOG feature extraction on a sample image
- Show positive vs negative training patches
- Display HOG visualization using skimage.feature.hog(visualize=True)
- Run detection on 5 sample images, display results
- Show metrics and explain why classic ML fails on dense scenes:
  "HOG features are identical for different products → cannot discriminate"
  "Standard NMS with 147 overlapping detections → catastrophic suppression"
  "Sliding window O(scales × positions) → too slow for practical use"

03_DL_Training_and_Evaluation.ipynb:
- Show model architecture summary (torchsummary or print)
- Display training curves from saved results
- Show detection results on val set
- Ablation: Soft-NMS vs Hard NMS comparison
- Density analysis: performance at different object counts
- Failure case analysis with explanations

04_ONNX_Deployment.ipynb:
- Show ONNX export process
- Display model size comparison table
- Show benchmark results
- Demo: run inference with ONNX Runtime on sample images
- Compare inference speed: PyTorch vs ONNX FP32 vs ONNX INT8

Each notebook should be CLEAN, with markdown headers, explanations, and code cells that produce output.

Commit: "feat: comprehensive Jupyter notebooks for reproducibility"
```

### Sub-Agent 9: LaTeX Report (runs while training)
```
Create report/main.tex — Full LaTeX report in NeurIPS or IEEE format (8-12 pages):

Use article class with reasonable formatting if NeurIPS template not available.

Structure:
\title{High-Density Object Segmentation: Efficient Instance Segmentation for Densely Packed Retail Scenes using YOLACT with MobileNetV3 Backbone and Soft-NMS}

\begin{abstract}
~150 words. Problem → approach → key result → conclusion.
Dense retail environments with ~147 objects/image overwhelm standard detectors.
We combine YOLACT's prototype-based segmentation with MobileNetV3's efficient backbone and Soft-NMS post-processing.
On SKU-110K, our approach achieves [X]% mAP@0.5 at [Y] FPS, with [Z] MB INT8 model size.
\end{abstract}

1. Introduction (1.5 pages)
   - Problem: dense scenes, occlusion, edge deployment
   - Motivation: retail analytics, inventory management
   - Research gap: no combined YOLACT+MobileNetV3+SoftNMS
   - Contributions: (1) novel architecture combination, (2) density-aware training, (3) edge deployment pipeline

2. Related Work (1.5 pages)
   - Instance segmentation: Mask R-CNN → YOLACT → YOLACT++
   - Lightweight architectures: MobileNetV1/V2/V3, EfficientNet
   - Dense detection: SKU-110K, Soft-NMS, crowd counting
   - Edge deployment: ONNX, TensorRT, quantization

3. Methodology (2.5 pages)
   3.1 Dataset and Preprocessing
   3.2 MobileNetV3 Backbone Integration
   3.3 YOLACT Architecture with Modifications
   3.4 Soft-NMS for Dense Post-Processing
   3.5 Training Strategy (focal loss, AMP, cosine annealing)
   3.6 Edge Deployment Pipeline

4. Experiments (2 pages)
   4.1 Experimental Setup
   4.2 Classic ML Baseline Results
   4.3 Deep Learning Results
   4.4 Ablation Study (with/without Soft-NMS, anchor optimization)
   4.5 Edge Deployment Results

5. Discussion (0.5 pages)
   - Failure analysis with specific examples
   - Limitations: pseudo-masks, training time constraints
   - Comparison with prior work on SKU-110K

6. Conclusion (0.5 pages)
   - Summary, practical impact, future work

References: 15-20 entries including:
Goldman et al. 2019 (SKU-110K), Bolya et al. 2019 (YOLACT), Bolya et al. 2020 (YOLACT++),
Howard et al. 2019 (MobileNetV3), Bodla et al. 2017 (Soft-NMS), He et al. 2017 (Mask R-CNN),
Lin et al. 2017 (FPN), Lin et al. 2017 (Focal Loss/RetinaNet), Hu et al. 2018 (SE-Net),
Sandler et al. 2018 (MobileNetV2), Ren et al. 2015 (Faster R-CNN), Liu et al. 2016 (SSD),
Redmon et al. 2016 (YOLO), Dalal & Triggs 2005 (HOG)

PLACEHOLDER for results — use [PLACEHOLDER] where actual numbers go. These will be filled after training.

Create report/references.bib with all citations in BibTeX format.

Commit: "feat: LaTeX report draft with full structure and references"
```

---

## PHASE 4 — FINALIZATION (4:00–5:00) — Run 3 sub-agents in parallel

### Sub-Agent 10: Fill in Results + Final Report
```
After training completes:
1. Read all metrics from results/ folder
2. Update report/main.tex — replace ALL [PLACEHOLDER] with actual numbers
3. Update all notebooks with actual results
4. Generate final comparison table:
   | Method | mAP@0.5 | mAP@[.5:.95] | FPS | Size |
   | HOG+SVM | X% | X% | X | N/A |
   | YOLACT+MobileNetV3 (Hard NMS) | X% | X% | X | X MB |
   | YOLACT+MobileNetV3 (Soft-NMS) | X% | X% | X | X MB |
   | ONNX FP32 | X% | X% | X | X MB |
   | ONNX INT8 | X% | X% | X | X MB |

Commit: "feat: final results and metrics"
```

### Sub-Agent 11: Professional README
```
Create README.md with:

# High-Density Object Segmentation
### YOLACT + MobileNetV3 + Soft-NMS for Dense Retail Scene Understanding

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Problem
Retail environments contain 100-200+ products per shelf image. Standard NMS discards overlapping detections.

## Our Approach  
[Architecture diagram or description]
- **Backbone**: MobileNetV3-Large (5.4M params, ImageNet pretrained)
- **Segmentation**: YOLACT with 32 prototype masks
- **Post-processing**: Gaussian Soft-NMS (σ=0.5)
- **Deployment**: ONNX Runtime with INT8 quantization

## Key Results
| Metric | Value |
|--------|-------|
| mAP@0.5 | XX% |
| FPS (ONNX INT8) | XX |
| Model Size (INT8) | XX MB |

## Quick Start
```bash
make install
make download-data
make train
make evaluate
make export
```

## Project Structure
[Tree diagram]

## Dataset
SKU-110K: 11,762 images, ~1.73M annotations, avg 147 objects/image

## Methodology
[Brief explanation with architecture diagram reference]

## Results
[Tables and figure references]

## Edge Deployment
[ONNX pipeline description]

## References
[Key papers]

## Author
[Name, University, Course]

Commit: "docs: professional README with results and usage instructions"
```

### Sub-Agent 12: Git History Cleanup + Final Polish
```
1. Ensure all results/ files are committed
2. Run any remaining tests
3. Verify README renders correctly
4. Check all file imports work: python -c "from src.models.yolact import YOLACT; print('OK')"
5. Verify requirements.txt has all needed packages
6. Make final commit: "docs: final polish and cleanup"
7. Print final git log --oneline showing clean development history
```

---

## CRITICAL IMPLEMENTATION NOTES

### If MobileNetV3 backbone integration is too complex (>30 min):
**FALLBACK**: Use ResNet-50 backbone (already supported in YOLACT). The Soft-NMS contribution alone justifies the project. Focus time on:
- Clean implementation of Soft-NMS (the main innovation per instructor's slide)
- Thorough evaluation and ablation
- Professional report and repository

### If full YOLACT training is too slow:
**FALLBACK**: Use YOLOv8-seg from Ultralytics (3 lines to train on SKU-110K):
```python
from ultralytics import YOLO
model = YOLO('yolov8n-seg.pt')
results = model.train(data='SKU-110K.yaml', epochs=20, imgsz=640)
```
Then add Soft-NMS as custom post-processing. This trains MUCH faster and the focus shifts to the Soft-NMS analysis.

### If ONNX export fails:
Export only the backbone+heads (without NMS/mask assembly). Handle post-processing in NumPy. This is standard practice.

### Priority order if running out of time:
1. Working DL model with evaluation results (MUST HAVE)
2. Classic ML baseline with comparison (MUST HAVE)  
3. Professional README and repo structure (MUST HAVE)
4. EDA visualizations (HIGH)
5. LaTeX report (HIGH)
6. ONNX deployment (MEDIUM)
7. Notebooks (MEDIUM)
8. Unit tests (LOW — skip if needed)

### Git commit strategy:
Make commits in this order with these messages:
1. "feat: project scaffolding and configuration"
2. "feat: data pipeline with SKU-110K loader and augmentations"
3. "feat: YOLACT architecture with MobileNetV3 backbone"
4. "feat: Soft-NMS implementation"
5. "feat: training pipeline with focal loss and AMP"
6. "feat: EDA and data analysis"
7. "feat: HOG+SVM classic ML baseline"
8. "feat: evaluation suite with COCO metrics"
9. "feat: ONNX export and INT8 quantization"
10. "feat: LaTeX report draft"
11. "feat: Jupyter notebooks"
12. "feat: final results and metrics"
13. "docs: professional README"
14. "docs: final polish and cleanup"

Each commit should be a WORKING state — no broken imports, no syntax errors.

---

## REMEMBER
- Read RESEARCH_REPORT.md for all technical details, paper references, mathematical formulations
- Use Task tool to spawn sub-agents for parallel work
- GPU training takes 1.5-2 hours — use that time for CPU tasks (baseline, EDA, report, notebooks)
- Mixed precision (AMP) is MANDATORY — saves 30-40% training time
- If something breaks, use the FALLBACK option and move on. Don't waste >15 min debugging any single component
- The GOAL is a complete, professional project that scores 9-10/10 on the rubric, not perfect code