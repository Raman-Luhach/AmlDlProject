# High-density object segmentation: a complete B.Tech project blueprint

**YOLACT + MobileNetV3 + Soft-NMS on SKU-110K can be executed in 6 hours and score 9–10/10 across all rubric criteria, but requires precise planning.** The core pipeline—a lightweight MobileNetV3 backbone feeding YOLACT's prototype-based instance segmentation with Soft-NMS post-processing—addresses a genuine research gap: no published work combines these three components for dense retail detection. SKU-110K's extreme density (~147 objects/image) makes it the ideal proving ground, and the single-class annotation format simplifies training while amplifying the localization challenge. The critical constraint is that **SKU-110K provides only bounding boxes, not segmentation masks**, so pseudo-masks or box-cropped mask supervision must be used for the YOLACT mask branch.

---

## 1. SKU-110K: the dataset that breaks standard detectors

SKU-110K was introduced at CVPR 2019 by Goldman et al. in "Precise Detection in Densely Packed Scenes." It contains **11,762 images** of retail shelves from supermarkets across the United States, Europe, and East Asia, captured by smartphone cameras. The dataset holds **~1.73 million bounding box annotations** across a single class ("object"), averaging **~147 annotated products per image**—roughly 20× denser than MS COCO's average of ~7 objects per image.

**Data splits and format.** The official split is **8,233 train / 588 validation / 2,941 test** images. Annotations use a **custom CSV format** (not COCO JSON or Pascal VOC XML), with columns for `image_name, x1, y1, x2, y2, class, image_width, image_height`. Each row represents one bounding box; images with 147 objects have 147 rows. Conversion scripts to YOLO format exist in Ultralytics' `SKU-110K.yaml`, and COCO-format conversion is available via FiftyOne or custom scripts. The download is **~13.6 GB** from `http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz`, with mirrors on Google Drive, Kaggle, and Hugging Face. The license restricts use to **academic and non-commercial purposes**.

**Known issues demand careful preprocessing.** Some images are corrupt and fail to load with PIL/OpenCV—iterate and filter before training. Price tags and promotional materials are occasionally annotated as products. Bounding box boundaries in extremely dense regions suffer from annotator disagreement. For a 6-hour project, critical data cleaning includes: (1) corrupt image removal via `PIL.Image.open()` try/catch, (2) zero-area box check (`x1 == x2` or `y1 == y2`), (3) out-of-bounds coordinate validation, and (4) duplicate annotation removal.

**EDA strategies that earn 10/10.** Five essential visualizations: a histogram of objects-per-image (expect a distribution centered around 147 with a right tail), a scatter plot of normalized bounding box width vs. height revealing dominant aspect ratios (~0.3–0.7 width/height for vertical products), a pairwise IoU histogram between neighboring ground-truth boxes (expect many pairs with IoU > 0 but < 0.3, quantifying the density challenge), an aspect ratio distribution informing anchor design, and a box area distribution on log scale showing the preponderance of small objects. Run K-means on ground-truth box dimensions (using `1 - IoU` as distance) to derive custom anchors—this alone can boost mAP by **2–5%** over default COCO anchors.

**For a 6-hour constraint, use a 3,000–5,000 image training subset** while keeping the full 588-image validation set for proper evaluation. The full dataset requires ~20 minutes to download and another 15 minutes for format conversion.

---

## 2. Literature review structured as a publishable introduction

The literature review should be organized thematically, not chronologically, progressing from broad to narrow: instance segmentation landscape → real-time architectures → dense scene challenges → retail-specific detection → research gap.

**Instance segmentation evolution.** Mask R-CNN (He et al., ICCV 2017) extended Faster R-CNN with a parallel mask branch using RoIAlign, achieving **37.1 mask mAP on COCO at ~13.5 FPS**—accurate but too slow for edge deployment. YOLACT (Bolya et al., ICCV 2019) broke the real-time barrier by decomposing instance segmentation into two parallel subtasks: generating **k = 32 prototype masks** via a Protonet FCN and predicting per-instance mask coefficients. The final mask is assembled as **M = σ(P · cᵀ)**, a single matrix multiplication taking ~5ms. YOLACT achieved **29.8 mAP at 33.5 FPS** on a Titan Xp. YOLACT++ (Bolya et al., TPAMI 2020) added deformable convolutions, optimized anchors, and a fast mask re-scoring branch, reaching **34.1 mAP** with negligible overhead.

**Lightweight backbone architectures.** MobileNetV3 (Howard et al., ICCV 2019) combined hardware-aware NAS with NetAdapt, producing two variants: **MobileNetV3-Large (5.4M params, 219M MAdds, 75.2% ImageNet top-1)** and **MobileNetV3-Small (2.5M params, 66M MAdds, 67.4% top-1)**. Key innovations include inverted residual blocks with depthwise separable convolutions (reducing computation by ~8× versus standard convolutions), squeeze-and-excitation channel attention, and h-swish activation (`h-swish(x) = x · ReLU6(x+3)/6`). For detection with SSDLite, MobileNetV3-Large is **27% faster** than MobileNetV2 with near-identical mAP.

**Dense detection and Soft-NMS.** Standard NMS with a hard IoU threshold catastrophically fails in dense scenes—adjacent products with overlapping boxes are suppressed as duplicates. Bodla et al. (ICCV 2017) proposed Soft-NMS, which decays detection scores as a continuous function of overlap rather than eliminating them. The Gaussian formulation, **sᵢ = sᵢ · exp(−IoU²/σ)**, requires no threshold parameter and improved mAP by **+1.3% on COCO** with identical computational complexity to hard NMS. Goldman et al.'s SKU-110K paper demonstrated that even state-of-the-art RetinaNet struggles in dense retail scenes, proposing a Soft-IoU layer and EM-Merger unit as alternatives. Faster R-CNN achieved a surprisingly poor **~4.5% mAP** on SKU-110K, underscoring the severity of the density challenge.

**The research gap.** No published work combines YOLACT's real-time prototype-based segmentation with MobileNetV3's mobile-optimized backbone and Soft-NMS's overlap-preserving post-processing, evaluated on dense retail imagery. Each component exists independently, but their synergistic combination for edge-deployable retail shelf analysis is unexplored—a clear and defensible contribution.

**Citation targets for 10/10 quality:** aim for 15–20 references with 2–4 citations per paragraph in the body. Group related works: "Several approaches [A, B, C] have explored..." For key papers (YOLACT, Soft-NMS, MobileNetV3, SKU-110K), dedicate 1–2 paragraphs each.

---

## 3. The classic ML baseline proves why deep features matter

**HOG + Linear SVM with multi-scale sliding window** is the most defensible classic baseline for SKU-110K. It is historically significant (Dalal & Triggs, 2005), produces a clear quantifiable gap versus deep learning, and can be implemented in ~30 minutes using scikit-learn and OpenCV.

**HOG descriptor parameters for retail products:** window size **64×64** (adapted from 64×128 pedestrian default), cell size 8×8, block size 16×16, block stride 8×8, **9 orientation bins** (0°–180° unsigned), L2-Hys normalization. This produces a **feature vector of 1,764 dimensions** per window. Train a `LinearSVC(C=0.01)` on positive crops (products from ground-truth boxes) and negative patches (random background regions), then apply hard negative mining—re-run the detector on training images, collect false positives, and retrain.

**The implementation pipeline has six steps:** (1) extract positive/negative HOG features, (2) train Linear SVM, (3) hard negative mining and retrain, (4) multi-scale sliding window detection at scales [1.0, 0.75, 0.5, 0.35], (5) NMS on detections, (6) evaluate. Expected performance: **~5–12% mAP@0.5** versus ~37–50% for deep learning methods. This dramatic gap provides four key insights for the report: HOG features look nearly identical for different products (proving hand-crafted features cannot discriminate similar objects), standard NMS completely breaks with 147 overlapping detections (motivating Soft-NMS), the sliding window's exhaustive search is paradoxically slower overall despite being faster per-window (motivating anchor-based methods), and scale sensitivity demonstrates why FPN is essential.

Other classic approaches—watershed segmentation, connected components, DPM, template matching—are less suitable. Watershed over-segments textured products, connected components fails on complex retail scenes, DPM is too slow (~0.07 Hz) and complex to implement in 30 minutes, and template matching is infeasible with 110K+ product variations.

---

## 4. YOLACT with MobileNetV3: architecture and training recipe

**YOLACT's two-branch design** is the key to real-time instance segmentation. The Protonet branch takes the largest FPN feature map (P3, at 1/4 input resolution) and produces **k = 32 prototype masks** through sequential 3×3 convolutions with one bilinear upsample. The detection branch adds a parallel coefficient head to the standard SSD/RetinaNet prediction head, outputting **4 + c + k** values per anchor (box offsets, class scores, mask coefficients with tanh activation). Mask assembly is a single matrix multiply followed by sigmoid: **M = σ(P · cᵀ)**, then cropped by the predicted bounding box. The entire mask branch adds ~5ms overhead.

**Integrating MobileNetV3-Large as backbone** requires mapping its inverted residual blocks to FPN tap points equivalent to ResNet's C3/C4/C5. The recommended mapping: layers 0–3 (stride 4, ~24 channels), layers 4–6 → **C3 equivalent** (stride 8, ~40 channels), layers 7–12 → **C4 equivalent** (stride 16, ~112 channels), layers 13–16 → **C5 equivalent** (stride 32, ~960 channels). Add lateral connections with 256-channel 1×1 convolutions for FPN, plus 2 downsample layers for 5 total prediction heads. **No existing public YOLACT + MobileNetV3 implementation exists**—this requires custom integration, but the YOLACT codebase (`dbolya/yolact`, 4.7k GitHub stars) has a modular `config.py` backbone system that simplifies this. Use `torchvision.models.mobilenet_v3_large(weights='IMAGENET1K_V2')` for pretrained weights. MobileNetV3-Large as YOLACT backbone yields an estimated **15–18M total parameters** (~60–70 MB FP32) versus 31M for ResNet-50 and 50M for ResNet-101 variants.

**Soft-NMS replaces YOLACT's Fast NMS** in the post-processing step. The Gaussian decay function **sᵢ = sᵢ · exp(−IoU(M, bᵢ)²/σ)** with **σ = 0.5** is the recommended default. At IoU = 0.5, the decay factor is exp(−0.5) ≈ 0.607 (score reduced by ~40%); at IoU = 0.9, it drops to exp(−1.62) ≈ 0.198 (score reduced by ~80%). For SKU-110K's dense scenes, expect **+2–5% mAP improvement** over hard NMS, particularly at higher IoU thresholds (AP75). Use a score pruning threshold of 0.001 to remove negligible detections. The `DocF/Soft-NMS` GitHub repo provides both CPU and GPU PyTorch implementations.

**Training hyperparameters for the 6-hour constraint.** Optimizer: SGD with momentum 0.9, weight decay 5×10⁻⁴. Initial learning rate: **1×10⁻³** with 3-epoch linear warmup followed by cosine annealing to 1×10⁻⁶. Batch size: **8–16** (MobileNetV3's smaller footprint enables larger batches). Input resolution: **550×550** (YOLACT default). Loss weights: **L_cls = 1.0, L_box = 1.5, L_mask = 6.125** (YOLACT defaults). Replace standard cross-entropy with **focal loss (α = 0.25, γ = 2.0)** for the classification head to handle extreme foreground-background imbalance. Data augmentation: random horizontal flip, SSD-style random crop, color jitter (HSV: h=0.015, s=0.7, v=0.4), and photometric distortion. Mixed precision training via `torch.cuda.amp` provides **~1.5–2× speedup** on T4 GPUs with ~30–40% memory reduction.

**Critical configuration for dense scenes**: increase max detections from the default 200 to **500+**, reduce confidence threshold to 0.05 during evaluation, and use the K-means-derived custom anchors from the EDA stage.

---

## 5. Evaluation metrics and expected performance

The primary metric is **COCO-style mAP (AP@[.50:.95])**, averaging AP across 10 IoU thresholds from 0.5 to 0.95. For dense retail detection, **AP75** (strict localization) and **AR@300** (average recall with up to 300 detections per image) are especially critical—they directly measure the system's ability to precisely localize closely packed products without missing any. Compute metrics using `pycocotools.COCOeval` or `torchmetrics.detection.MeanAveragePrecision`, extending `max_detection_thresholds` to `[1, 10, 100, 300]` for SKU-110K's density.

Expected performance estimates:

| Method | mAP@0.5 | mAP@[.50:.95] | FPS | Model Size |
|--------|---------|---------------|-----|------------|
| HOG + SVM baseline | ~5–12% | ~2–5% | <1 | N/A |
| YOLACT + MobileNetV3 (ours) | ~35–45% | ~18–25% | 40–60 | ~65 MB |
| YOLACT + MobileNetV3 + Soft-NMS | ~38–48% | ~20–28% | 35–55 | ~65 MB |
| RetinaNet (Goldman et al. baseline) | ~37.7% | — | ~15 | ~145 MB |
| SKU-110K paper method | ~49%+ | — | ~12 | ~200 MB |

---

## 6. Feature engineering as the project's breakthrough component

Feature engineering in computer vision spans three tiers: data-level (augmentation, preprocessing), architecture-level (anchor design, FPN modifications), and representation-level (learned feature analysis). To score 10/10, chain these into a coherent narrative driven by EDA.

**Tier 1: EDA-driven anchor optimization.** Extract all ground-truth box widths and heights, normalize by image dimensions, and run K-means with `1 - IoU` as the distance metric. Test k = 5, 7, 9 clusters and select via elbow plot (mean IoU vs. k). SKU-110K products cluster around small, vertically oriented rectangles (aspect ratios ~0.5–1.0). Visualize the coverage improvement: overlay default COCO anchors versus custom anchors on sample images, showing **IoU increase from ~0.55 to ~0.72** with optimized anchors. This is a concrete, data-driven contribution.

**Tier 2: Density-aware augmentation.** Design occlusion-simulation augmentation: randomly paste product crops over other products to train robustness to partial occlusion. Add shelf-specific random crops that maintain row structure. Normalize for fluorescent retail lighting with aggressive brightness/contrast/gamma augmentation. These are motivated by the specific failure modes revealed in EDA.

**Tier 3: Density heatmap as auxiliary signal.** Generate Gaussian density maps (kernel at each ground-truth center) and train a parallel head to predict object density. Use the predicted density map as an attention mask on FPN features—regions with predicted high density receive stronger feature emphasis. This directly addresses the core challenge and has support from density-guided anchor (DGA) literature.

**Visualization suite for proving impact:** PCA/t-SNE plots of backbone features (extract from P3/P4 via forward hooks, pool to fixed size, reduce with PCA to 50 dims, then t-SNE to 2D) colored by object density; Grad-CAM or EigenCAM activation maps on FPN levels showing where the model attends in dense vs. sparse regions; side-by-side HOG features vs. deep features for similar products proving the representation gap.

---

## 7. Mathematical foundations for theoretical rigor

**Instance segmentation as multi-task optimization.** The total loss minimized during YOLACT training is:

**L_total = λ₁L_cls + λ₂L_box + λ₃L_mask**

where L_cls is focal loss (for class imbalance): **FL(pₜ) = −αₜ(1 − pₜ)^γ log(pₜ)** with α = 0.25, γ = 2; L_box is Smooth L1 loss over parameterized box offsets (tₓ, tᵧ, t_w, t_h); and L_mask is pixel-wise binary cross-entropy between assembled mask M = σ(P · cᵀ) and ground truth G: **L_mask = −(1/n) Σᵢⱼ[Gᵢⱼ log(Mᵢⱼ) + (1−Gᵢⱼ) log(1−Mᵢⱼ)]**. Mask loss is divided by ground-truth bounding box area to preserve small object gradients.

**Soft-NMS derivation.** Traditional NMS applies a step function f(IoU) that sets overlapping scores to zero—discontinuous and destructive in dense scenes. Soft-NMS replaces this with a continuous Gaussian decay: **f(IoU) = exp(−IoU²/σ)**. Analysis of σ: as σ → 0, the function approaches hard NMS; as σ → ∞, it approaches no suppression. At the empirically optimal σ = 0.5, a detection overlapping the top-scoring box with IoU = 0.5 retains 60.7% of its score, while one with IoU = 0.9 retains only 19.8%. The algorithm's O(N²) complexity is identical to standard NMS, and it requires **no retraining**—it is a pure drop-in replacement during inference.

**Depthwise separable convolution complexity.** Standard convolution costs D_K² · M · N · D_F² operations. Depthwise separable convolution decomposes this into depthwise (D_K² · M · D_F²) plus pointwise (M · N · D_F²), yielding a reduction ratio of **1/N + 1/D_K²**. For a 3×3 kernel with 64 output channels, this is ~8× fewer operations—the foundation of MobileNetV3's efficiency.

**Squeeze-and-excitation channel attention** in MobileNetV3 computes channel descriptors via global average pooling: **zc = (1/HW) Σᵢⱼ uc(i,j)**, then applies a bottleneck excitation: **s = σ(W₂ · δ(W₁ · z))** with reduction ratio r = 4, and rescales features: **x̃c = sc · uc**. Overhead is negligible (~0.26% FLOP increase in SE-ResNet-50) but provides consistent accuracy gains through learned channel importance.

**Bias-variance tradeoff in dense detection.** More anchors per location increase recall (capturing more densely packed objects) but quadratically increase false positives and pairwise overlap computations in NMS. With SKU-110K's ~147 objects per image and potentially 100K+ anchors, the positive-negative ratio is ~1:700. This motivates focal loss (down-weights easy background anchors by factor (1−pₜ)² ≈ 0.01 for well-classified examples) during training and Soft-NMS (preserves true positives among overlapping detections) during inference.

---

## 8. ONNX deployment pipeline for edge inference

**Exporting YOLACT to ONNX** requires opset version 11+ and careful handling of dynamic shapes. The recommended approach: export the backbone + FPN + prediction heads as one graph, handling NMS/post-processing in Python. Pre-compute prior boxes as constants to avoid tracing issues with dynamic loops. Use `torch.onnx.export()` with `dynamic_axes` for variable batch size. PINTO0309's `yolact_edge_onnx_tensorrt_myriad` GitHub repo provides a proven conversion pipeline with benchmarks showing **4.5ms average inference** with TensorRT execution provider.

**ONNX Runtime optimization** operates at three levels. Graph optimization (constant folding, operator fusion) via `ORT_ENABLE_ALL` provides ~1.5–2× speedup over PyTorch eager mode. Dynamic quantization (`onnxruntime.quantization.quantize_dynamic` with QUInt8 weights) requires no calibration data and reduces model size by ~4×. Static INT8 quantization requires a calibration dataset (50–100 representative images) but delivers the best results—**~2–3× additional speedup** on CPU. Combined, expect **~3–5× total speedup** over PyTorch and a model size reduction from **~65 MB (FP32) to ~16 MB (INT8)**.

**Edge simulation** when no physical device is available: configure ONNX Runtime with `CPUExecutionProvider`, limit threads (`intra_op_num_threads=2`), and test at reduced resolution (320×320 or 416×416). Benchmark by averaging 100 inference runs after 10 warmup iterations using `time.perf_counter()`. Report latency (ms), FPS, and model size in a comparison table: PyTorch FP32 → ONNX FP32 → ONNX INT8.

---

## 9. GitHub repository structure and professional presentation

The ideal folder structure separates concerns cleanly:

```
yolact-mobilenetv3-sku110k/
├── configs/           # YAML hyperparameter files
├── src/
│   ├── models/        # backbone.py, yolact.py, protonet.py, detection.py
│   ├── data/          # dataset.py, augmentations.py
│   ├── training/      # trainer.py, losses.py
│   ├── evaluation/    # evaluator.py, metrics.py
│   ├── deployment/    # export_onnx.py, quantize.py, benchmark.py
│   └── utils/         # soft_nms.py, visualization.py
├── notebooks/         # 01_EDA → 02_baseline → 03_training → 04_eval → 05_deploy
├── scripts/           # Shell scripts for download, train, export
├── tests/             # Unit tests
├── report/            # LaTeX source + figures
├── Makefile           # install, data, train, eval, export, test targets
├── requirements.txt   # ~20 packages: torch, torchvision, onnxruntime, etc.
└── README.md          # Badges, overview, architecture diagram, results table
```

**Git commit strategy for "steady development"**: use descriptive conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`) in a logical progression from project setup → backbone integration → architecture implementation → training → evaluation → deployment → documentation. Plan ~15–20 commits across the 6-hour window, with timestamps showing consistent development.

**LaTeX report** should follow NeurIPS or IEEE conference format at **8–12 pages**. Sections: Abstract, Introduction, Related Work, Methodology (subsections for architecture, loss functions, Soft-NMS, ONNX pipeline), Experimental Setup, Results (with ablation: e.g., with/without Soft-NMS), Discussion, Conclusion. Key differentiators for 9–10: an architecture diagram (even a simple box diagram), a proper ablation study table, deployment analysis with quantitative speed/size comparisons, error analysis with failure case visualizations, and a reproducibility section linking to the GitHub repo. Use `booktabs` for professional tables and `minted` or `listings` for code snippets. Target **15–20 references**.

---

## 10. Six-hour execution timeline with critical shortcuts

The following timeline is calibrated for Google Colab T4 GPU with 16 GB VRAM:

| Phase | Time | Task |
|-------|------|------|
| 0:00–0:10 | 10 min | Environment setup, package installation, clone repos |
| 0:10–0:30 | 20 min | Download SKU-110K subset (3K–5K images), convert CSV → COCO JSON |
| 0:30–1:00 | 30 min | EDA: distribution plots, anchor analysis, data cleaning |
| 1:00–1:30 | 30 min | Classic ML baseline (HOG + SVM on 500 images subset) |
| 1:30–2:00 | 30 min | Model architecture setup: MobileNetV3 backbone + YOLACT + Soft-NMS |
| 2:00–4:00 | 120 min | DL training: 15–20 epochs with mixed precision, cosine annealing |
| 4:00–4:20 | 20 min | Evaluation: mAP computation, qualitative visualization |
| 4:20–4:40 | 20 min | ONNX export, INT8 quantization, speed benchmarking |
| 4:40–5:30 | 50 min | LaTeX report writing using prepared template |
| 5:30–6:00 | 30 min | Git cleanup, README polish, final commit |

**Critical shortcuts that save hours.** First, **start from pre-trained YOLACT COCO weights** (available from dbolya/yolact HuggingFace collection) and fine-tune rather than training from scratch—this cuts convergence time from 100+ epochs to 15–20. If MobileNetV3 backbone integration proves too complex in 30 minutes, fall back to ResNet-50 backbone with pre-trained YOLACT weights and focus on the Soft-NMS contribution instead. Second, **use 3,000–5,000 training images** rather than the full 8,233—sufficient for transfer learning with minimal quality loss. Third, run the classic ML baseline on CPU **in parallel** with GPU training. Fourth, use `torch.cuda.amp` for mixed precision—T4 has Tensor Cores supporting FP16, yielding ~1.5–2× speedup and ~30–40% memory reduction. Fifth, use `DataLoader` with `num_workers=2` (Colab's 2 CPU cores), `pin_memory=True`, and pre-resized images.

**The single highest-risk item** is MobileNetV3 backbone integration into YOLACT's architecture. Mitigate by preparing the backbone wrapper class (mapping inverted residual blocks to C3/C4/C5 FPN tap points) in advance and testing with a single forward pass before starting training. If integration fails, `Yolact_minimal` (GitHub: feiyuhuahuo/Yolact_minimal) supports custom backbones more easily than the original repo, and YolactEdge provides TensorRT-optimized variants with MobileNetV2 support that can be adapted.

---

## Conclusion: a novel combination targeting a real gap

This project sits at the intersection of three well-studied but never-combined components: YOLACT's efficient prototype-based segmentation, MobileNetV3's mobile-optimized backbone, and Soft-NMS's overlap-preserving post-processing. The novelty is not in any single component but in their integration for **edge-deployable, real-time instance segmentation of densely packed retail scenes**—a practical industrial need with no existing solution in the literature. The classic ML baseline (HOG + SVM achieving ~5–12% mAP) quantifies why learned representations are necessary. The deep learning pipeline (targeting ~38–48% mAP@0.5 at 35–55 FPS) demonstrates practical viability. The ONNX deployment path (INT8 quantized to ~16 MB at ~3–5× speedup) proves edge readiness. The feature engineering chain—from K-means anchor optimization through density-aware augmentation to density heatmap attention—provides a coherent, EDA-driven narrative that transforms dataset analysis into architectural decisions. Every rubric criterion maps directly to a concrete deliverable within the 6-hour window: the literature review frames the gap, the EDA justifies design choices, the feature engineering provides the breakthrough, the theoretical foundations supply mathematical rigor, and the deployment pipeline demonstrates real-world applicability.