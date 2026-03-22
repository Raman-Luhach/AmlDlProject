"""
HOG + Linear SVM baseline for dense object detection.

This module implements a classic computer vision pipeline using Histogram of
Oriented Gradients (HOG) features with a Linear SVM classifier. The purpose
is to demonstrate that traditional CV methods fail on dense retail shelf
scenes (SKU-110K-style), achieving only ~5-12% mAP@0.5 compared to modern
deep learning detectors.

Dependencies: numpy, opencv-python, scikit-learn, scikit-image, matplotlib
"""

import time
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog


class HOGSVMBaseline:
    """HOG + Linear SVM baseline for dense object detection.

    This pipeline extracts HOG features from image patches and trains a
    Linear SVM to classify windows as object / background. Detection is
    performed via multi-scale sliding window with non-maximum suppression.

    Parameters
    ----------
    window_size : tuple of int
        (width, height) of the detection window in pixels.
    cell_size : tuple of int
        (width, height) of HOG cells in pixels.
    block_size : tuple of int
        (width, height) of HOG blocks in pixels.
    nbins : int
        Number of orientation bins for HOG.
    """

    def __init__(
        self,
        window_size=(64, 64),
        cell_size=(8, 8),
        block_size=(16, 16),
        nbins=9,
    ):
        self.window_size = window_size  # (w, h)
        self.cell_size = cell_size
        self.block_size = block_size
        self.nbins = nbins

        # Linear SVM with low C for regularisation (many noisy features)
        self.svm = LinearSVC(C=0.01, max_iter=10000, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_hog_features(self, image_patch, visualize=False):
        """Extract HOG feature vector from a single image patch.

        Parameters
        ----------
        image_patch : np.ndarray
            Grayscale or BGR image patch. Will be resized to
            ``self.window_size`` and converted to grayscale if needed.
        visualize : bool
            If True, also return the HOG visualisation image.

        Returns
        -------
        features : np.ndarray
            1-D HOG feature vector (~1764 dims for 64x64 / 8x8 / 16x16 / 9).
        hog_image : np.ndarray or None
            HOG visualisation image (only when *visualize=True*).
        """
        # Ensure correct size
        patch = cv2.resize(image_patch, self.window_size)

        # Convert to grayscale if colour
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        pixels_per_cell = self.cell_size
        cells_per_block = (
            self.block_size[0] // self.cell_size[0],
            self.block_size[1] // self.cell_size[1],
        )

        if visualize:
            features, hog_image = hog(
                patch,
                orientations=self.nbins,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm="L2-Hys",
                visualize=True,
                feature_vector=True,
            )
            return features, hog_image
        else:
            features = hog(
                patch,
                orientations=self.nbins,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm="L2-Hys",
                visualize=False,
                feature_vector=True,
            )
            return features, None

    # ------------------------------------------------------------------
    # Training data preparation
    # ------------------------------------------------------------------

    def prepare_training_data(
        self, images, annotations, num_pos=5000, num_neg=10000
    ):
        """Build training set of HOG feature vectors.

        Parameters
        ----------
        images : list of np.ndarray
            Training images (BGR).
        annotations : list of np.ndarray
            Per-image ground-truth boxes, each (N, 4) with columns
            [x1, y1, x2, y2].
        num_pos : int
            Maximum number of positive (object) patches to collect.
        num_neg : int
            Maximum number of negative (background) patches to collect.

        Returns
        -------
        X_train : np.ndarray, shape (n_samples, n_features)
        y_train : np.ndarray, shape (n_samples,)  -- 1 for positive, 0 for negative
        """
        rng = np.random.RandomState(42)
        pos_features = []
        neg_features = []

        # -- Positive patches from GT boxes --
        pos_per_image = max(1, num_pos // max(len(images), 1))
        for img, boxes in zip(images, annotations):
            if len(boxes) == 0:
                continue
            h_img, w_img = img.shape[:2]
            indices = rng.choice(len(boxes), size=min(pos_per_image, len(boxes)), replace=False)
            for idx in indices:
                x1, y1, x2, y2 = boxes[idx].astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                crop = img[y1:y2, x1:x2]
                feat, _ = self.extract_hog_features(crop)
                pos_features.append(feat)
                if len(pos_features) >= num_pos:
                    break
            if len(pos_features) >= num_pos:
                break

        # -- Negative patches (random background crops) --
        neg_per_image = max(1, num_neg // max(len(images), 1))
        for img, boxes in zip(images, annotations):
            h_img, w_img = img.shape[:2]
            attempts = 0
            collected = 0
            while collected < neg_per_image and attempts < neg_per_image * 5:
                attempts += 1
                pw = rng.randint(20, max(21, w_img // 3))
                ph = rng.randint(20, max(21, h_img // 3))
                px = rng.randint(0, max(1, w_img - pw))
                py = rng.randint(0, max(1, h_img - ph))

                # Check IoU with all GT boxes -- keep only if low overlap
                if len(boxes) > 0 and self._max_iou(
                    np.array([px, py, px + pw, py + ph]), boxes
                ) > 0.3:
                    continue

                crop = img[py : py + ph, px : px + pw]
                feat, _ = self.extract_hog_features(crop)
                neg_features.append(feat)
                collected += 1
                if len(neg_features) >= num_neg:
                    break
            if len(neg_features) >= num_neg:
                break

        pos_features = np.array(pos_features)
        neg_features = np.array(neg_features)

        X_train = np.vstack([pos_features, neg_features])
        y_train = np.concatenate(
            [np.ones(len(pos_features)), np.zeros(len(neg_features))]
        )

        # Shuffle
        perm = rng.permutation(len(y_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        return X_train, y_train

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_train, y_train):
        """Train the Linear SVM on HOG features.

        Parameters
        ----------
        X_train : np.ndarray
        y_train : np.ndarray

        Returns
        -------
        train_accuracy : float
        """
        print(f"  Training SVM on {len(y_train)} samples "
              f"({int(y_train.sum())} pos, {int(len(y_train) - y_train.sum())} neg) ...")
        X_scaled = self.scaler.fit_transform(X_train)
        self.svm.fit(X_scaled, y_train)
        self.is_trained = True
        acc = self.svm.score(X_scaled, y_train)
        print(f"  Training accuracy: {acc:.4f}")
        return acc

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def sliding_window(
        self, image, scales=(1.0, 0.75, 0.5, 0.35), step_size=16
    ):
        """Run multi-scale sliding window detection.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR).
        scales : tuple of float
            Image pyramid scales.
        step_size : int
            Stride in pixels at each scale.

        Returns
        -------
        detections : list of (x1, y1, x2, y2, score)
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")

        detections = []
        h_orig, w_orig = image.shape[:2]
        win_w, win_h = self.window_size

        for scale in scales:
            resized = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
            )
            h_r, w_r = resized.shape[:2]

            if h_r < win_h or w_r < win_w:
                continue

            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

            for y in range(0, h_r - win_h + 1, step_size):
                for x in range(0, w_r - win_w + 1, step_size):
                    patch = gray[y : y + win_h, x : x + win_w]
                    feat = hog(
                        patch,
                        orientations=self.nbins,
                        pixels_per_cell=self.cell_size,
                        cells_per_block=(
                            self.block_size[0] // self.cell_size[0],
                            self.block_size[1] // self.cell_size[1],
                        ),
                        block_norm="L2-Hys",
                        visualize=False,
                        feature_vector=True,
                    )
                    feat_scaled = self.scaler.transform(feat.reshape(1, -1))
                    score = self.svm.decision_function(feat_scaled)[0]

                    if score > 0:
                        # Map back to original image coordinates
                        x1 = int(x / scale)
                        y1 = int(y / scale)
                        x2 = int((x + win_w) / scale)
                        y2 = int((y + win_h) / scale)
                        detections.append((x1, y1, x2, y2, float(score)))

        return detections

    def detect(self, image, score_threshold=0.5, nms_threshold=0.3):
        """Detect objects in an image using sliding window + NMS.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR).
        score_threshold : float
            Minimum SVM decision score to keep a detection.
        nms_threshold : float
            IoU threshold for non-maximum suppression.

        Returns
        -------
        boxes : np.ndarray, shape (K, 4)
        scores : np.ndarray, shape (K,)
        """
        raw_dets = self.sliding_window(image)

        if len(raw_dets) == 0:
            return np.zeros((0, 4)), np.zeros(0)

        raw_dets = np.array(raw_dets)
        boxes = raw_dets[:, :4]
        scores = raw_dets[:, 4]

        # Filter by score threshold
        mask = scores >= score_threshold
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)

        # Non-maximum suppression
        keep = self._nms(boxes, scores, nms_threshold)
        return boxes[keep], scores[keep]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, images, annotations, num_images=50, score_threshold=0.3):
        """Compute mAP@0.5 on a set of images.

        Parameters
        ----------
        images : list of np.ndarray
        annotations : list of np.ndarray  -- each (N, 4) [x1, y1, x2, y2]
        num_images : int
            Max images to evaluate.
        score_threshold : float
            Minimum score for detections.

        Returns
        -------
        metrics : dict
            Keys: mAP, precision, recall, avg_time_per_image,
                  total_gt, total_det, per_image_ap
        """
        num_eval = min(num_images, len(images))
        all_ap = []
        total_gt = 0
        total_det = 0
        total_tp = 0
        times = []

        for i in range(num_eval):
            img = images[i]
            gt_boxes = annotations[i]
            total_gt += len(gt_boxes)

            t0 = time.time()
            det_boxes, det_scores = self.detect(img, score_threshold=score_threshold)
            times.append(time.time() - t0)
            total_det += len(det_boxes)

            ap, tp_count = self._compute_ap(det_boxes, det_scores, gt_boxes, iou_threshold=0.5)
            all_ap.append(ap)
            total_tp += tp_count

        mAP = float(np.mean(all_ap)) if all_ap else 0.0
        precision = total_tp / max(total_det, 1)
        recall = total_tp / max(total_gt, 1)
        avg_time = float(np.mean(times)) if times else 0.0

        metrics = {
            "mAP@0.5": round(mAP * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "avg_time_per_image_sec": round(avg_time, 3),
            "total_gt_boxes": int(total_gt),
            "total_detections": int(total_det),
            "total_true_positives": int(total_tp),
            "num_images_evaluated": num_eval,
            "per_image_ap": [round(a * 100, 2) for a in all_ap],
        }
        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(box_a, box_b):
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / max(union, 1e-6)

    def _max_iou(self, box, boxes):
        """Max IoU of *box* against an array of *boxes*."""
        return max(self._iou(box, b) for b in boxes)

    @staticmethod
    def _nms(boxes, scores, iou_threshold=0.3):
        """Greedy non-maximum suppression.

        Parameters
        ----------
        boxes : np.ndarray (N, 4)
        scores : np.ndarray (N,)
        iou_threshold : float

        Returns
        -------
        keep : list of int
        """
        x1 = boxes[:, 0].astype(float)
        y1 = boxes[:, 1].astype(float)
        x2 = boxes[:, 2].astype(float)
        y2 = boxes[:, 3].astype(float)
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def _compute_ap(self, det_boxes, det_scores, gt_boxes, iou_threshold=0.5):
        """Compute Average Precision for a single image.

        Uses the 11-point interpolation method.

        Returns
        -------
        ap : float
        tp_count : int
        """
        if len(gt_boxes) == 0:
            return (0.0 if len(det_boxes) > 0 else 1.0), 0

        if len(det_boxes) == 0:
            return 0.0, 0

        # Sort detections by score descending
        order = np.argsort(-det_scores)
        det_boxes = det_boxes[order]

        matched = np.zeros(len(gt_boxes), dtype=bool)
        tp = np.zeros(len(det_boxes))
        fp = np.zeros(len(det_boxes))

        for d, det in enumerate(det_boxes):
            best_iou = 0.0
            best_gt = -1
            for g, gt in enumerate(gt_boxes):
                iou_val = self._iou(det, gt)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt = g

            if best_iou >= iou_threshold and not matched[best_gt]:
                tp[d] = 1
                matched[best_gt] = True
            else:
                fp[d] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall_curve = tp_cum / len(gt_boxes)
        precision_curve = tp_cum / (tp_cum + fp_cum)

        # 11-point interpolated AP
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            prec_at_recall = precision_curve[recall_curve >= t]
            p = prec_at_recall.max() if len(prec_at_recall) > 0 else 0.0
            ap += p / 11.0

        return ap, int(tp.sum())
