# propagate_confidence.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


@dataclass
class PCAConfidenceSVMModel:
    pca: PCA
    scaler: StandardScaler
    base_clf: LinearSVC
    calib_clf: CalibratedClassifierCV
    class_ids: np.ndarray
    info: Dict[str, Any]


def _sample_indices(idx: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if max_samples <= 0 or idx.size <= max_samples:
        return idx
    return rng.choice(idx, size=max_samples, replace=False)


def fit_pca_linear_svm_calibrated(
    cube: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 8,
    class_ids: Iterable[int] = (1, 2, 3),
    # PCA fit sampling (speed)
    pca_max_pixels: int = 200_000,
    # Training sampling (speed / balance)
    max_samples_per_class: int = 50_000,
    min_pixels_per_class: int = 20,
    # Calibration method
    calibration_method: str = "sigmoid",  # "sigmoid" is Platt scaling; "isotonic" needs more data
    calibration_cv: int = 3,              # small CV for speed
    random_state: int = 0,
) -> PCAConfidenceSVMModel:
    """
    Fit PCA + StandardScaler + LinearSVC, then calibrate to produce probabilities.

    - PCA is fit on a sampled set of all pixels in the scene.
    - LinearSVC is fit on labeled pixels (balanced sampling per class).
    - CalibratedClassifierCV wraps the LinearSVC to provide predict_proba().
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be (H,W,B), got {cube.shape}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be (H,W), got {labels.shape}")
    if cube.shape[:2] != labels.shape:
        raise ValueError(f"shape mismatch cube {cube.shape[:2]} vs labels {labels.shape}")
    if k <= 0:
        raise ValueError("k must be > 0")
    if calibration_method not in ("sigmoid", "isotonic"):
        raise ValueError("calibration_method must be 'sigmoid' or 'isotonic'")
    if calibration_cv < 2:
        raise ValueError("calibration_cv must be >= 2")

    H, W, B = cube.shape
    X = cube.reshape(-1, B).astype(np.float32, copy=False)
    y = labels.reshape(-1)

    rng = np.random.default_rng(random_state)

    # ---- PCA on sampled pixels from the whole scene ----
    all_idx = np.arange(X.shape[0], dtype=np.int64)
    pca_idx = _sample_indices(all_idx, pca_max_pixels, rng)

    pca = PCA(n_components=min(k, B), whiten=False, random_state=random_state)
    pca.fit(X[pca_idx])

    # ---- Build labeled training set (balanced sampling per class) ----
    class_ids_arr = np.array(list(class_ids), dtype=np.int64)

    present_ids = []
    train_idx_list = []
    y_train_list = []

    for cid in class_ids_arr:
        idx = np.flatnonzero(y == cid)
        if idx.size < min_pixels_per_class:
            continue
        idx_s = _sample_indices(idx, max_samples_per_class, rng)
        train_idx_list.append(idx_s)
        y_train_list.append(np.full(idx_s.shape[0], cid, dtype=np.int64))
        present_ids.append(cid)

    if len(present_ids) < 2:
        raise ValueError(
            f"Need at least 2 classes with >= {min_pixels_per_class} labeled pixels. Found: {present_ids}"
        )

    class_ids_arr = np.array(present_ids, dtype=np.int64)
    train_idx = np.concatenate(train_idx_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    # ---- PCA transform labeled pixels ----
    Z_train = pca.transform(X[train_idx]).astype(np.float32, copy=False)

    # ---- Standardize PCA features ----
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z_train_s = scaler.fit_transform(Z_train)

    # ---- Base Linear SVM ----
    base = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    )

    # ---- Calibrate to get probabilities ----
    # CalibratedClassifierCV will internally do CV splits on the provided training data.
    calib = CalibratedClassifierCV(
        estimator=base,
        method=calibration_method,
        cv=calibration_cv,
    )
    calib.fit(Z_train_s, y_train)

    info = {
        "k": int(pca.n_components_),
        "calibration_method": calibration_method,
        "calibration_cv": int(calibration_cv),
        "class_ids_present": class_ids_arr.tolist(),
    }

    return PCAConfidenceSVMModel(
        pca=pca,
        scaler=scaler,
        base_clf=base,
        calib_clf=calib,
        class_ids=class_ids_arr,
        info=info,
    )


def predict_confidence_fill_stable(
    cube: np.ndarray,
    model: PCAConfidenceSVMModel,
    *,
    labels: Optional[np.ndarray] = None,
    fill_only_unlabeled: bool = True,
    # Stability controls:
    max_fill_fraction: float = 0.15,      # never fill more than 15% of candidate pixels
    min_confidence: float = 0.80,         # absolute floor (don’t fill below this)
    gap_min: float = 0.15,                # require separation from runner-up class
    chunk_size: int = 200_000,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Stable version:
    - Compute max probability per candidate pixel
    - Use adaptive threshold based on top-quantile (max_fill_fraction)
    - Require margin gap (p1 - p2) >= gap_min
    - Also require p1 >= min_confidence (absolute floor)
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be (H,W,B), got {cube.shape}")
    if not (0.0 < max_fill_fraction <= 1.0):
        raise ValueError("max_fill_fraction must be in (0,1]")
    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("min_confidence must be in [0,1]")
    if not (0.0 <= gap_min <= 1.0):
        raise ValueError("gap_min must be in [0,1]")

    H, W, B = cube.shape
    X = cube.reshape(-1, B).astype(np.float32, copy=False)

    if fill_only_unlabeled:
        if labels is None:
            raise ValueError("labels must be provided when fill_only_unlabeled=True")
        if labels.shape != (H, W):
            raise ValueError(f"labels shape must be {(H,W)}, got {labels.shape}")
        y0 = labels.reshape(-1)
        target_idx = np.flatnonzero(y0 == 0)
        out = labels.reshape(-1).astype(np.uint8, copy=True)
    else:
        target_idx = np.arange(X.shape[0], dtype=np.int64)
        out = np.zeros((H * W,), dtype=np.uint8)

    total = int(target_idx.size)
    if total == 0:
        return out.reshape(H, W), {
            "filled_pixels": 0,
            "candidate_pixels": 0,
            "filled_fraction": 0.0,
            "effective_threshold": None,
        }

    # --- Pass 1: compute confidence (p1) and gap (p1-p2) for all candidate pixels ---
    # Store as float32 arrays sized to number of candidates
    p1_all = np.empty((total,), dtype=np.float32)
    gap_all = np.empty((total,), dtype=np.float32)
    pred_all = np.empty((total,), dtype=np.uint8)

    write_pos = 0
    for start in range(0, total, chunk_size):
        ids = target_idx[start : start + chunk_size]
        Xi = X[ids]

        Zi = model.pca.transform(Xi).astype(np.float32, copy=False)
        Zi_s = model.scaler.transform(Zi)

        proba = model.calib_clf.predict_proba(Zi_s)  # (n, C)

        # top1 + top2
        # argsort is heavier; do partial selection
        top1 = np.argmax(proba, axis=1)
        p1 = proba[np.arange(proba.shape[0]), top1]

        # get p2 efficiently by masking top1
        proba2 = proba.copy()
        proba2[np.arange(proba.shape[0]), top1] = -1.0
        top2 = np.argmax(proba2, axis=1)
        p2 = proba[np.arange(proba.shape[0]), top2]

        gap = p1 - p2
        pred = model.calib_clf.classes_[top1].astype(np.uint8)

        n = ids.size
        p1_all[write_pos : write_pos + n] = p1.astype(np.float32)
        gap_all[write_pos : write_pos + n] = gap.astype(np.float32)
        pred_all[write_pos : write_pos + n] = pred
        write_pos += n

    # --- Adaptive threshold: top max_fill_fraction by confidence ---
    # We choose threshold so that at most that fraction can pass (before gap/min_conf filters).
    k_keep = max(1, int(round(max_fill_fraction * total)))
    # kth largest threshold:
    # partition gives kth smallest; for kth largest use negative or select index total-k_keep
    thr_quant = float(np.partition(p1_all, total - k_keep)[total - k_keep])

    # Effective threshold is max(min_confidence, quantile threshold)
    thr_eff = max(min_confidence, thr_quant)

    # --- Apply filters and write output ---
    keep = (p1_all >= thr_eff) & (gap_all >= gap_min)
    filled = int(np.count_nonzero(keep))

    if filled > 0:
        out[target_idx[keep]] = pred_all[keep]

    stats = {
        "candidate_pixels": total,
        "filled_pixels": filled,
        "filled_fraction": float(filled / total),
        "effective_threshold": float(thr_eff),
        "quantile_threshold": float(thr_quant),
        "min_confidence": float(min_confidence),
        "gap_min": float(gap_min),
        "max_fill_fraction": float(max_fill_fraction),
    }
    return out.reshape(H, W), stats



def propagate_confidence_linear_svm_pca(
    cube: np.ndarray,
    labels: np.ndarray,
    *,
    class_ids: Iterable[int] = (1, 2, 3),
    fill_only_unlabeled: bool = True,
    k: int = 8,
    confidence_threshold: float = 0.90,
    pca_max_pixels: int = 200_000,
    max_samples_per_class: int = 50_000,
    min_pixels_per_class: int = 20,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 3,
    chunk_size: int = 200_000,
    random_state: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience wrapper:
    - Fit PCA + LinearSVM + calibration
    - Fill only high-confidence predictions (and optionally only unlabeled pixels)

    Returns: (mask, stats)
    """
    model = fit_pca_linear_svm_calibrated(
        cube=cube,
        labels=labels,
        k=k,
        class_ids=class_ids,
        pca_max_pixels=pca_max_pixels,
        max_samples_per_class=max_samples_per_class,
        min_pixels_per_class=min_pixels_per_class,
        calibration_method=calibration_method,
        calibration_cv=calibration_cv,
        random_state=random_state,
    )
    pred, stats = predict_confidence_fill_stable(
        cube=cube,
        model=model,
        labels=labels,
        fill_only_unlabeled=fill_only_unlabeled,
        max_fill_fraction=0.40,
        min_confidence=0.80,
        gap_min=0.15,
        chunk_size=chunk_size,
    )
    # include model info
    stats.update(model.info)
    return pred, stats
