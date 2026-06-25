# propagate_svm_pca.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass
class PCALinearSVMModel:
    pca: PCA
    scaler: StandardScaler
    clf: LinearSVC
    class_ids: np.ndarray


def _sample_indices(idx: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if max_samples <= 0 or idx.size <= max_samples:
        return idx
    return rng.choice(idx, size=max_samples, replace=False)


def fit_pca_linear_svm(
    cube: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 8,
    class_ids: Iterable[int] = (1, 2, 3),
    # PCA fit sampling (for speed)
    pca_max_pixels: int = 200_000,
    # SVM training sampling (for speed / balance)
    max_samples_per_class: int = 50_000,
    min_pixels_per_class: int = 20,
    random_state: int = 0,
) -> PCALinearSVMModel:
    """
    Fit PCA(k=8 by default) on pixels from this scene, then train a LinearSVM on labeled pixels in PCA space.

    cube:   (H,W,B) float32
    labels: (H,W) uint8/int with 0=unlabeled, classes in class_ids
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be (H,W,B), got {cube.shape}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be (H,W), got {labels.shape}")
    if cube.shape[:2] != labels.shape:
        raise ValueError(f"shape mismatch cube {cube.shape[:2]} vs labels {labels.shape}")
    if k <= 0:
        raise ValueError("k must be > 0")

    H, W, B = cube.shape
    X = cube.reshape(-1, B).astype(np.float32, copy=False)
    y = labels.reshape(-1)

    rng = np.random.default_rng(random_state)

    # --- Fit PCA on a (possibly sampled) subset of all pixels in the scene ---
    all_idx = np.arange(X.shape[0], dtype=np.int64)
    pca_idx = _sample_indices(all_idx, pca_max_pixels, rng)
    pca = PCA(n_components=min(k, B), whiten=False, random_state=random_state)
    pca.fit(X[pca_idx])

    # --- Build training set from labeled pixels (balanced sampling per class) ---
    class_ids = np.array(list(class_ids), dtype=np.int64)
    train_idx_list = []
    y_train_list = []
    
    present_ids = []
    train_idx_list = []
    y_train_list = []

    for cid in class_ids:
        idx = np.flatnonzero(y == cid)
        if idx.size < min_pixels_per_class:
            continue  # skip missing/too-small classes
        idx_s = _sample_indices(idx, max_samples_per_class, rng)
        train_idx_list.append(idx_s)
        y_train_list.append(np.full(idx_s.shape[0], cid, dtype=np.int64))
        present_ids.append(cid)

    if len(present_ids) < 2:
        raise ValueError(
            f"Need at least 2 classes with >= {min_pixels_per_class} labeled pixels. "
            f"Found: {present_ids}"
        )

    class_ids = np.array(present_ids, dtype=np.int64)

    train_idx = np.concatenate(train_idx_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    # PCA transform labeled pixels
    Z_train = pca.transform(X[train_idx]).astype(np.float32, copy=False)

    # Standardize PCA features (recommended for linear SVM)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z_train_s = scaler.fit_transform(Z_train)

    # Linear SVM (multi-class via one-vs-rest)
    clf = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    )
    clf.fit(Z_train_s, y_train)

    return PCALinearSVMModel(pca=pca, scaler=scaler, clf=clf, class_ids=class_ids)


def predict_pca_linear_svm(
    cube: np.ndarray,
    model: PCALinearSVMModel,
    *,
    labels: Optional[np.ndarray] = None,
    fill_only_unlabeled: bool = True,
    chunk_size: int = 200_000,
) -> np.ndarray:
    """
    Predict full-scene labels using PCA+LinearSVM, optionally filling only unlabeled (0) pixels.

    Returns: (H,W) uint8
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be (H,W,B), got {cube.shape}")

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

    for start in range(0, target_idx.size, chunk_size):
        ids = target_idx[start : start + chunk_size]
        Xi = X[ids]

        Zi = model.pca.transform(Xi).astype(np.float32, copy=False)
        Zi_s = model.scaler.transform(Zi)

        pred = model.clf.predict(Zi_s).astype(np.uint8)
        out[ids] = pred

    return out.reshape(H, W)


def propagate_svm_pca_k8(
    cube: np.ndarray,
    labels: np.ndarray,
    *,
    class_ids: Iterable[int] = (1, 2, 3),
    fill_only_unlabeled: bool = True,
    k: int = 8,
    pca_max_pixels: int = 200_000,
    max_samples_per_class: int = 50_000,
    min_pixels_per_class: int = 20,
    chunk_size: int = 200_000,
    random_state: int = 0,
) -> np.ndarray:
    """
    Convenience function: fit PCA(k=8) + LinearSVM on labeled pixels, then fill unlabeled pixels.
    """
    model = fit_pca_linear_svm(
        cube=cube,
        labels=labels,
        k=k,
        class_ids=class_ids,
        pca_max_pixels=pca_max_pixels,
        max_samples_per_class=max_samples_per_class,
        min_pixels_per_class=min_pixels_per_class,
        random_state=random_state,
    )
    return predict_pca_linear_svm(
        cube=cube,
        model=model,
        labels=labels,
        fill_only_unlabeled=fill_only_unlabeled,
        chunk_size=chunk_size,
    )