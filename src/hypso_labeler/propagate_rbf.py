# propagate_rbf.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any, Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score


ProgressCB = Callable[[float, str], None]  # (fraction_0_to_1, message)


@dataclass
class PCARbfSVMModel:
    pca: PCA
    scaler: StandardScaler
    clf: SVC
    class_ids: np.ndarray
    best_params: Dict[str, Any]


def _sample_indices(idx: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if max_samples <= 0 or idx.size <= max_samples:
        return idx
    return rng.choice(idx, size=max_samples, replace=False)


def _stratified_split(
    y: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    (tr, va) = next(sss.split(np.zeros_like(y), y))
    return tr, va


def _report(cb: Optional[ProgressCB], frac: float, msg: str) -> None:
    if cb is None:
        return
    # clamp for safety
    f = float(max(0.0, min(1.0, frac)))
    cb(f, msg)


def fit_pca_rbf_svm_grid(
    cube: np.ndarray,
    labels: np.ndarray,
    *,
    k: int = 8,
    class_ids: Iterable[int] = (1, 2, 3),
    pca_max_pixels: int = 200_000,
    max_samples_per_class: int = 50_000,
    min_pixels_per_class: int = 20,
    C_grid: Tuple[float, ...] = (1.0, 10.0, 100.0),
    gamma_multipliers: Tuple[float, ...] = (0.1, 1.0, 10.0),
    val_fraction: float = 0.2,
    random_state: int = 0,
    progress_cb: Optional[ProgressCB] = None,
) -> PCARbfSVMModel:
    """
    Fit PCA + StandardScaler + RBF SVM, selecting (C, gamma) on a stratified holdout split.

    progress_cb: optional callback receiving (fraction_0_to_1, message)
    """
    if cube.ndim != 3:
        raise ValueError(f"cube must be (H,W,B), got {cube.shape}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be (H,W), got {labels.shape}")
    if cube.shape[:2] != labels.shape:
        raise ValueError(f"shape mismatch cube {cube.shape[:2]} vs labels {labels.shape}")
    if k <= 0:
        raise ValueError("k must be > 0")
    if not (0.0 < val_fraction < 0.9):
        raise ValueError("val_fraction must be in (0, 0.9)")

    _report(progress_cb, 0.01, "Preparing data…")

    H, W, B = cube.shape
    X = cube.reshape(-1, B).astype(np.float32, copy=False)
    y = labels.reshape(-1)
    rng = np.random.default_rng(random_state)

    # ---- PCA fit ----
    _report(progress_cb, 0.05, "Fitting PCA…")
    all_idx = np.arange(X.shape[0], dtype=np.int64)
    pca_idx = _sample_indices(all_idx, pca_max_pixels, rng)

    pca = PCA(n_components=min(k, B), whiten=False, random_state=random_state)
    pca.fit(X[pca_idx])

    # ---- Build labeled training set ----
    _report(progress_cb, 0.15, "Sampling labeled pixels…")
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

    # ---- PCA transform + scaling ----
    _report(progress_cb, 0.22, "Transforming features…")
    Z = pca.transform(X[train_idx]).astype(np.float32, copy=False)

    _report(progress_cb, 0.28, "Standardizing features…")
    scaler = StandardScaler(with_mean=True, with_std=True)
    Zs = scaler.fit_transform(Z)

    # ---- Holdout split ----
    _report(progress_cb, 0.32, "Preparing validation split…")
    tr_i, va_i = _stratified_split(y_train, test_size=val_fraction, random_state=random_state)
    Z_tr, y_tr = Zs[tr_i], y_train[tr_i]
    Z_va, y_va = Zs[va_i], y_train[va_i]

    gamma_base = 1.0 / float(pca.n_components_)

    # ---- Grid search ----
    nC = len(C_grid)
    nG = len(gamma_multipliers)
    n_total = nC * nG
    done = 0

    best_score = -np.inf
    best_params: Dict[str, Any] = {}
    best_clf: Optional[SVC] = None

    _report(progress_cb, 0.35, f"Grid search ({n_total} configs)…")

    # allocate progress window 0.35 -> 0.70
    p0, p1 = 0.35, 0.70

    for C in C_grid:
        for mult in gamma_multipliers:
            done += 1
            gamma = gamma_base * mult

            # update progress each iteration
            frac = p0 + (p1 - p0) * (done / n_total)
            _report(progress_cb, frac, f"Grid search {done}/{n_total} (C={C}, gamma={gamma:.3g})…")

            clf = SVC(
                kernel="rbf",
                C=float(C),
                gamma=float(gamma),
                class_weight="balanced",
            )
            clf.fit(Z_tr, y_tr)
            pred = clf.predict(Z_va)
            score = balanced_accuracy_score(y_va, pred)

            if score > best_score:
                best_score = score
                best_params = {"C": float(C), "gamma": float(gamma), "score_bal_acc": float(score)}
                best_clf = clf

    assert best_clf is not None

    # ---- Refit best model on all training data ----
    _report(progress_cb, 0.75, "Refitting best model on all labeled samples…")
    final_clf = SVC(
        kernel="rbf",
        C=best_params["C"],
        gamma=best_params["gamma"],
        class_weight="balanced",
    )
    final_clf.fit(Zs, y_train)

    _report(progress_cb, 0.82, "Model ready.")
    return PCARbfSVMModel(
        pca=pca,
        scaler=scaler,
        clf=final_clf,
        class_ids=class_ids_arr,
        best_params=best_params,
    )


def predict_pca_rbf_svm(
    cube: np.ndarray,
    model: PCARbfSVMModel,
    *,
    labels: Optional[np.ndarray] = None,
    fill_only_unlabeled: bool = True,
    chunk_size: int = 200_000,
    progress_cb: Optional[ProgressCB] = None,
) -> np.ndarray:
    """
    Predict full-scene labels using PCA + scaler + RBF SVM.
    progress_cb reports progress from ~0.85 to 1.00 in the wrapper, or 0..1 if used directly.
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

    n = int(target_idx.size)
    if n == 0:
        _report(progress_cb, 1.0, "No pixels to fill.")
        return out.reshape(H, W)

    _report(progress_cb, 0.0, "Predicting…")

    steps = (n + chunk_size - 1) // chunk_size
    for s, start in enumerate(range(0, n, chunk_size), start=1):
        ids = target_idx[start : start + chunk_size]
        Xi = X[ids]

        Zi = model.pca.transform(Xi).astype(np.float32, copy=False)
        Zi_s = model.scaler.transform(Zi)

        pred = model.clf.predict(Zi_s).astype(np.uint8)
        out[ids] = pred

        _report(progress_cb, s / steps, f"Predicting {s}/{steps}…")

    _report(progress_cb, 1.0, "Prediction done.")
    return out.reshape(H, W)


def propagate_svm_rbf_pca_grid(
    cube: np.ndarray,
    labels: np.ndarray,
    *,
    class_ids: Iterable[int] = (1, 2, 3),
    fill_only_unlabeled: bool = True,
    k: int = 8,
    pca_max_pixels: int = 200_000,
    max_samples_per_class: int = 50_000,
    min_pixels_per_class: int = 20,
    C_grid: Tuple[float, ...] = (1.0, 10.0, 100.0),
    gamma_multipliers: Tuple[float, ...] = (0.1, 1.0, 10.0),
    val_fraction: float = 0.2,
    chunk_size: int = 200_000,
    random_state: int = 0,
    progress_cb: Optional[ProgressCB] = None,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Wrapper with progress:
    - Fit model (0.0 -> ~0.85)
    - Predict scene (0.85 -> 1.0)
    """
    _report(progress_cb, 0.0, "Starting RBF SVM propagation…")

    # Fit stage
    def fit_cb(frac: float, msg: str) -> None:
        # map 0..1 into 0..0.85
        _report(progress_cb, 0.85 * frac, msg)

    model = fit_pca_rbf_svm_grid(
        cube=cube,
        labels=labels,
        k=k,
        class_ids=class_ids,
        pca_max_pixels=pca_max_pixels,
        max_samples_per_class=max_samples_per_class,
        min_pixels_per_class=min_pixels_per_class,
        C_grid=C_grid,
        gamma_multipliers=gamma_multipliers,
        val_fraction=val_fraction,
        random_state=random_state,
        progress_cb=fit_cb,
    )

    # Predict stage
    def pred_cb(frac: float, msg: str) -> None:
        # map 0..1 into 0.85..1.0
        _report(progress_cb, 0.85 + 0.15 * frac, msg)

    pred = predict_pca_rbf_svm(
        cube=cube,
        model=model,
        labels=labels,
        fill_only_unlabeled=fill_only_unlabeled,
        chunk_size=chunk_size,
        progress_cb=pred_cb,
    )

    _report(progress_cb, 1.0, "RBF SVM propagation done.")
    return pred, model.best_params
