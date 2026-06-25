# gui_app.py
from __future__ import annotations
from .propagate_svm_pca import propagate_svm_pca_k8
from .propagate_rbf import propagate_svm_rbf_pca_grid
from .propagate_confidence import propagate_confidence_linear_svm_pca

from qtpy.QtWidgets import QProgressDialog
from qtpy.QtGui import QAction
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication
from qtpy.QtWidgets import QDockWidget
from qtpy.QtWidgets import QMessageBox

import json
import json
import webbrowser

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import copy
import numpy as np
import napari
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QFileDialog, QSpinBox
)
import pyqtgraph as pg

from .io_hypso import Scene, load_scene, save_labels, load_labels, Mission, Calibration, make_rgb, default_rgb_bands


CLASSES = {
    0: "unlabeled",
    1: "cloud",
    2: "land",
    3: "sea",
    4: "snow",
    5: "sand",
    6: "shallow_water",
    7: "man_made",
}

def load_meta_for_nc(nc_path: Path, meta_dir: Optional[Path] = None) -> Optional[dict]:
    """
    Try to load a matching meta json for an nc file.
    Expected layout: <repo>/meta/<stem>-meta.json
    Example: data/kutch_...-l1a.nc -> meta/kutch_...-meta.json
    """
    nc_path = Path(nc_path)

    if meta_dir is None:
        # assume repo structure: data/ and meta/ are siblings
        # nc_path.parent is likely .../data
        meta_dir = nc_path.parent.parent / "meta"

    meta_path = meta_dir / f"{nc_path.stem}-meta.json"
    if not meta_path.exists():
        return None

    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------
# Scaling helper functions
# -----------------------
def upsample_y_nearest(a: np.ndarray, scale_y: int) -> np.ndarray:
    """Repeat rows by scale_y (nearest neighbor). Works for (H,W) or (H,W,C)."""
    if scale_y == 1:
        return a
    return np.repeat(a, repeats=scale_y, axis=0)


def downsample_y_first_nonzero(labels_display: np.ndarray, scale_y: int) -> np.ndarray:
    """
    Downsample stretched labels (H*scale_y, W) -> (H, W) by taking the first non-zero
    label in each vertical block (top-to-bottom). If none -> 0.
    """
    if scale_y == 1:
        return labels_display.astype(np.uint8, copy=False)

    if labels_display.ndim != 2:
        raise ValueError(f"Expected 2D labels, got {labels_display.shape}")

    Hs, W = labels_display.shape
    if Hs % scale_y != 0:
        raise ValueError(f"Height {Hs} not divisible by scale_y={scale_y}")

    H = Hs // scale_y
    blk = labels_display.reshape(H, scale_y, W)  # (H, scale_y, W)

    # mask of labeled pixels
    nz = blk != 0  # bool (H, scale_y, W)

    # find first index along axis=1; argmax gives first True, but returns 0 if all False
    first_idx = nz.argmax(axis=1)  # (H, W)

    # check if there is any nonzero in the block
    has_any = nz.any(axis=1)  # (H, W)

    out = np.take_along_axis(blk, first_idx[:, None, :], axis=1).squeeze(1).astype(np.uint8)
    out[~has_any] = 0
    return out

def snap_labels_to_grid(labels_display: np.ndarray, scale_y: int) -> np.ndarray:
    """Return a snapped display mask (H*scale_y, W) from a display mask."""
    labels_orig = downsample_y_first_nonzero(labels_display, scale_y)   # (H, W)
    return upsample_y_nearest(labels_orig, scale_y)            # (H*scale_y, W)


@dataclass
class AppState:
    scene: Scene
    mission: Mission
    nc_path: Path
    out_dir: Path
    label_path: Path
    meta_path: Optional[Path] = None
    current_class_id: int = 1  # default to "cloud"
    scale_y: int = 4           # display stretch factor


class SpectrumWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.title = QLabel("Spectrum")
        layout.addWidget(self.title)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True)
        layout.addWidget(self.plot)

        self.curve_curr = self.plot.plot([], [], pen=pg.mkPen((120, 200, 255), width=2))  # light blue
        self.curve_pin  = self.plot.plot([], [], pen=pg.mkPen((255, 120, 120), width=2))  # light red

        self._pinned_spec = None
        self._pinned_xy = None

    def set_current(self, y: int, x: int, spec: np.ndarray, wavelength_nm=None):
        spec = np.asarray(spec, dtype=np.float32)
        xs = np.arange(len(spec)) if wavelength_nm is None else wavelength_nm
        self.curve_curr.setData(xs, spec)
        pin_txt = f" pinned (y={self._pinned_xy[0]}, x={self._pinned_xy[1]})" if self._pinned_xy else ""
        self.title.setText(f"Current (y={y}, x={x}){pin_txt}")

    def set_pinned(self, y: int, x: int, spec: np.ndarray, wavelength_nm=None):
        self._pinned_spec = np.asarray(spec, dtype=np.float32)
        self._pinned_xy = (y, x)
        xs = np.arange(len(self._pinned_spec)) if wavelength_nm is None else wavelength_nm
        self.curve_pin.setData(xs, self._pinned_spec)
        self.title.setText(f"Pinned (y={y}, x={x})")

    def clear_pinned(self):
        self._pinned_spec = None
        self._pinned_xy = None
        self.curve_pin.setData([], [])


class ControlPanel(QWidget):
    """Dock widget with class selector + save/load."""
    def __init__(self, state: AppState, labels_layer: napari.layers.Labels, rgb_layer: napari.layers.Image, spectrum: SpectrumWidget):
        super().__init__()
        self.state = state
        self.labels_layer = labels_layer
        self.rgb_layer = rgb_layer
        self.spectrum = spectrum

        self._undo_stack: list[np.ndarray] = []
        self._max_undo = 20

        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel("Class"))
        self.class_box = QComboBox()
        # Keep a list so we can map combo index -> class_id
        self.class_ids = list(CLASSES.keys())
        for cid in self.class_ids:
            self.class_box.addItem(f"{cid}: {CLASSES[cid]}", userData=cid)

        # Set current selection to state.current_class_id
        if state.current_class_id in self.class_ids:
            self.class_box.setCurrentIndex(self.class_ids.index(state.current_class_id))
        else:
            self.class_box.setCurrentIndex(0)
            state.current_class_id = self.class_ids[0]

        self.class_box.currentIndexChanged.connect(self.on_class_changed)
        layout.addWidget(self.class_box)

        # Buttons row
        btn_row = QHBoxLayout()

        # Button Save labels
        self.btn_save = QPushButton("Save labels")
        self.btn_save.clicked.connect(self.on_save)
        btn_row.addWidget(self.btn_save)

        # Button Load labels
        self.btn_load = QPushButton("Load labels")
        self.btn_load.clicked.connect(self.on_load)
        btn_row.addWidget(self.btn_load)

        # Button Clear all labels
        self.btn_clear = QPushButton("Clear all labels")
        self.btn_clear.clicked.connect(self.on_clear)
        layout.addWidget(self.btn_clear)

        self.btn_snap = QPushButton(f"Snap labels to grid (x{self.state.scale_y})")
        self.btn_snap.clicked.connect(self.on_snap_labels)
        layout.addWidget(self.btn_snap)

        layout.addLayout(btn_row)

        self.info = QLabel(f"Output: {state.label_path}")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        

        # --------------------
        # RGB settings section
        # --------------------
        layout.addWidget(QLabel("RGB settings"))

        # Default bands (based on cube after dropping bands)
        B_after_drop = self.state.scene.cube.shape[-1]
        self.default_rgb = default_rgb_bands(self.state.mission, B_after_drop)

        # Row for R/G/B spin boxes
        rgb_row = QHBoxLayout()

        rgb_row.addWidget(QLabel("R"))
        self.spin_r = QSpinBox()
        self.spin_r.setRange(0, B_after_drop - 1)
        self.spin_r.setValue(self.default_rgb[0])
        rgb_row.addWidget(self.spin_r)

        rgb_row.addWidget(QLabel("G"))
        self.spin_g = QSpinBox()
        self.spin_g.setRange(0, B_after_drop - 1)
        self.spin_g.setValue(self.default_rgb[1])
        rgb_row.addWidget(self.spin_g)

        rgb_row.addWidget(QLabel("B"))
        self.spin_b = QSpinBox()
        self.spin_b.setRange(0, B_after_drop - 1)
        self.spin_b.setValue(self.default_rgb[2])
        rgb_row.addWidget(self.spin_b)

        layout.addLayout(rgb_row)

        # Apply / Reset buttons
        rgb_btn_row = QHBoxLayout()

        self.btn_apply_rgb = QPushButton("Apply")
        self.btn_apply_rgb.clicked.connect(self.on_apply_rgb)
        rgb_btn_row.addWidget(self.btn_apply_rgb)

        self.btn_reset_rgb = QPushButton("Reset")
        self.btn_reset_rgb.clicked.connect(self.on_reset_rgb)
        rgb_btn_row.addWidget(self.btn_reset_rgb)

        self.btn_rgb_preset = QPushButton("Preset: 94 / 51 / 62")
        self.btn_rgb_preset.clicked.connect(self.on_rgb_preset_94_51_62)
        rgb_btn_row.addWidget(self.btn_rgb_preset)

        layout.addLayout(rgb_btn_row)

        # Button Propagate (centroid) 
        layout.addWidget(QLabel("Linear SVM & PCA"))
        self.btn_prop_svm = QPushButton("Propagate (Linear SVM, PCA k=8)")
        self.btn_prop_svm.clicked.connect(self.on_propagate_svm_pca)
        layout.addWidget(self.btn_prop_svm)

        """
        layout.addWidget(QLabel("RBF SVM"))
        self.btn_prop_svm_rbf = QPushButton("Propagate (RBF SVM, PCA k=8, small grid)")
        self.btn_prop_svm_rbf.clicked.connect(self.on_propagate_svm_rbf)
        layout.addWidget(self.btn_prop_svm_rbf)

        layout.addWidget(QLabel("Confidence propagation"))
        self.btn_prop_conf = QPushButton("Propagate (Calibrated SVM, PCA k=8, confidence fill)")
        self.btn_prop_conf.clicked.connect(self.on_propagate_confidence_svm_pca)
        layout.addWidget(self.btn_prop_conf)
        """

        # Undo button
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self.undo)
        layout.addWidget(self.btn_undo)

        self.btn_open_earth = QPushButton("Open in Google Earth")
        self.btn_open_earth.clicked.connect(self.on_open_google_earth)
        layout.addWidget(self.btn_open_earth)

        self.btn_open_worldview = QPushButton("Open in NASA Worldview")
        self.btn_open_worldview.clicked.connect(self.on_open_worldview)
        layout.addWidget(self.btn_open_worldview)



        layout.addStretch(1)

    def _push_undo(self):
        """Save current labels for undo (store a copy)."""
        cur = self.labels_layer.data
        self._undo_stack.append(cur.astype(np.uint8, copy=True))
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def undo(self):
        if not self._undo_stack:
            self.info.setText("Nothing to undo.")
            return
        prev = self._undo_stack.pop()
        self.labels_layer.data = prev
        self.info.setText("Undid last operation.")

    def on_class_changed(self, combo_idx: int):
        cid = int(self.class_box.itemData(combo_idx))
        self.state.current_class_id = cid
        self.labels_layer.selected_label = cid

    def on_save(self):
        labels_display = self.labels_layer.data.astype(np.uint8, copy=False)
        labels_orig = downsample_y_first_nonzero(labels_display, self.state.scale_y)
        save_labels(self.state.label_path, labels_orig)

    def on_load(self):
        # optional dialog: load arbitrary label file
        path, _ = QFileDialog.getOpenFileName(self, "Load labels", str(self.state.out_dir), "NumPy (*.npy)")
        if not path:
            return
        H, W = self.state.scene.cube.shape[:2]
        labels_orig = load_labels(Path(path), (H, W))
        labels_display = upsample_y_nearest(labels_orig, self.state.scale_y)
        self.labels_layer.data = labels_display
        self.info.setText(f"Loaded: {path}")
    
    def on_open_google_earth(self):
        mp = self.state.meta_path
        if mp is None:
            self.info.setText("No meta_path configured for this scene.")
            return

        if not mp.exists():
            self.info.setText(f"Meta file not found: {mp}")
            return

        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))

            # Prefer 'latitude'/'longitude' if present; fallback to 'target_latitude'/'target_longitude'
            lat = meta.get("latitude", meta.get("target_latitude", None))
            lon = meta.get("longitude", meta.get("target_longitude", None))

            if lat is None or lon is None:
                self.info.setText("Meta file missing latitude/longitude fields.")
                return

            lat = float(lat)
            lon = float(lon)

            # Google Earth Web accepts search lat/lon
            url = f"https://earth.google.com/web/search/{lat},{lon}"
            webbrowser.open(url)

            self.info.setText(f"Opened Google Earth at lat={lat:.6f}, lon={lon:.6f}")

        except Exception as e:
            self.info.setText(f"Failed to open Google Earth: {e}")
    
    def on_open_worldview(self):
        mp = self.state.meta_path
        if mp is None:
            self.info.setText("No meta_path configured for this scene.")
            return
        if not mp.exists():
            self.info.setText(f"Meta file not found: {mp}")
            return

        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))

            # Prefer 'latitude'/'longitude' if present; fallback to 'target_latitude'/'target_longitude'
            lat = meta.get("latitude", meta.get("target_latitude", None))
            lon = meta.get("longitude", meta.get("target_longitude", None))
            if lat is None or lon is None:
                self.info.setText("Meta file missing latitude/longitude fields.")
                return
            lat = float(lat)
            lon = float(lon)

            # Worldview time: use midnight UTC for that acquisition date
            # meta example: "timestamp_acquired_string": "2026-01-31 06:24:02+0000"
            tstr = meta.get("timestamp_acquired_string", "")
            if isinstance(tstr, str) and len(tstr) >= 10:
                date_str = tstr[:10]  # "YYYY-MM-DD"
            else:
                # fallback: no date available
                date_str = "2026-01-01"

            # Build a bounding box around the point.
            # Format is: v=lon_min,lat_min,lon_max,lat_max
            # Choose a small window; tweak if you want more/less zoom.
            dlon = 1.5
            dlat = 0.8
            lon_min = lon - dlon
            lon_max = lon + dlon
            lat_min = lat - dlat
            lat_max = lat + dlat

            # Worldview expects URL-encoded ":" in the time
            t_param = f"{date_str}-T00%3A00%3A00Z"

            url = (
                "https://worldview.earthdata.nasa.gov/"
                f"?v={lon_min},{lat_min},{lon_max},{lat_max}"
                f"&s={lon},{lat}"
                f"&t={t_param}"
            )

            webbrowser.open(url)
            self.info.setText(f"Opened NASA Worldview at lat={lat:.6f}, lon={lon:.6f} (date {date_str}).")

        except Exception as e:
            self.info.setText(f"Failed to open NASA Worldview: {e}")


    def on_propagate_confidence_svm_pca(self):
        """
        Confidence thresholding:
        - Fit PCA(k=8) + LinearSVM
        - Calibrate to get probabilities
        - Only fill pixels where max(prob) >= threshold
        """
        try:
            self._push_undo()
            from .propagate_confidence import propagate_confidence_linear_svm_pca

            cube = self.state.scene.cube
            labels_display = self.labels_layer.data.astype(np.uint8, copy=False)
            labels_orig = downsample_y_first_nonzero(labels_display, self.state.scale_y)

            # Tune this: 0.90 is a good starting point; higher = safer but fills less
            thr = 0.90

            new_mask_orig, stats = propagate_confidence_linear_svm_pca(
                cube=cube,
                labels=labels_orig,
                class_ids=tuple(cid for cid in CLASSES.keys() if cid != 0),
                fill_only_unlabeled=True,
                k=8,
                confidence_threshold=thr,
                # speed knobs (match your linear defaults)
                pca_max_pixels=2_000_000,
                max_samples_per_class=500_000,
                min_pixels_per_class=20,
                calibration_method="sigmoid",
                calibration_cv=3,
                chunk_size=200_000,
                random_state=0,
            )

            new_mask_display = upsample_y_nearest(new_mask_orig, self.state.scale_y)
            self.labels_layer.data = new_mask_display

            self.info.setText(
                f"Confidence fill done (thr={thr}). "
                f"filled {stats['filled_fraction']*100:.1f}% of unlabeled pixels."
            )

        except Exception as e:
            self.info.setText(f"Confidence SVM+PCA error: {e}")
    
    
    def on_propagate_svm_rbf(self):
        try:
            self._push_undo()
            from .propagate_rbf import propagate_svm_rbf_pca_grid

            cube = self.state.scene.cube
            labels_display = self.labels_layer.data.astype(np.uint8, copy=False)
            labels_orig = downsample_y_first_nonzero(labels_display, self.state.scale_y)

            # Progress dialog
            dlg = QProgressDialog("Starting…", "Cancel", 0, 100, self)
            dlg.setWindowTitle("RBF SVM propagation")
            dlg.setWindowModality(Qt.WindowModal)
            dlg.setMinimumDuration(0)
            dlg.setValue(0)

            cancelled = {"flag": False}

            def progress_cb(frac: float, msg: str) -> None:
                if dlg.wasCanceled():
                    cancelled["flag"] = True
                    return
                dlg.setLabelText(msg)
                dlg.setValue(int(frac * 100))
                QApplication.processEvents()

            new_mask_orig, best = propagate_svm_rbf_pca_grid(
                cube=cube,
                labels=labels_orig,
                class_ids=tuple(cid for cid in CLASSES.keys() if cid != 0),
                fill_only_unlabeled=True,
                k=8,
                pca_max_pixels=100_000,
                max_samples_per_class=5_000,
                min_pixels_per_class=20,
                chunk_size=200_000,
                random_state=0,
                progress_cb=progress_cb,
            )

            if cancelled["flag"]:
                self.info.setText("RBF SVM propagation cancelled.")
                return

            new_mask_display = upsample_y_nearest(new_mask_orig, self.state.scale_y)
            self.labels_layer.data = new_mask_display

            dlg.setValue(100)
            self.info.setText(
                f"Propagate done (RBF SVM + PCA k=8). best C={best.get('C')}, gamma={best.get('gamma')}, "
                f"bal_acc={best.get('score_bal_acc', 0.0):.3f}"
            )

        except Exception as e:
            self.info.setText(f"RBF SVM+PCA error: {e}")

    def on_propagate_svm_pca(self):
        try:
            self._push_undo()
            cube = self.state.scene.cube
            labels_display = self.labels_layer.data.astype(np.uint8, copy=False)
            labels_orig = downsample_y_first_nonzero(labels_display, self.state.scale_y)

            new_mask_orig = propagate_svm_pca_k8(
                cube=cube,
                labels=labels_orig,
                class_ids=tuple(cid for cid in CLASSES.keys() if cid != 0),
                fill_only_unlabeled=True,
                k=8,  # or 16, but match your button text
                pca_max_pixels=2_000_000,
                max_samples_per_class=500_000,
                min_pixels_per_class=20,
                chunk_size=200_000,
                random_state=0,
            )

            new_mask_display = upsample_y_nearest(new_mask_orig, self.state.scale_y)
            self.labels_layer.data = new_mask_display
            self.info.setText("Propagate done (Linear SVM + PCA k=8).")
        except Exception as e:
            self.info.setText(f"SVM+PCA error: {e}")
    
    def on_clear(self):
        reply = QMessageBox.question(
            self,
            "Confirm clear",
            "Clear ALL labels in this scene?\n",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,  # default selection
        )
        if reply != QMessageBox.Yes:
            return
        
        self._push_undo()
        self.labels_layer.data = np.zeros_like(self.labels_layer.data, dtype=np.uint8)
        self.info.setText("All labels cleared.")
    
    def on_apply_rgb(self):
        r = int(self.spin_r.value())
        g = int(self.spin_g.value())
        b = int(self.spin_b.value())

        try:
            rgb_orig = make_rgb(self.state.scene.cube, bands=(r, g, b))
            rgb_display = upsample_y_nearest(rgb_orig, self.state.scale_y)
            self.rgb_layer.data = rgb_display
            self.rgb_layer.contrast_limits = (0.0, 1.0)
            self.rgb_layer.gamma = 1.0
            self.info.setText(f"RGB updated: (R,G,B)=({r},{g},{b})")
        except Exception as e:
            self.info.setText(f"RGB apply error: {e}")

    def on_reset_rgb(self):
        r, g, b = self.default_rgb
        self.spin_r.setValue(int(r))
        self.spin_g.setValue(int(g))
        self.spin_b.setValue(int(b))
        self.on_apply_rgb()

    def on_rgb_preset_94_51_62(self):
        """
        Set RGB bands to (94, 51, 62) and update view.
        """
        r, g, b = 94, 51, 62
        B = self.state.scene.cube.shape[-1]

        # Safety check (important for robustness)
        if max(r, g, b) >= B:
            self.info.setText(
                f"RGB preset (94,51,62) out of range for B={B}"
            )
            return

        self.spin_r.setValue(r)
        self.spin_g.setValue(g)
        self.spin_b.setValue(b)

        self.on_apply_rgb()

    def on_snap_labels(self):
        try:
            labels_display = self.labels_layer.data.astype(np.uint8, copy=False)
            snapped = snap_labels_to_grid(labels_display, self.state.scale_y)
            self.labels_layer.data = snapped
            self.info.setText("Snapped labels to stretched grid.")
        except Exception as e:
            self.info.setText(f"Snap error: {e}")


def _infer_meta_path(nc_path: Path) -> Path:
        # Example:
        # data\kutch_...-l1a.nc  ->  meta\kutch_...-meta.json
        name = nc_path.name

        if name.endswith("-l1a.nc"):
            meta_name = name.replace("-l1a.nc", "-meta.json")
        elif name.endswith(".nc"):
            meta_name = name.replace(".nc", "-meta.json")
        else:
            meta_name = name + "-meta.json"

        # If nc is in /data/, meta is typically in sibling /meta/
        if nc_path.parent.name.lower() == "data":
            meta_dir = nc_path.parent.parent / "meta"
        else:
            meta_dir = nc_path.parent / "meta"

        return (meta_dir / meta_name)

def run_gui(
    nc_path: Path,
    out_dir: Path,
    mission: Mission = "H2",
    calibration: Calibration = "1a",
    verbose: bool = True,
) -> None:
    from qtpy import sip
    from qtpy.QtGui import QAction
    from qtpy.QtWidgets import QDockWidget

    nc_path = Path(nc_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = load_scene(nc_path, mission=mission, calibration=calibration, verbose=verbose)
    label_path = out_dir / f"{nc_path.stem}_labels.npy"

    # ----------------
    # Create viewer
    # ----------------
    viewer = napari.Viewer(title="HYPSO Labeler")

    # Display stretch factor (your current approach: stretch height)
    scale_y = 4

    # RGB layer
    rgb_display = upsample_y_nearest(scene.rgb, scale_y)
    rgb_layer = viewer.add_image(rgb_display, name="RGB", rgb=True)
    rgb_layer.contrast_limits = (0.0, 1.0)
    rgb_layer.gamma = 1.0

    # Labels layer
    H, W = scene.cube.shape[:2]
    labels_orig = load_labels(label_path, (H, W))
    labels_display = upsample_y_nearest(labels_orig, scale_y)
    labels_layer = viewer.add_labels(labels_display, name="Labels")
    default_colormap = copy.deepcopy(labels_layer.colormap)

    
    meta_path = _infer_meta_path(nc_path)


    # ----------------
    # State
    # ----------------
    state = AppState(
        scene=scene,
        mission=mission,
        nc_path=nc_path,
        out_dir=out_dir,
        label_path=label_path,
        meta_path=meta_path,
        scale_y=scale_y,
    )

    # -----------------------------
    # Helpers to create dock widgets
    # -----------------------------
    spectrum: Optional[SpectrumWidget] = None
    panel: Optional[ControlPanel] = None
    spectrum_dock = None
    controls_dock = None

    def _dock_alive(dock) -> bool:
        return dock is not None and not sip.isdeleted(dock)

    def _qt_dock(dock):
        # napari sometimes wraps the real QDockWidget
        return getattr(dock, "_qt_dock_widget", dock)

    def _try_disable_close(dock):
        """Best-effort: remove the close button (may vary by napari/qt)."""
        try:
            qd = _qt_dock(dock)
            qd.setFeatures(qd.features() & ~QDockWidget.DockWidgetClosable)
        except Exception:
            pass

    def _create_spectrum_dock():
        nonlocal spectrum, spectrum_dock
        spectrum = SpectrumWidget()
        spectrum_dock = viewer.window.add_dock_widget(spectrum, area="right", name="Spectrum")
        _try_disable_close(spectrum_dock)

    def _create_controls_dock():
        nonlocal panel, controls_dock
        # panel needs spectrum widget instance (even if recreated)
        panel = ControlPanel(state=state, labels_layer=labels_layer, rgb_layer=rgb_layer, spectrum=spectrum)
        controls_dock = viewer.window.add_dock_widget(panel, area="right", name="Controls")
        _try_disable_close(controls_dock)

    # Create initial docks
    _create_spectrum_dock()
    _create_controls_dock()

    # Default brush class
    labels_layer.selected_label = state.current_class_id

    # ----------------
    # Spectrum click callback
    # ----------------
    @viewer.bind_key("Ctrl-Z", overwrite=True)
    def _undo(viewer_):
        panel.undo()

    @viewer.mouse_drag_callbacks.append
    def on_click(viewer_, event):
        if event.type != "mouse_press":
            return

        y_disp = int(round(event.position[0]))
        x = int(round(event.position[1]))
        y = y_disp // state.scale_y

        H, W, _ = state.scene.cube.shape
        if not (0 <= y < H and 0 <= x < W):
            yield
            return

        spec = state.scene.cube[y, x, :]

        # Hold Shift while clicking to pin/update the second spectrum
        if "Shift" in event.modifiers:
            spectrum.set_pinned(y, x, spec, state.scene.wavelength_nm)
        else:
            spectrum.set_current(y, x, spec, state.scene.wavelength_nm)

        yield


    # ----------------
    # Reset + menu actions
    # ----------------
    def _show_dock(dock):
        if not _dock_alive(dock):
            return
        dock.setVisible(True)
        dock.show()
        dock.raise_()

    def reset_ui():
        nonlocal spectrum_dock, controls_dock

        # Recreate if deleted
        if not _dock_alive(spectrum_dock):
            _create_spectrum_dock()
        if not _dock_alive(controls_dock):
            _create_controls_dock()

        # Show again
        _show_dock(controls_dock)
        _show_dock(spectrum_dock)

        viewer.reset_view()
        labels_layer.selected_label = state.current_class_id
        if panel is not None:
            panel.info.setText("UI reset: panels reopened + view reset.")
        
        labels_layer.colormap = copy.deepcopy(default_colormap)
        labels_layer.refresh()


    # Add menu items once
    qt_window = viewer.window._qt_window  # QMainWindow
    menubar = qt_window.menuBar()

    view_menu = None
    for action in menubar.actions():
        if action.text().replace("&", "") == "View":
            view_menu = action.menu()
            break
    if view_menu is None:
        view_menu = menubar.addMenu("View")

    act_reset = QAction("Reset UI (reopen panels + reset view)", qt_window)
    act_reset.triggered.connect(reset_ui)
    view_menu.addAction(act_reset)

    # Always allow re-opening via menu toggles
    view_menu.addSeparator()
    view_menu.addAction(controls_dock.toggleViewAction())
    view_menu.addAction(spectrum_dock.toggleViewAction())

    napari.run()