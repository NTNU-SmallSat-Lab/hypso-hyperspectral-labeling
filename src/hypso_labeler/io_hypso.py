# io_hypso.py
from __future__ import annotations

from hypso import Hypso1, Hypso2
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Literal
from dataclasses import dataclass


Mission = Literal["H1", "H2"]
Calibration = Literal["1a", "1b", "1c", "1d", "2a"]

@dataclass
class Scene:
    cube: np.ndarray
    rgb: np.ndarray
    wavelength_nm: Optional[np.ndarray] = None
    dropped_bands: Optional[list[int]] = None


def load_hyperspectral_dataH2(load_nc_path: Path, calibration: Calibration, verbose: bool = True) -> np.ndarray:
    satobj_h2 = Hypso2(path=load_nc_path, verbose=verbose)

    if calibration == "1a":
        cube = satobj_h2.l1a_cube.to_numpy().astype(np.float32, copy=False)
    elif calibration == "1b":
        satobj_h2.generate_l1b_cube()
        cube = satobj_h2.l1b_cube.to_numpy().astype(np.float32, copy=False)
    elif calibration == "1c":
        satobj_h2.generate_l1c_cube()
        cube = satobj_h2.l1c_cube.to_numpy().astype(np.float32, copy=False)
    elif calibration == "1d":
        satobj_h2.generate_l1c_cube()
        satobj_h2.generate_l1d_cube()
        cube = satobj_h2.l1d_cube.to_numpy().astype(np.float32, copy=False)
    elif calibration == "2a":
        from .ac import LightACConfig, light_ac
        satobj_h2.generate_l1c_cube()
        satobj_h2.generate_l1d_cube()
        rho_toa = satobj_h2.l1d_cube.to_numpy().astype(np.float32, copy=False)
        rho_toa = processing_drop_bandsH2(rho_toa)

        wl_nm = np.asarray(satobj_h2.wavelengths, dtype=np.float64)
        wl_nm = drop_bands_1dH2(wl_nm)

        #Lightweight AC to surface reflectance (ρs)
        cfg = LightACConfig(
            dark_percentile=2.0,   
            nir_ref_nm=780.0,      # HYPSO max band region
            pressure_hpa=1013.25,  # sea-level default
            smooth_window=9,       # smooth the dark spectrum
            verbose=verbose,
        )

        rho_s  = light_ac(
            satobj=satobj_h2,
            rho_toa=rho_toa ,
            wl_nm=wl_nm,
            cfg=cfg,
        )
        return rho_s.astype(np.float32, copy=False)
    
    else:
        raise ValueError(f"Unknown calibration: {calibration}")
        
    cube = processing_drop_bandsH2(cube)
    return cube
        
def drop_bands_1dH2(wavelengths_nm: np.ndarray) -> np.ndarray:
    drop = [0, 1, 2, 3, 4, 5, 6, 7, 119, 118]
    wavelengths_nm = np.asarray(wavelengths_nm)
    if wavelengths_nm.ndim != 1:
        raise ValueError(f"Expected wavelengths (B,), got {wavelengths_nm.shape}")
    if wavelengths_nm.shape[0] <= max(drop):
        raise ValueError(f"H2 drop list invalid for B={wavelengths_nm.shape[0]}")
    return np.delete(wavelengths_nm, drop, axis=0)

def drop_bands_1dH1(wavelengths_nm: np.ndarray) -> np.ndarray:
    drop = [0, 1, 2, 3, 4, 5, 119, 118, 117]
    wavelengths_nm = np.asarray(wavelengths_nm)
    if wavelengths_nm.ndim != 1:
        raise ValueError(f"Expected wavelengths (B,), got {wavelengths_nm.shape}")
    if wavelengths_nm.shape[0] <= max(drop):
        raise ValueError(f"H1 drop list invalid for B={wavelengths_nm.shape[0]}")
    return np.delete(wavelengths_nm, drop, axis=0)


def load_hyperspectral_dataH1(
    load_nc_path: Path,
    calibration: Calibration,
    verbose: bool = False,
) -> np.ndarray:
    satobj_h1 = Hypso1(path=load_nc_path, verbose=verbose)

    if calibration == "1a":
        cube = satobj_h1.l1a_cube.to_numpy().astype(np.float32, copy=False)

    elif calibration == "1b":
        satobj_h1.generate_l1b_cube()
        cube = satobj_h1.l1b_cube.to_numpy().astype(np.float32, copy=False)

    elif calibration == "1c":
        satobj_h1.generate_l1c_cube()
        cube = satobj_h1.l1c_cube.to_numpy().astype(np.float32, copy=False)

    elif calibration == "1d":
        satobj_h1.generate_l1c_cube()
        satobj_h1.generate_l1d_cube()
        cube = satobj_h1.l1d_cube.to_numpy().astype(np.float32, copy=False)

    elif calibration == "2a":
        # Interpret 2a as: surface reflectance (rho_s) from TOA reflectance (L1d)
        satobj_h1.generate_l1c_cube()
        satobj_h1.generate_l1d_cube()

        rho_toa = satobj_h1.l1d_cube.to_numpy().astype(np.float32, copy=False)
        rho_toa = processing_drop_bandsH1(rho_toa)

        wl_nm = np.asarray(satobj_h1.wavelengths, dtype=np.float64)
        wl_nm = drop_bands_1dH1(wl_nm)  # must match processing_drop_bandsH1 exactly

        cfg = LightACConfig(
            dark_percentile=2.0,
            nir_ref_nm=780.0,
            pressure_hpa=1013.25,
            smooth_window=9,
            verbose=verbose,
        )

        rho_s = light_ac(
            satobj=satobj_h1,
            rho_toa=rho_toa,
            wl_nm=wl_nm,
            cfg=cfg,
        )
        return rho_s.astype(np.float32, copy=False)

    else:
        raise ValueError(f"Unknown calibration: {calibration}")

    cube = processing_drop_bandsH1(cube)
    return cube

def processing_drop_bandsH2(data_cube: np.ndarray) -> np.ndarray:
    drop = [0, 1, 2, 3, 4, 5, 6, 7, 119, 118]
    if data_cube.ndim != 3:
        raise ValueError(f"Expected cube (H,W,B), got {data_cube.shape}")
    if data_cube.shape[-1] <= max(drop):
        raise ValueError(f"H2 drop list invalid for B={data_cube.shape[-1]}")
    return np.delete(data_cube, drop, axis=-1)


def processing_drop_bandsH1(data_cube: np.ndarray) -> np.ndarray:
    drop = [0, 1, 2, 3, 4, 5, 119, 118, 117]
    if data_cube.ndim != 3:
        raise ValueError(f"Expected cube (H,W,B), got {data_cube.shape}")
    if data_cube.shape[-1] <= max(drop):
        raise ValueError(f"H1 drop list invalid for B={data_cube.shape[-1]}")
    return np.delete(data_cube, drop, axis=-1)


def load_cube(nc_path: Path, mission: Mission, calibration: Calibration, verbose: bool = True) -> np.ndarray:
    path_nc = Path(nc_path)
    if not path_nc.exists():
        raise FileNotFoundError(f"Error in load_cube: path does not exist: {path_nc}")

    if mission == "H2":
        data_cube = load_hyperspectral_dataH2(path_nc, calibration, verbose)
    elif mission == "H1":
        data_cube = load_hyperspectral_dataH1(path_nc, calibration, verbose)
    else:
        raise ValueError("Error in load_cube: mission parameter needs to be set to 'H1' or 'H2'")

    if data_cube.ndim != 3:
        raise ValueError(f"Expected cube (H,W,B). Got {data_cube.shape}")

    return data_cube


def make_rgb(
    cube: np.ndarray,
    bands: Tuple[int, int, int],
    stretch_percentiles: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"Expected cube (H,W,B). Got {cube.shape}")

    H, W, B = cube.shape
    r, g, b = bands
    if not (0 <= r < B and 0 <= g < B and 0 <= b < B):
        raise ValueError(f"RGB band indices out of range for B={B}: {bands}")

    rgb = np.stack([cube[..., r], cube[..., g], cube[..., b]], axis=-1).astype(np.float32, copy=False)

    # behave like old version: one global stretch for all channels

    lo, hi = np.percentile(rgb, stretch_percentiles)
    if hi <= lo:
        return np.clip(rgb, 0.0, 1.0)
    rgb = (rgb - lo) / (hi - lo)
    return np.clip(rgb, 0.0, 1.0)

def default_rgb_bands(mission: Mission, B_after_drop: int) -> Tuple[int, int, int]:
    if B_after_drop <= 0:
        raise ValueError("B_after_drop must be > 0")

    if mission in ("H1", "H2"):
        return (
            int(0.75 * (B_after_drop - 1)),
            int(0.50 * (B_after_drop - 1)),
            int(0.25 * (B_after_drop - 1)),
        )

    raise ValueError("Error in default_rgb_bands: mission parameter needs to be set to 'H1' or 'H2'")


def load_scene(nc_path: Path, mission: Mission, calibration: Calibration, verbose: bool = True) -> Scene:
    if verbose:
        print("Loading scene:", nc_path)

    cube = load_cube(nc_path, mission, calibration, verbose)
    bands = default_rgb_bands(mission, cube.shape[-1])
    rgb = make_rgb(cube, bands=bands)

    return Scene(
        cube=cube,
        rgb=rgb,
        wavelength_nm=None,
        dropped_bands=None,
    )


def save_labels(path: Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask.astype(np.uint8, copy=False))


def load_labels(path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        # No labels yet -> return empty mask
        return np.zeros(shape_hw, dtype=np.uint8)

    m = np.load(path)
    if m.shape != shape_hw:
        raise ValueError(f"Label shape mismatch. Expected {shape_hw}, got {m.shape}")

    return m.astype(np.uint8, copy=False)
