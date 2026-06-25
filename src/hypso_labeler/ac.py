# ac.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class LightACConfig:
    # Dark pixel selection: percentile of NIR band used to estimate "dark spectrum"
    dark_percentile: float = 2.0

    # Reference wavelength for dark-mask band choice (nm).
    # HYPSO-2 max ~790 nm, so 780 is a good target.
    nir_ref_nm: float = 780.0

    # Surface reflectance clipping (ρs should usually be in [0, 1] for most surfaces)
    clip_min: float = 0.0
    clip_max: float = 1.2

    # Pressure (hPa). If you don't have it, use sea-level standard.
    pressure_hpa: float = 1013.25

    # Grid search for aerosol Angström exponent alpha
    alpha_min: float = 0.0
    alpha_max: float = 2.5
    alpha_step: float = 0.05

    # Smooth the dark spectrum with a moving average window (odd integer).
    # Set 1 to disable smoothing.
    smooth_window: int = 9

    # Verbose debug prints
    verbose: bool = False


def _mean_angles_from_satobj(satobj) -> Tuple[float, float]:
    """
    Returns mean solar zenith angle and mean view zenith angle in degrees.
    Falls back to metadata attrs if needed.
    """
    def _safe_mean(x) -> Optional[float]:
        try:
            v = float(np.nanmean(x))
            if np.isfinite(v):
                return v
        except Exception:
            pass
        return None

    sza = _safe_mean(getattr(satobj, "solar_zenith_angles", None))
    vza = _safe_mean(getattr(satobj, "sat_zenith_angles", None))

    if sza is None:
        sza = float(getattr(satobj, "solar_zenith_angle", 30.0))  # fallback guess
    if vza is None:
        vza = float(getattr(satobj, "sat_zenith_angle", 20.0))    # fallback guess

    return sza, vza


def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(xp, kernel, mode="valid")


def _rayleigh_optical_thickness(wl_um: np.ndarray, pressure_hpa: float) -> np.ndarray:
    """
    Rayleigh optical thickness approximation.
    Common empirical form (Bodhaine-like):
      tau_R = (P/P0) * 0.008569 * λ^-4 * (1 + 0.0113*λ^-2 + 0.00013*λ^-4)
    wl_um in micrometers.
    """
    wl = np.asarray(wl_um, dtype=np.float64)
    inv2 = wl ** -2
    inv4 = wl ** -4
    P0 = 1013.25
    tau = (pressure_hpa / P0) * 0.008569 * inv4 * (1.0 + 0.0113 * inv2 + 0.00013 * inv4)
    return tau


def _fit_path_spectrum_dark(
    dark_spec: np.ndarray,
    wl_nm: np.ndarray,
    cfg: LightACConfig,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Fit path reflectance spectrum p(λ) to the observed dark spectrum using:
      p(λ) = cR * Rshape(λ) + cA * Ashape(λ; alpha)
    where Rshape ~ λ^-4 (Rayleigh-like) and Ashape ~ (λ/0.55)^-alpha (aerosol power law).

    Returns:
      p (B,), best_alpha, cR, cA
    """
    wl_um = wl_nm.astype(np.float64) / 1000.0
    # Rayleigh-like spectral shape (not absolute; used as basis function)
    Rshape = _rayleigh_optical_thickness(wl_um, pressure_hpa=1013.25)  # shape only

    # Build alpha grid
    alphas = np.arange(cfg.alpha_min, cfg.alpha_max + 1e-12, cfg.alpha_step, dtype=np.float64)
    y = dark_spec.astype(np.float64)

    best = None  # (rss, alpha, cR, cA, p)
    for alpha in alphas:
        Ashape = (wl_um / 0.55) ** (-alpha)

        # Solve least squares for [cR, cA]
        A = np.stack([Rshape, Ashape], axis=1)  # (B,2)
        coeff, *_ = np.linalg.lstsq(A, y, rcond=None)  # coeff = [cR, cA]
        cR, cA = float(coeff[0]), float(coeff[1])
        p = cR * Rshape + cA * Ashape

        rss = float(np.mean((y - p) ** 2))
        if (best is None) or (rss < best[0]):
            best = (rss, float(alpha), cR, cA, p)

    assert best is not None
    _, best_alpha, cR, cA, p = best

    # Keep physically sensible nonnegative path reflectance
    p = np.maximum(p, 0.0)

    return p.astype(np.float64), best_alpha, cR, cA


def light_ac(
    satobj,
    rho_toa: np.ndarray,     # (H,W,B) TOA reflectance
    wl_nm: np.ndarray,       # (B,) band centers (nm)
    cfg: Optional[LightACConfig] = None,
) -> np.ndarray:
    """
    Lightweight atmospheric correction to approximate surface reflectance ρs:

      1) Select dark pixels using a NIR band (deep water / shadows).
      2) Compute "dark spectrum" across bands (median over dark pixels).
      3) Smooth the dark spectrum.
      4) Fit a smooth path reflectance spectrum p(λ) = Rayleigh-like + aerosol power-law.
      5) Subtract p(λ) from TOA and apply simple Rayleigh transmittance.

    Returns:
      rho_s (H,W,B) float32
    """
    if cfg is None:
        cfg = LightACConfig()

    rho_toa = np.asarray(rho_toa, dtype=np.float64)
    wl_nm = np.asarray(wl_nm, dtype=np.float64)

    if rho_toa.ndim != 3:
        raise ValueError(f"rho_toa must be (H,W,B), got {rho_toa.shape}")
    H, W, B = rho_toa.shape
    if wl_nm.shape != (B,):
        raise ValueError(f"wl_nm must be (B,), got {wl_nm.shape}")

    # Pick the band closest to nir_ref_nm for dark masking
    nir_idx = int(np.argmin(np.abs(wl_nm - cfg.nir_ref_nm)))
    nir_band = rho_toa[:, :, nir_idx]

    # Dark mask by percentile on the NIR band
    thr = np.nanpercentile(nir_band, cfg.dark_percentile)
    dark_mask = np.isfinite(nir_band) & (nir_band <= thr)

    # Ensure we have enough dark pixels
    n_dark = int(np.sum(dark_mask))
    if n_dark < 100:
        raise RuntimeError(
            f"Too few dark pixels found (n={n_dark}). "
            f"Try increasing dark_percentile (e.g., 5–10) or check scene content."
        )

    # Compute dark spectrum: median across dark pixels for each band
    dark_spec = np.empty((B,), dtype=np.float64)
    for b in range(B):
        band = rho_toa[:, :, b]
        vals = band[dark_mask]
        vals = vals[np.isfinite(vals)]
        dark_spec[b] = np.nanmedian(vals) if vals.size else np.nan

    # Replace any NaNs by linear interpolation over wavelength
    if np.any(~np.isfinite(dark_spec)):
        good = np.isfinite(dark_spec)
        if np.sum(good) < 5:
            raise RuntimeError("Dark spectrum has too many invalid bands to interpolate.")
        dark_spec[~good] = np.interp(wl_nm[~good], wl_nm[good], dark_spec[good])

    # Smooth dark spectrum (optional)
    dark_spec_sm = _moving_average_1d(dark_spec, cfg.smooth_window)

    # Fit smooth path spectrum
    p_spec, best_alpha, cR, cA = _fit_path_spectrum_dark(dark_spec_sm, wl_nm, cfg)

    # Simple Rayleigh transmittance (applied as a first-order correction)
    sza_deg, vza_deg = _mean_angles_from_satobj(satobj)
    mu_s = max(np.cos(np.deg2rad(sza_deg)), 1e-3)
    mu_v = max(np.cos(np.deg2rad(vza_deg)), 1e-3)

    wl_um = wl_nm / 1000.0
    tau_R = _rayleigh_optical_thickness(wl_um, pressure_hpa=cfg.pressure_hpa)
    # Two-way (down + up) extinction approximation
    T_R = np.exp(-tau_R * (1.0 / mu_s + 1.0 / mu_v))
    T_R = np.maximum(T_R, 1e-4)

    if cfg.verbose:
        print("[LightAC] NIR idx:", nir_idx, "wl:", wl_nm[nir_idx])
        print("[LightAC] dark_percentile:", cfg.dark_percentile, "thr:", float(thr), "n_dark:", n_dark)
        print("[LightAC] mean SZA/VZA:", sza_deg, vza_deg)
        print(f"[LightAC] fitted aerosol alpha={best_alpha:.2f}, cR={cR:.4g}, cA={cA:.4g}")
        print(f"[LightAC] dark_spec (first/last): {dark_spec_sm[:3]} ... {dark_spec_sm[-3:]}")

    # Apply correction: subtract path spectrum, divide by Rayleigh transmittance
    rho_s = np.empty_like(rho_toa, dtype=np.float64)
    for b in range(B):
        rho_s[:, :, b] = (rho_toa[:, :, b] - p_spec[b]) / T_R[b]

    # Clip to a sensible range
    rho_s = np.clip(rho_s, cfg.clip_min, cfg.clip_max)

    return rho_s.astype(np.float32, copy=False)
