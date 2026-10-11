# -*- coding: utf-8 -*-
"""
ESA's measured Sentinel-2 spectral response functions, resampled onto the
Tanager band grid.

Replaces the Gaussian approximations of nominal band centres/FWHM used in the
first versions of the comparison figures. Source:

  COPE-GSEG-EOPG-TN-15-0007 "Sentinel-2 Spectral Response Functions" (S2-SRF),
  v5, Copernicus / ESA. Downloaded to s2_srf_esa_v5.xlsx.

The spreadsheet gives response at 1 nm. Tanager samples at ~5 nm, so each
Tanager band is integrated over its own interval rather than point-sampled --
otherwise narrow structure in the response is aliased away.
"""
import os
import functools

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "s2_srf_esa_v5.xlsx")
SHEET = {"S2A": "Spectral Responses (S2A)",
         "S2B": "Spectral Responses (S2B)",
         "S2C": "Spectral Responses (S2C)"}


@functools.lru_cache(maxsize=4)
def load(unit="S2A"):
    """-> (wavelength_nm, {band_name: response}) at ESA's native 1 nm step."""
    if not os.path.exists(XLSX):
        raise FileNotFoundError(
            f"{XLSX} not found. Download the ESA S2-SRF spreadsheet "
            "(COPE-GSEG-EOPG-TN-15-0007) from sentiwiki.copernicus.eu.")
    df = pd.read_excel(XLSX, sheet_name=SHEET[unit])
    wl = df["SR_WL"].to_numpy(dtype=float)
    prefix = f"{unit}_SR_AV_"
    bands = {c[len(prefix):]: df[c].to_numpy(dtype=float)
             for c in df.columns if c.startswith(prefix)}
    return wl, bands


def band_stats(band, unit="S2A"):
    """Real centre of mass and FWHM in nm, measured from ESA's curve."""
    wl, bands = load(unit)
    r = bands[band]
    centre = float((wl * r).sum() / r.sum())
    half = r.max() / 2.0
    above = np.where(r >= half)[0]
    fwhm = float(wl[above[-1]] - wl[above[0]]) if len(above) > 1 else float("nan")
    lo = float(wl[above[0]]) if len(above) else float("nan")
    hi = float(wl[above[-1]]) if len(above) else float("nan")
    return {"centre": centre, "fwhm": fwhm, "lo50": lo, "hi50": hi}


def weights_on_grid(band, grid_nm, unit="S2A", grid_width=None):
    """
    Normalised weights for simulating `band` from a spectrum sampled on
    `grid_nm`. Each grid band is integrated over its own width, so both
    instruments' sampling is accounted for.

    Returns zeros (summing to 0) if the grid does not cover the band -- the
    caller should treat that as "this sensor cannot see here".
    """
    wl, bands = load(unit)
    r = bands[band]
    grid_nm = np.asarray(grid_nm, dtype=float)
    if grid_width is None:
        grid_width = float(np.median(np.diff(grid_nm)))

    w = np.empty(len(grid_nm))
    for i, c in enumerate(grid_nm):
        lo, hi = c - grid_width / 2.0, c + grid_width / 2.0
        m = (wl >= lo) & (wl < hi)
        w[i] = r[m].sum() if m.any() else 0.0
    total = w.sum()
    return w / total if total > 0 else w


def coverage(band, grid_nm, unit="S2A"):
    """Fraction of the band's total response that the grid actually spans."""
    wl, bands = load(unit)
    r = bands[band]
    lo, hi = float(np.min(grid_nm)), float(np.max(grid_nm))
    inside = r[(wl >= lo) & (wl <= hi)].sum()
    return float(inside / r.sum()) if r.sum() > 0 else 0.0
