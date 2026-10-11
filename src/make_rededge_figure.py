import json
import os
import sys

# Resolve the repository root and make it importable BEFORE any local import.
# Scripts run from the repo root or from a src/ subfolder (as shipped in the
# submission package); data files always live at the root.
REPO = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(REPO) == "src":
    REPO = os.path.dirname(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def repo_file(name):
    """Find a data file at the repo root, or under examples/output/."""
    for p in (os.path.join(REPO, name),
              os.path.join(REPO, "examples", "output", name)):
        if os.path.exists(p):
            return p
    return os.path.join(REPO, name)
# -*- coding: utf-8 -*-
"""
Sentinel-2 vs Tanager in the red edge: the documented-limitation figure.

Rather than download a Sentinel-2 scene from another date (different sun angle,
different atmosphere, different day of growth -- all confounds), this SIMULATES
Sentinel-2 by integrating the Tanager spectrum through Sentinel-2's band
responses. Same ground, same instant, same atmosphere; the only thing that
changes is how finely the spectrum is sampled. That isolates the variable we
actually want to argue about.

Produces rededge_sentinel2_vs_tanager.png and rededge_stats.txt
"""

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import ndimage
from scipy.interpolate import CubicSpline

SCENE_ID = "20250608_091605_90_4001"
H5 = os.path.join(REPO, f"{SCENE_ID}_ortho_sr_hdf5.h5")
ROOT = "HDFEOS/GRIDS/HYP/Data Fields"
FIGDIR = os.path.join(REPO, "executed", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# must match make_parcel_table.py exactly
NDVI_CROP_MIN, MIN_PIXELS, GRAD_PCT, MIN_MARKER, EPS = 0.35, 20, 50, 4, 1e-6

# Sentinel-2 bands, taken from ESA's MEASURED spectral response functions
# (S2-SRF v5), not from a Gaussian fit to the nominal centre/FWHM.
import s2_srf                                                    # noqa: E402

S2_UNIT = "S2A"
S2_BANDS = {"B4 red": "B4", "B5 RE1": "B5", "B6 RE2": "B6",
            "B7 RE3": "B7", "B8A nNIR": "B8A"}
S2 = {label: (s2_srf.band_stats(b, S2_UNIT)["centre"],
              s2_srf.band_stats(b, S2_UNIT)["fwhm"])
      for label, b in S2_BANDS.items()}

INK, GREEN, ORANGE, BLUE, GREY = "#14261f", "#0d3b2e", "#c2571a", "#1f6f8b", "#5d6b64"


def log(m):
    print(m, flush=True)


log("1/6  wavelengths")
url = (f"https://www.planet.com/data/stac/tanager-core-imagery/agriculture/"
       f"{SCENE_ID}/{SCENE_ID}.json")
_local = os.path.join(REPO, "examples", "input",
                      f"stac_item_{SCENE_ID}.json")
if os.path.exists(_local):
    with open(_local, encoding="utf-8") as _f:
        item = json.load(_f)
else:
    item = requests.get(url, timeout=120).json()
key = "ortho_sr_hdf5"
wl = np.array([b["eo:center_wavelength"]
               for b in item["assets"][key]["bands"]
               if "eo:center_wavelength" in b], dtype=float) * 1000.0
spacing = float(np.median(np.diff(wl)))
n_re = int(((wl >= 690) & (wl <= 740)).sum())
n_re_wide = int(((wl >= 670) & (wl <= 790)).sum())
log(f"     {len(wl)} bands, median spacing {spacing:.2f} nm")
log(f"     {n_re} Tanager bands in 690-740 nm; {n_re_wide} in 670-790 nm")

log("2/6  reading scene")
sel = np.where((wl >= 640) & (wl <= 900))[0]          # red edge working window
with h5py.File(H5, "r") as f:
    cloud = f[f"{ROOT}/beta_cloud_mask"][:]
    nodata = f[f"{ROOT}/nodata_pixels"][:]
    cirrus = f[f"{ROOT}/beta_cirrus_mask"][:]
    sr = f[f"{ROOT}/surface_reflectance"]
    cube = np.stack([sr[i, :, :].astype("float32") for i in sel])   # (B,Y,X)
    i_swir = int(np.argmin(np.abs(wl - 1650)))
    swir = sr[i_swir, :, :].astype("float32")
wl_sel = wl[sel]
valid = (nodata == 0) & (cloud == 0) & (cirrus == 0)
cube[:, ~valid] = np.nan
cube[cube < 0] = np.nan
swir[~valid] = np.nan
swir[swir < 0] = np.nan

i_red = int(np.argmin(np.abs(wl_sel - 665)))
i_nir = int(np.argmin(np.abs(wl_sel - 860)))
ndvi = (cube[i_nir] - cube[i_red]) / (cube[i_nir] + cube[i_red] + EPS)
ndmi = (cube[i_nir] - swir) / (cube[i_nir] + swir + EPS)

log("3/6  re-deriving parcels (same method as the table)")
full = np.ones((3, 3), bool)
crop = (np.nan_to_num(ndvi, nan=-9.0) > NDVI_CROP_MIN) & valid
eroded = ndimage.binary_erosion(crop, structure=full)
sm = ndimage.uniform_filter(np.nan_to_num(ndvi, nan=0.0), size=3)
grad = np.hypot(ndimage.sobel(sm, axis=0), ndimage.sobel(sm, axis=1))
g_in = grad[eroded]
flat = (grad < np.percentile(g_in, GRAD_PCT)) & eroded
mk, n_mk = ndimage.label(flat, structure=full)
sz = ndimage.sum(np.ones_like(mk), mk, index=np.arange(1, n_mk + 1))
kp = np.where(sz >= MIN_MARKER)[0] + 1
rm = np.zeros(n_mk + 1, np.int32)
rm[kp] = np.arange(1, len(kp) + 1)
mk = rm[mk]
g8 = np.clip(grad / np.percentile(g_in, 99) * 255, 0, 255).astype(np.uint8)
labels = ndimage.watershed_ift(g8, mk.astype(np.int32))
labels = np.where(eroded, labels, 0)
labels[labels < 0] = 0

# --- split spatially disconnected pieces (must match make_parcel_table.py) --
_conn4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], int)
_objs = ndimage.find_objects(labels)
_out = np.zeros_like(labels)
_next = 0
for _i, _sl in enumerate(_objs, start=1):
    if _sl is None:
        continue
    _cc, _k = ndimage.label(labels[_sl] == _i, structure=_conn4)
    for _j in range(1, _k + 1):
        _next += 1
        _view = _out[_sl]
        _view[_cc == _j] = _next
labels = _out

nr = int(labels.max())
sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, nr + 1))
keep = np.where(sizes >= MIN_PIXELS)[0] + 1
remap = np.zeros(nr + 1, np.int32)
remap[keep] = np.arange(1, len(keep) + 1)
labels = remap[labels]
n = len(keep)
log(f"     {n} parcels")

tbl = pd.read_csv(os.path.join(REPO, "parcel_table.csv"))
assert n == len(tbl), f"segmentation drifted: {n} vs {len(tbl)} in the table"

log("4/6  per-parcel mean spectra + simulated Sentinel-2")
ids = np.arange(1, n + 1)
spec = np.empty((n, len(wl_sel)), dtype="float64")
for b in range(len(wl_sel)):
    a = cube[b]
    ok = np.isfinite(a).astype("float32")
    tot = ndimage.sum(np.where(np.isfinite(a), a, 0.0), labels, index=ids)
    cnt = ndimage.sum(ok, labels, index=ids)
    spec[:, b] = np.array(tot) / np.maximum(np.array(cnt), 1)

pm_ndmi = np.array(ndimage.mean(np.nan_to_num(ndmi, nan=0.0), labels, index=ids))


def s2_band(label):
    """Integrate the Tanager spectrum through ESA's measured band response."""
    w = s2_srf.weights_on_grid(S2_BANDS[label], wl_sel, S2_UNIT)
    if w.sum() == 0:
        raise ValueError(f"{label} lies outside the loaded Tanager window")
    return spec @ w


s2 = {label: s2_band(label) for label in S2_BANDS}
for label, b in S2_BANDS.items():
    st = s2_srf.band_stats(b, S2_UNIT)
    log(f"     {label:10} ESA centre {st['centre']:7.2f} nm  "
        f"FWHM {st['fwhm']:4.0f} nm  coverage "
        f"{s2_srf.coverage(b, wl_sel, S2_UNIT)*100:.1f}%")
s2_centres = np.array([c for c, _ in S2.values()], dtype=float)

# --- Red Edge Position, two ways -----------------------------------------
# (a) Sentinel-2 can only manage the 4-point linear approximation
#     (Guyot & Baret): interpolate where reflectance crosses the midpoint.
r670, r700, r740, r780 = s2["B4 red"], s2["B5 RE1"], s2["B6 RE2"], s2["B7 RE3"]
mid = (r670 + r780) / 2.0
rep_s2 = 700.0 + 40.0 * (mid - r700) / (r740 - r700 + EPS)

# (b) Tanager can locate the actual inflection: the wavelength of maximum
#     first derivative, from contiguous ~5 nm bands.
m_re = (wl_sel >= 670) & (wl_sel <= 790)
x, fine = wl_sel[m_re], np.arange(670, 790.1, 0.5)
rep_tan = np.empty(n)
deriv_store = []
for i in range(n):
    y = spec[i, m_re]
    d = CubicSpline(x, y)(fine, 1)
    rep_tan[i] = fine[int(np.argmax(d))]
    deriv_store.append(d)
deriv_store = np.array(deriv_store)

ndre_s2 = (s2["B8A nNIR"] - s2["B5 RE1"]) / (s2["B8A nNIR"] + s2["B5 RE1"] + EPS)

log("5/6  scoring discriminating power")
# Grouping variable must share NO band with any metric being compared.
# NDMI is NOT safe: it uses 860 nm NIR, and Sentinel-2's NDRE uses B8A at
# 865 nm -- the same feature. Grouping by NDMI therefore hands NDRE the
# answer. Use a SWIR-only ratio instead: 1650/2200 nm touches neither the
# red edge nor the NIR shoulder that any of these metrics read.
i_s1 = int(np.argmin(np.abs(wl - 1650)))
i_s2b = int(np.argmin(np.abs(wl - 2200)))
with h5py.File(H5, "r") as f:
    srd = f[f"{ROOT}/surface_reflectance"]
    sw1 = srd[i_s1, :, :].astype("float32")
    sw2 = srd[i_s2b, :, :].astype("float32")
for a in (sw1, sw2):
    a[~valid] = np.nan
    a[a < 0] = np.nan
swir_ratio = (sw1 - sw2) / (sw1 + sw2 + EPS)
pm_swir = np.array(ndimage.mean(np.nan_to_num(swir_ratio, nan=0.0),
                                labels, index=ids))
lo_c, hi_c = np.percentile(pm_swir, [20, 80])
dry, wet = pm_swir <= lo_c, pm_swir >= hi_c
log("     grouping by SWIR 1650/2200 ratio (shares no band with any metric)")


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    return (a.mean() - b.mean()) / (sp + EPS)


def auc(a, b):
    """P(a sample from `b` scores above a sample from `a`) -- rank based."""
    allv = np.concatenate([a, b])
    r = pd.Series(allv).rank().values
    ra = r[:len(a)].sum()
    return (ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


metrics = {"Sentinel-2 NDRE": ndre_s2,
           "Sentinel-2 REP (4-point)": rep_s2,
           "Tanager REP (max derivative)": rep_tan}
stats = {}
for name, v in metrics.items():
    d = cohens_d(v[dry], v[wet])
    a = auc(v[dry], v[wet])
    stats[name] = (d, max(a, 1 - a))
    log(f"     {name:30} |d|={abs(d):.3f}  AUC={max(a, 1-a):.3f}")

# --- information test ----------------------------------------------------
# The honest question is not "which metric wins" but "does the dense sampling
# carry anything the five Sentinel-2 bands cannot reconstruct?". Regress the
# Tanager REP on the simulated S-2 bands: high R^2 means hyperspectral adds
# nothing here; residual variance is information S-2 physically cannot reach.
X = np.column_stack([s2[nm] for nm in S2] + [np.ones(n)])
beta, *_ = np.linalg.lstsq(X, rep_tan, rcond=None)
pred = X @ beta
ss_res = float(((rep_tan - pred) ** 2).sum())
ss_tot = float(((rep_tan - rep_tan.mean()) ** 2).sum())
r2_info = 1 - ss_res / ss_tot
rmse_info = float(np.sqrt(ss_res / n))
log(f"     S-2 bands predict Tanager REP: R2={r2_info:.3f}  "
    f"RMSE={rmse_info:.2f} nm")
log(f"     dynamic range  Tanager sd {rep_tan.std():.2f} nm  vs  "
    f"S-2 4-point sd {rep_s2.std():.2f} nm")

log("6/6  drawing")
fig = plt.figure(figsize=(15, 10.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], hspace=0.33, wspace=0.28)

# ---- panel A: one parcel, both samplings
ax = fig.add_subplot(gs[0, :2])
k = int(np.argmin(np.abs(rep_tan - np.median(rep_tan))))
m_plot = (wl_sel >= 640) & (wl_sel <= 900)
ax.plot(wl_sel[m_plot], spec[k, m_plot], "-", color=GREEN, lw=1.9,
        label=f"Tanager — {int(m_plot.sum())} bands @ {spacing:.0f} nm", zorder=3)
ax.plot(wl_sel[m_plot], spec[k, m_plot], ".", color=GREEN, ms=4, zorder=4)
sv = np.array([s2[nm][k] for nm in S2])
ax.plot(s2_centres, sv, "o", color=ORANGE, ms=11, zorder=5,
        label="Sentinel-2 — 5 bands (simulated)")
for (nm, (c, f)), v in zip(S2.items(), sv):
    ax.add_patch(plt.Rectangle((c - f / 2, 0), f, v, color=ORANGE, alpha=0.16, zorder=1))
    ax.annotate(nm, (c, v), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8, color=ORANGE)
ax.axvspan(690, 740, color=BLUE, alpha=0.07, zorder=0)
ax.annotate(f"red-edge window 690–740 nm\nTanager {n_re} bands · Sentinel-2 2 bands",
            (715, ax.get_ylim()[1] * 0.1), ha="center", fontsize=8.5, color=BLUE)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("surface reflectance")
ax.set_title(f"A  The same parcel, sampled two ways  (parcel {k+1}, "
             f"Çumra, 8 June 2025)", fontsize=11.5, color=INK, loc="left")
ax.legend(frameon=False, fontsize=9, loc="center left")
ax.set_ylim(top=ax.get_ylim()[1] * 1.12)   # headroom for the band labels
ax.grid(alpha=0.25)

# ---- panel B: first derivative, where REP actually lives
ax = fig.add_subplot(gs[0, 2])
ax.plot(fine, deriv_store[k], color=GREEN, lw=1.8)
ax.axvline(rep_tan[k], color=GREEN, ls="--", lw=1.4,
           label=f"Tanager REP {rep_tan[k]:.1f} nm")
ax.axvline(rep_s2[k], color=ORANGE, ls="--", lw=1.4,
           label=f"S-2 4-point REP {rep_s2[k]:.1f} nm")
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("dR/dλ")
ax.set_title("B  First derivative", fontsize=11.5, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8.5)
ax.grid(alpha=0.25)

# ---- panel C: REP distribution across all parcels
ax = fig.add_subplot(gs[1, 0])
ax.hist(rep_tan, bins=28, color=GREEN, alpha=0.85)
ax.set_xlabel("Tanager REP (nm)")
ax.set_ylabel("parcels")
ax.set_title(f"C  REP across {n} parcels\nspread {rep_tan.min():.0f}–"
             f"{rep_tan.max():.0f} nm", fontsize=11.5, color=INK, loc="left")
ax.grid(alpha=0.25, axis="y")

# ---- panel D: discriminating power
ax = fig.add_subplot(gs[1, 1])
names = list(stats)
vals = [stats[nm][1] for nm in names]
cols = [ORANGE, ORANGE, GREEN]
ax.barh(range(len(names)), vals, color=cols, alpha=0.9)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([nm.replace(" (", "\n(") for nm in names], fontsize=8.5)
ax.axvline(0.5, color=GREY, ls=":", lw=1.2)
ax.annotate("chance", (0.5, -0.62), fontsize=8, color=GREY, ha="center")
for i, v in enumerate(vals):
    ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0.45, max(vals) + 0.06)
ax.set_xlabel("AUC — separating low- from high-SWIR-ratio parcels")
ax.set_title("D  Separating an independent SWIR grouping\n"
             "(no shared bands — not a test of 'better')",
             fontsize=11.5, color=INK, loc="left")
ax.grid(alpha=0.25, axis="x")

# ---- panel E: the two REP estimates disagree
ax = fig.add_subplot(gs[1, 2])
ax.scatter(rep_s2, rep_tan, s=13, c=pm_ndmi, cmap="BrBG", alpha=0.85)
lim = [min(rep_s2.min(), rep_tan.min()) - 2, max(rep_s2.max(), rep_tan.max()) + 2]
ax.plot(lim, lim, color=GREY, ls="--", lw=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Sentinel-2 4-point REP (nm)")
ax.set_ylabel("Tanager derivative REP (nm)")
r = np.corrcoef(rep_s2, rep_tan)[0, 1]
ax.set_title(f"E  The two disagree  (r = {r:.2f})", fontsize=11.5,
             color=INK, loc="left")
plt.colorbar(ax.collections[0], ax=ax, shrink=0.8, label="parcel NDMI")
ax.grid(alpha=0.25)

fig.suptitle("Red-edge sampling: what Sentinel-2 can resolve vs what Tanager can",
             fontsize=14, color=INK, x=0.012, ha="left", y=0.985)
fig.text(0.012, 0.945,
         "Sentinel-2 simulated by integrating the Tanager spectrum through S-2 band "
         "responses — identical ground, instant and atmosphere, so sampling is the "
         "only difference.", fontsize=9.3, color=GREY, ha="left")
out = os.path.join(FIGDIR, "rededge_sentinel2_vs_tanager.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")

txt = f"""Red edge: Sentinel-2 vs Tanager
{'=' * 62}
Scene      {SCENE_ID}  Çumra, Konya, Turkey  2025-06-08
Parcels    {n}
Method     Sentinel-2 simulated from the Tanager spectrum through ESA's
           MEASURED spectral response functions (S2-SRF v5, {S2_UNIT}).
           Same pixels, same moment.

Spectral sampling
  Tanager median band spacing      {spacing:.2f} nm
  Tanager bands 690-740 nm         {n_re}
  Sentinel-2 bands 690-740 nm      2   (B5 705, B6 740)
  Tanager bands 670-790 nm         {n_re_wide}
  Sentinel-2 bands 670-790 nm      4   (B4, B5, B6, B7)

Red Edge Position
  Sentinel-2 4-point REP   mean {rep_s2.mean():.2f} nm   sd {rep_s2.std():.2f}
  Tanager derivative REP   mean {rep_tan.mean():.2f} nm   sd {rep_tan.std():.2f}
  correlation between them r = {r:.3f}

  Tanager REP dynamic range is {rep_tan.std() / (rep_s2.std() + EPS):.1f}x the
  Sentinel-2 4-point estimate's. The 4-point formula interpolates between
  700 and 740 nm, so it is bounded near there by construction and compresses
  real variation.

Does dense sampling carry anything Sentinel-2 cannot reconstruct?
  Tanager REP regressed on the 5 simulated Sentinel-2 bands
    R^2   {r2_info:.3f}
    RMSE  {rmse_info:.2f} nm
  {'Most REP variation IS recoverable from 5 bands.' if r2_info > 0.8 else
   'A substantial part of REP variation is NOT recoverable from 5 bands.'}

Discriminating power -- driest 20% vs wettest 20% of parcels, grouped by a
SWIR 1650/2200 ratio. That grouping shares no band with any metric compared,
so no metric is handed the answer.
"""
for nm, (d, a) in stats.items():
    txt += f"  {nm:32} |Cohen d| {abs(d):.3f}   AUC {a:.3f}\n"
best = max(stats, key=lambda k_: stats[k_][1])
txt += f"""
Best separator: {best}

HONEST READING
  An earlier version of this test grouped parcels by NDMI. That was wrong:
  NDMI uses 860 nm and Sentinel-2's NDRE uses B8A at 865 nm, so the grouping
  shared a band with one of the competitors and handed it a perfect score
  (AUC 1.000). The SWIR grouping above fixes that.
  Do NOT claim hyperspectral detects stress better than Sentinel-2 on the
  strength of this figure. Without ground truth there is no fair arbiter of
  "better". What this figure does support is narrower and defensible:
  sampling density, dynamic range, and how much REP information the five
  Sentinel-2 bands can reconstruct.

CAVEATS
  Sentinel-2 responses are ESA's published measured curves (S2-SRF v5, {S2_UNIT}),
  resampled onto the Tanager grid by integrating each 5 nm interval.
  Simulation shares Tanager's 30 m grid; real Sentinel-2 red-edge bands are
  20 m, so a real comparison also differs in spatial sampling.
  Single scene, single date, one region, no ground truth.
"""
with open(os.path.join(REPO, "rededge_stats.txt"), "w") as f:
    f.write(txt)
print()
print(txt)
log(f"wrote {out}")
