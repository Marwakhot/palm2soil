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
SWIR absorption: the hyperspectral argument that actually holds up.

Two corrections over the first attempt, both worth knowing:

  1. The red-edge comparison showed Sentinel-2 reconstructs red-edge position
     better than expected (R2 0.68). The SWIR is a different story.
  2. A first pass measured SWIR features over CROPPED parcels and found a
     weak, largely artefactual "2200 nm feature". That was wrong by physics:
     salt crusts are a bare-soil phenomenon, and a closed canopy hides the
     soil. This version works on bare ground, where real, clean absorption
     features appear.

Sentinel-2 is simulated from the Tanager spectrum, so sampling is the only
variable.

Produces swir_absorption_argument.png and swir_stats.txt
"""

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy import ndimage

import s2_srf   # ESA's measured spectral response functions

SCENE_ID = "20250608_091605_90_4001"
H5 = os.path.join(REPO, f"{SCENE_ID}_ortho_sr_hdf5.h5")
ROOT = "HDFEOS/GRIDS/HYP/Data Fields"
FIGDIR = os.path.join(REPO, "executed", "figures")
os.makedirs(FIGDIR, exist_ok=True)
EPS = 1e-6

BARE_NDVI_MAX = 0.15     # bare ground
MIN_PATCH = 20           # pixels, ~1.8 ha
S2_UNIT = "S2A"
S2_SWIR_BANDS = ["B11", "B12"]
SALT_WINDOWS = [(1931, 2123), (2153, 2254)]
WV = [(1340, 1460), (1790, 1960)]
FEATURES = {"2216 nm  clay Al-OH / halite crust": (2150, 2270),
            "2340 nm  carbonate / gypsum": (2280, 2400)}

INK, GREEN, ORANGE, BLUE, GREY = "#14261f", "#0d3b2e", "#c2571a", "#1f6f8b", "#5d6b64"
CRIM = "#a02020"


def log(m):
    print(m, flush=True)


def continuum_remove(x, y):
    """Divide by the upper convex hull."""
    hull = [0]
    for j in range(1, len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (y[b] - y[a]) * (x[j] - x[a]) <= (y[j] - y[a]) * (x[b] - x[a]):
                hull.pop()
            else:
                break
        hull.append(j)
    return y / np.maximum(np.interp(x, x[hull], y[hull]), EPS)


log("1/5  wavelengths")
url = (f"https://www.planet.com/data/stac/tanager-core-imagery/agriculture/"
       f"{SCENE_ID}/{SCENE_ID}.json")
_local = os.path.join(REPO, "examples", "input",
                      f"stac_item_{SCENE_ID}.json")
if os.path.exists(_local):
    with open(_local, encoding="utf-8") as _f:
        item = json.load(_f)
else:
    item = requests.get(url, timeout=120).json()
wl = np.array([b["eo:center_wavelength"]
               for b in item["assets"]["ortho_sr_hdf5"]["bands"]
               if "eo:center_wavelength" in b], dtype=float) * 1000.0


def nb(a, b):
    return int(((wl >= a) & (wl <= b)).sum())


log("2/5  reading SWIR cube")
sel = np.where((wl >= 1450) & (wl <= 2450))[0]
with h5py.File(H5, "r") as f:
    cloud = f[f"{ROOT}/beta_cloud_mask"][:]
    nodata = f[f"{ROOT}/nodata_pixels"][:]
    cirrus = f[f"{ROOT}/beta_cirrus_mask"][:]
    sr = f[f"{ROOT}/surface_reflectance"]
    cube = np.stack([sr[i, :, :].astype("float32") for i in sel])
    red = sr[int(np.argmin(np.abs(wl - 665)))].astype("float32")
    nir = sr[int(np.argmin(np.abs(wl - 860)))].astype("float32")
    grn = sr[int(np.argmin(np.abs(wl - 570)))].astype("float32")
    sw1 = sr[int(np.argmin(np.abs(wl - 1650)))].astype("float32")
wls = wl[sel]
valid = (nodata == 0) & (cloud == 0) & (cirrus == 0)
cube[:, ~valid] = np.nan
cube[cube < 0] = np.nan
for a in (red, nir, grn, sw1):
    a[~valid] = np.nan
    a[a < 0] = np.nan
ndvi = (nir - red) / (nir + red + EPS)
bsi = ((sw1 + red) - (nir + grn)) / ((sw1 + red) + (nir + grn) + EPS)

wv_mask = np.zeros(len(wls), bool)
for a, b in WV:
    wv_mask |= (wls >= a) & (wls <= b)

log("3/5  segmenting bare-soil patches")
bare = (np.nan_to_num(ndvi, nan=9.0) < BARE_NDVI_MAX) & valid
bare = ndimage.binary_erosion(bare, structure=np.ones((3, 3), bool))
labels, nraw = ndimage.label(bare, structure=np.ones((3, 3), int))
sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, nraw + 1))
keep = np.where(sizes >= MIN_PATCH)[0] + 1
remap = np.zeros(nraw + 1, np.int32)
remap[keep] = np.arange(1, len(keep) + 1)
labels = remap[labels]
n = len(keep)
ids = np.arange(1, n + 1)
log(f"     {int(bare.sum())} bare pixels -> {n} patches >= {MIN_PATCH} px")

spec = np.empty((n, len(wls)))
for b in range(len(wls)):
    a = cube[b]
    tot = ndimage.sum(np.where(np.isfinite(a), a, 0.0), labels, index=ids)
    cnt = ndimage.sum(np.isfinite(a).astype("float32"), labels, index=ids)
    spec[:, b] = np.array(tot) / np.maximum(np.array(cnt), 1)
patch_bsi = np.array(ndimage.mean(np.nan_to_num(bsi, nan=0.0), labels, index=ids))

# Real band extents, measured from ESA's curves at the 1% response level.
_wl_srf, _srf = s2_srf.load(S2_UNIT)


def srf_extent(band, thr=0.01):
    r = _srf[band] / _srf[band].max()
    nz = _wl_srf[r > thr]
    return float(nz.min()), float(nz.max())


def srf_fraction(lo, hi, bands=("B11", "B12")):
    """Share of each band's total response that lands inside a window."""
    out = 0.0
    for b in bands:
        r = _srf[b]
        out += r[(_wl_srf >= lo) & (_wl_srf <= hi)].sum() / r.sum()
    return out


B11_LO, B11_HI = srf_extent("B11")
B12_LO, B12_HI = srf_extent("B12")
GAP = B12_LO - B11_HI


# Simulate Sentinel-2 through ESA's MEASURED response functions, not a
# Gaussian guess at the nominal band.
S2_STATS = {b: s2_srf.band_stats(b, S2_UNIT) for b in S2_SWIR_BANDS}
s2 = {}
for b in S2_SWIR_BANDS:
    w = s2_srf.weights_on_grid(b, wls, S2_UNIT)
    cov = s2_srf.coverage(b, wls, S2_UNIT)
    s2[b] = spec @ w
    st = S2_STATS[b]
    log(f"     {b}: ESA centre {st['centre']:.1f} nm, FWHM {st['fwhm']:.0f} nm, "
        f"50% {st['lo50']:.0f}-{st['hi50']:.0f}, grid covers {cov*100:.1f}%")

log("4/5  measuring absorption features")
feat, crs, xs = {}, {}, {}
for name, (lo, hi) in FEATURES.items():
    m = (wls >= lo) & (wls <= hi) & ~wv_mask
    x = wls[m]
    d = np.empty(n)
    pos = np.empty(n)
    cr_all = np.empty((n, m.sum()))
    for i in range(n):
        cr = continuum_remove(x, spec[i, m])
        cr_all[i] = cr
        k = int(np.argmin(cr))
        d[i] = 1.0 - cr[k]
        pos[i] = x[k]
    feat[name] = {"depth": d, "pos": pos}
    crs[name], xs[name] = cr_all, x
    log(f"     {name:36} depth {d.mean():.4f} +- {d.std():.4f}  "
        f"at {pos.mean():.0f} nm")

X2 = np.column_stack([s2["B11"], s2["B12"], np.ones(n)])
for name in FEATURES:
    for what in ("depth", "pos"):
        t = feat[name][what]
        beta, *_ = np.linalg.lstsq(X2, t, rcond=None)
        r2 = 1 - float(((t - X2 @ beta) ** 2).sum()) / float(((t - t.mean()) ** 2).sum())
        feat[name][f"r2_{what}"] = r2
    log(f"     B11+B12 reconstruct {name[:12]} depth: "
        f"R2={feat[name]['r2_depth']:.3f}")

log("5/5  drawing")
fig = plt.figure(figsize=(15, 10.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1.08, 1], hspace=0.36, wspace=0.27)

ax = fig.add_subplot(gs[0, :])
q = np.nanpercentile(spec, [10, 50, 90], axis=0)
good = ~wv_mask
ax.fill_between(wls, q[0], q[2], where=good, color=GREEN, alpha=0.18,
                label="bare-soil patch spread (10–90%)")
ax.plot(np.where(good, wls, np.nan), np.where(good, q[1], np.nan),
        color=GREEN, lw=1.7, label=f"Tanager median — {len(wls)} bands @ 5 nm")
for a, b in WV:
    ax.axvspan(a, b, color=GREY, alpha=0.14, zorder=0)
top = ax.get_ylim()[1]
ax.annotate("water vapour\n(unusable)", (1875, top * 0.9), ha="center",
            fontsize=8, color=GREY)
for nm_ in S2_SWIR_BANDS:
    st = S2_STATS[nm_]
    ax.axvspan(st["lo50"], st["hi50"], color=ORANGE, alpha=0.30, zorder=1)
    ax.annotate(f"S-2 {nm_}\nFWHM {st['fwhm']:.0f} nm", (st["centre"], top * 0.055),
                ha="center", fontsize=8.5, color=ORANGE, weight="bold")
for lo, hi, lab in ((B11_HI, B12_LO,
                     f"{GAP:.0f} nm blind gap\n{nb(B11_HI, B12_LO)} Tanager bands"),
                    (B12_HI, 2450,
                     f"beyond S-2 entirely\n{nb(B12_HI, 2450)} Tanager bands")):
    ax.axvspan(lo, hi, color=CRIM, alpha=0.07, zorder=0)
    ax.annotate(lab, ((lo + hi) / 2, top * 0.47), ha="center", fontsize=9.5,
                color=CRIM, weight="bold")
for (lo, hi) in [v for v in FEATURES.values()]:
    k = (wls >= lo) & (wls <= hi)
    ax.plot(wls[k], q[1][k], color=CRIM, lw=2.6, zorder=5)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("surface reflectance")
ax.set_title("A  The SWIR over bare soil, as each sensor sees it",
             fontsize=12, color=INK, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.grid(alpha=0.22)

for col, (name, (lo, hi)) in enumerate(FEATURES.items()):
    ax = fig.add_subplot(gs[1, col])
    x, cr = xs[name], crs[name]
    d = feat[name]["depth"]
    o = np.argsort(d)
    for i, c, lab in ((o[-1], "#6d2410", "deepest"),
                      (o[len(o) // 2], ORANGE, "median"),
                      (o[0], "#e0a070", "shallowest")):
        ax.plot(x, cr[i], color=c, lw=1.7, label=f"{lab} ({d[i]:.3f})")
    ax.axhline(1.0, color=GREY, ls=":", lw=1)
    inside = lo >= 2100 and hi <= 2280
    if inside:
        ax.axvspan(max(lo, 2100), min(hi, 2280), color=ORANGE, alpha=0.13)
        note = "all of this is ONE\nSentinel-2 number (B12)"
        nc = ORANGE
    else:
        note = "Sentinel-2 has NO band\nhere at all (B12 ends 2280)"
        nc = CRIM
    ax.text(0.5, 0.055, note, transform=ax.transAxes, ha="center",
            fontsize=8.8, color=nc, weight="bold")
    ax.set_xlabel("wavelength (nm)")
    if col == 0:
        ax.set_ylabel("continuum-removed reflectance")
    ax.set_title(f"{'BC'[col]}  {name}", fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.22)

ax = fig.add_subplot(gs[1, 2])
names = list(FEATURES)
vals = [feat[nm]["r2_depth"] for nm in names]
cols = [ORANGE, CRIM]
ax.barh(range(len(names)), vals, color=cols, alpha=0.9)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([nm.split("  ")[0] for nm in names], fontsize=10)
for i, v in enumerate(vals):
    ax.text(max(v, 0) + 0.015, i, f"R²={v:.2f}", va="center", fontsize=10)
    ax.text(0.02, i - 0.32, f"{(1-max(v,0))*100:.0f}% unrecoverable",
            fontsize=8.5, color=CRIM)
ax.set_xlim(0, 1.0)
ax.set_xlabel("R² — S-2's two SWIR bands reconstructing feature depth")
ax.set_title("D  What Sentinel-2 cannot recover", fontsize=11.5,
             color=INK, loc="left")
ax.grid(alpha=0.22, axis="x")

fig.suptitle("Why hyperspectral: the SWIR argument",
             fontsize=15, color=INK, x=0.012, ha="left", y=0.985)
fig.text(0.012, 0.947,
         "Sentinel-2 simulated from the Tanager spectrum — identical ground and "
         f"instant, so spectral sampling is the only difference. Çumra, 8 June 2025, "
         f"{n} bare-soil patches.", fontsize=9.4, color=GREY, ha="left")
out = os.path.join(FIGDIR, "swir_absorption_argument.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")

txt = f"""The SWIR argument for hyperspectral
{'=' * 66}
Scene    {SCENE_ID}  Çumra, Konya, Turkey  2025-06-08
Target   {n} bare-soil patches (NDVI < {BARE_NDVI_MAX}, >= {MIN_PATCH} px)
Method   Sentinel-2 simulated from the Tanager spectrum through ESA's
         MEASURED spectral response functions (S2-SRF v5, {S2_UNIT}).
         Same pixels, same instant -- sampling is the only variable.

WHY BARE SOIL AND NOT CROPS
  Salt crusts are a soil-surface phenomenon. Over a closed canopy the soil is
  hidden and the SWIR carries leaf water, not soil chemistry. Measuring these
  features over cropped parcels produced a weak, largely artefactual result;
  over bare ground the features are clean and deep. This also means detection
  and crop-stress monitoring need DIFFERENT pixels, which the product design
  should reflect.

THE PHYSICS, STATED CORRECTLY
  Pure halite (NaCl) has NO diagnostic absorption in VNIR-SWIR -- its crystal
  lattice produces no vibrational features there. Never claim direct NaCl
  detection; it is wrong and a technical panel will know it.
  What IS detectable: halite is hygroscopic, so salt crusts show secondary
  features in the water-absorption regions (reported near 1442, 1851, 1958
  and 2226 nm), and associated evaporites such as gypsum have genuinely
  diagnostic features (near 1023, 1225, 1457, 1757, 1800 and 2336 nm).
  Reported salt-correlated windows: 1931-2123 nm and 2153-2254 nm.

SENTINEL-2 SWIR BANDS, FROM ESA'S MEASURED RESPONSES
  (COPE-GSEG-EOPG-TN-15-0007 S2-SRF v5, unit {S2_UNIT}; extents at 1% response)
  B11  centre {S2_STATS['B11']['centre']:7.1f} nm  FWHM {S2_STATS['B11']['fwhm']:5.0f} nm  \
50% {S2_STATS['B11']['lo50']:.0f}-{S2_STATS['B11']['hi50']:.0f}  span {B11_LO:.0f}-{B11_HI:.0f}
  B12  centre {S2_STATS['B12']['centre']:7.1f} nm  FWHM {S2_STATS['B12']['fwhm']:5.0f} nm  \
50% {S2_STATS['B12']['lo50']:.0f}-{S2_STATS['B12']['hi50']:.0f}  span {B12_LO:.0f}-{B12_HI:.0f}

  B12's measured response is 0.002 of peak at 2320 nm and EXACTLY ZERO from
  2340 nm onward. The 2340 nm feature is not merely attenuated for
  Sentinel-2 -- it is entirely outside the instrument.

BAND COUNTS -- SENSOR DESIGN, TRUE REGARDLESS OF SCENE
  Window                       Tanager   Sentinel-2
  1931-2123 salt window        {nb(1931,2123):>5}      {srf_fraction(1931,2123)*100:.1f}% of S-2's SWIR response
  2153-2254 salt window        {nb(2153,2254):>5}      {srf_fraction(2153,2254)*100:.1f}%, all inside one band
  {B11_HI:.0f}-{B12_LO:.0f}                    {nb(B11_HI,B12_LO):>5}      ZERO -- a {GAP:.0f} nm blind gap
  {B12_HI:.0f}-2450                    {nb(B12_HI,2450):>5}      ZERO -- B12's response has ended
  1500-2500 total              {nb(1500,2500):>5}      2

  Between B11 and B12 lies {GAP:.0f} nm Sentinel-2 cannot see; past {B12_HI:.0f} nm it is
  blind again. Only {srf_fraction(1931,2123)*100:.1f}% of Sentinel-2's SWIR response falls in the
  1931-2123 salt window, and the gypsum/carbonate feature near 2340 nm sits
  beyond the instrument entirely.

FEATURES MEASURED ON BARE SOIL
"""
for name in FEATURES:
    d, p = feat[name]["depth"], feat[name]["pos"]
    txt += (f"  {name}\n"
            f"     depth    mean {d.mean():.4f}  sd {d.std():.4f}  "
            f"range {d.min():.4f}-{d.max():.4f}\n"
            f"     position mean {p.mean():.1f} nm  sd {p.std():.1f} nm\n")

txt += "\nHOW MUCH CAN SENTINEL-2'S TWO SWIR BANDS RECONSTRUCT?\n"
for name in FEATURES:
    r2 = feat[name]["r2_depth"]
    txt += (f"  {name.split('  ')[0]} depth   R2 {r2:.3f}   "
            f"{(1-max(r2,0))*100:.0f}% of the variance unrecoverable\n")
txt += f"""
  For comparison, Sentinel-2 reconstructed red-edge position at R2 0.68.
  In the SWIR it does far worse: B12 integrates a whole feature into one
  number, and beyond {B12_HI:.0f} nm its measured response is zero.

THE ARGUMENT, IN ONE PARAGRAPH
  Soil salinity is not directly visible to any optical sensor -- halite has no
  diagnostic absorption in the VNIR-SWIR. It is inferred from hydration
  features and associated evaporite minerals, whose absorptions are tens of
  nanometres wide and sit near 1450, 1850, 1950, 2200 and 2340 nm. Sentinel-2
  carries two SWIR bands; ESA's measured responses give them FWHM of \
{S2_STATS['B11']['fwhm']:.0f} and {S2_STATS['B12']['fwhm']:.0f} nm,
  a {GAP:.0f} nm gap between them, and no response at all beyond {B12_HI:.0f} nm. It cannot
  resolve a 40 nm absorption, and it misses several outright -- only \
{srf_fraction(1931,2123)*100:.1f}% of its
  SWIR response falls in the 1931-2123 nm salt window. Tanager samples the
  same range with {nb(1500,2500)} contiguous bands. Measured on bare soil in this scene,
  Sentinel-2's SWIR bands reconstruct only \
{min(feat[n_]['r2_depth'] for n_ in FEATURES)*100:.0f}-{max(feat[n_]['r2_depth'] for n_ in FEATURES)*100:.0f}% of the variance in
  absorption-feature depth. That is the documented limitation hyperspectral
  is being brought in to solve.

CAVEATS
  Sentinel-2 responses are ESA's published measured curves, resampled onto
  the Tanager grid by integrating each 5 nm interval. {S2_UNIT} used; B12 differs
  slightly between units (S2A centre 2202, S2B 2186, S2C 2191 nm), which does
  not change the conclusion.
  Water-vapour bands excluded from feature fitting.
  Single scene, one date, no ground truth. Feature depth is NOT calibrated to
  dS/m and must not be presented as a salinity measurement -- only as evidence
  that the signal exists and that broadband sensors cannot capture it.
  Çumra is an agricultural analogue, not a UAE scene. Mineralogy differs
  between the Konya plain and Al Ain, so the specific features present will
  differ even though the sampling argument does not.
"""
with open(os.path.join(REPO, "swir_stats.txt"), "w") as f:
    f.write(txt)
print()
print(txt)
log(f"wrote {out}")
