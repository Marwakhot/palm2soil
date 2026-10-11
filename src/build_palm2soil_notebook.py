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
"""Generate 07_palm2soil_salinity_prioritisation.ipynb, the Palm2Soil PoC notebook.

Written as a builder rather than by hand so the notebook can be regenerated
after edits to the method, and so cell content stays under version control.
"""
import json

OUT = os.path.join(REPO, "07_palm2soil_salinity_prioritisation.ipynb")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ───────────────────────────────── title ──────────────────────────────────
md(r"""
# 🌾 Palm2Soil — Satellite-Guided Salinity Remediation

### Theme 1 · Precision Agriculture & Crop Intelligence · Arab Youth Space Hackathon: 813 Challenge

**Proof of Concept.** This notebook turns one Tanager-1 hyperspectral scene into a
**ranked remediation priority list** — one row per field parcel, ordered by a
chlorophyll-deficit score that detects crop stress before it becomes visible.

It follows the same 12-step pattern as the repository's theme notebooks, so it
should feel familiar if you have run `01_agriculture_crop_intelligence.ipynb`.

---

### What this notebook produces

| Output | What it is |
|--------|-----------|
| `parcel_table.csv` | One row per parcel: eight indices, stress score, rank, data quality |
| `parcels.geojson` | The same attributes with polygons, ready for a web map |
| Figures | Parcel maps, and the evidence that hyperspectral is necessary |

---

### ⚠️ Read this before the results

We are deliberately explicit about scope, because overclaiming is the fastest
way to lose a technical review.

**What this notebook demonstrates**
- A spatially coherent crop-stress signal exists and is measurable per parcel
- That signal aligns with independent moisture and soil-exposure indicators
- Specific soil absorption features exist that Sentinel-2 physically cannot resolve

**What it does *not* do**
- ❌ It does **not** predict absolute soil salinity in dS/m. No calibration exists
  without paired ground measurements, so all ranking is **relative within one scene**.
- ❌ It does **not** use cadastral field boundaries. Parcels are segmented from the
  imagery itself.
- ❌ It does **not** detect salt directly. Halite has no diagnostic absorption in
  this spectral range — see Step 11 for what is actually being measured.

Obtaining paired ground truth is the first objective of the incubation phase.
""")

# ───────────────────────────── step 1: install ────────────────────────────
md(r"""
## Step 1 — Install dependencies

Same stack as the other notebooks, plus `scipy` for the segmentation.
""")

code(r"""
%pip install -q pystac-client planetary-computer rasterio xarray matplotlib \
    geopandas h5py requests scipy pandas openpyxl
""")

code(r"""
import json, os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from affine import Affine
from pyproj import Transformer
from rasterio import features
from scipy import ndimage

# Prevents a known leafmap/Planetary Computer endpoint clash if other
# notebooks ran earlier in the same kernel.
os.environ["TITILER_ENDPOINT"] = "https://titiler.xyz"

EPS = 1e-6
plt.rcParams["figure.dpi"] = 110
print("ready")
""")

# ───────────────────────── step 2: scene selection ────────────────────────
md(r"""
## Step 2 — Choosing the scene, and why it is in Turkey

Our target is UAE farmland. **There is no usable UAE hyperspectral scene in the
open Tanager archive**, and we established that by auditing it rather than
assuming it.

All 276 scenes across all nine Tanager collections were checked. Three fall in
the Middle East / North Africa:

| Collection | Scene | Cloud |
|---|---|---|
| coastal-water-bodies | Tarif, Al Dhafrah, Abu Dhabi | 51% |
| urban | Riyadh, Saudi Arabia | 74% |
| natural-lands | Al Mahrah, Yemen | 53% |

The `agriculture` collection contains **no Arabian Peninsula scene at all**, and
the single UAE scene is in the wrong collection, 51% cloud, 25.8° off-nadir, and
covers Al Dhafrah rather than Al Ain — so it does not even sit under our
evidence base.

### Why Çumra is the right analogue

The **Konya–Çumra Plain** is a documented secondary-salinisation case and
structurally the same process as Al Ain:

- semi-arid **endorheic (closed) basin**
- **~90%** of irrigation water drawn from groundwater
- non-renewable abstraction causing roughly **1 m/year** water-table decline
- documented soil salinity and waterlogging from groundwater over-use
- published water–salt balance studies with measured dS/m values

Reported root-zone salinity there (~2.3–2.9 dS/m) is **milder** than the UAE
bands (6.25–9.4 moderate, >9.4 severe). We therefore frame this as detecting the
condition *before* it reaches UAE severity — which is the premise of the project
anyway.
""")

code(r"""
SCENE_ID   = "20250608_091605_90_4001"
COLLECTION = "agriculture"

# The STAC item is bundled in examples/input/ so this notebook runs with no
# network. If it is missing we fetch it from Planet's open catalogue.
STAC_LOCAL = os.path.join("examples", "input",
                          f"stac_item_{SCENE_ID}.json")
if os.path.exists(STAC_LOCAL):
    item = json.load(open(STAC_LOCAL, encoding="utf-8"))
    print("STAC item  : bundled ->", STAC_LOCAL)
else:
    url = (f"https://www.planet.com/data/stac/tanager-core-imagery/"
           f"{COLLECTION}/{SCENE_ID}/{SCENE_ID}.json")
    item = requests.get(url, timeout=120).json()
    print("STAC item  : fetched from Planet open catalogue")
p = item["properties"]

print("Scene      :", SCENE_ID)
print("Location   :", p.get("location_description"))
print("Acquired   :", p.get("datetime"))
print("Cloud      :", p.get("cloud_percent"), "%")
print("GSD        :", p.get("gsd"), "m")
print("Off-nadir  :", p.get("view:off_nadir"), "deg")
print("Licence    :", p.get("license"))
print("Bbox       :", [round(v, 3) for v in item["bbox"]])
""")

# ───────────────────────────── step 3: download ───────────────────────────
md(r"""
## Step 3 — Download the surface reflectance file

About 860 MB. The cell skips the download if the file is already present, so it
is safe to re-run.
""")

code(r"""
sr_key = "ortho_sr_hdf5" if "ortho_sr_hdf5" in item["assets"] else "basic_sr_hdf5"
sr_url  = item["assets"][sr_key]["href"]
FULL    = os.path.basename(sr_url)
SAMPLE  = os.path.join("examples", "input", "sample_cumra_128px_ortho_sr.h5")

# Set DOWNLOAD_FULL = True to fetch the complete 860 MB acquisition.
DOWNLOAD_FULL = False

if os.path.exists(FULL):
    local_path, MODE = FULL, "full scene"
elif DOWNLOAD_FULL:
    print("Downloading", sr_url, "(~860 MB)")
    with requests.get(sr_url, stream=True) as r:
        r.raise_for_status()
        with open(FULL, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    local_path, MODE = FULL, "full scene"
elif os.path.exists(SAMPLE):
    local_path, MODE = SAMPLE, "bundled sample"
else:
    raise FileNotFoundError(
        "No input found. Either keep examples/input/ alongside this notebook, "
        "or set DOWNLOAD_FULL = True to fetch the full acquisition.")

print(f"Input      : {MODE}")
print(f"File       : {local_path}  ({os.path.getsize(local_path)/1e6:.0f} MB)")
if MODE == "bundled sample":
    print("\nNOTE  This is a 128 x 128 px crop (3.8 x 3.8 km) stored at reduced")
    print("      precision so the notebook runs without an 860 MB download.")
    print("      Numbers below will differ from the full-scene figures quoted")
    print("      in the report. Set DOWNLOAD_FULL = True to reproduce those.")
""")

# ────────────────────────── step 4: scene quality ─────────────────────────
md(r"""
## Step 4 — Scene quality

Every downstream number inherits these masks, so we check them first and carry
the result through to the per-parcel table as a quality column.
""")

code(r"""
ROOT = "HDFEOS/GRIDS/HYP/Data Fields"

with h5py.File(local_path, "r") as f:
    cloud  = f[f"{ROOT}/beta_cloud_mask"][:]
    nodata = f[f"{ROOT}/nodata_pixels"][:]
    cirrus = f[f"{ROOT}/beta_cirrus_mask"][:]

valid = (nodata == 0) & (cloud == 0) & (cirrus == 0)
print(f"Scene size   : {cloud.shape[0]} x {cloud.shape[1]} px")
print(f"Valid pixels : {valid.mean()*100:.1f}%")
print(f"Cloud        : {(cloud != 0).mean()*100:.1f}%")
print(f"No-data      : {(nodata != 0).mean()*100:.1f}%")
""")

# ─────────────────────────── step 5: wavelengths ──────────────────────────
md(r"""
## Step 5 — Wavelengths and band selection

Tanager records **426 bands** at roughly 5 nm spacing. We load only the handful
the indices need — reading 7 bands instead of 426 is what makes an 860 MB file
analysable on a laptop.

The first five targets match `01_agriculture_crop_intelligence.ipynb` exactly so
our numbers are comparable with the starter notebook; blue and SWIR-2 are added
for the salinity candidates.
""")

code(r"""
bands_meta = item["assets"][sr_key].get("bands", [])
wl = np.array([b["eo:center_wavelength"] for b in bands_meta
               if "eo:center_wavelength" in b], dtype=float) * 1000.0  # µm -> nm

print(f"{len(wl)} bands, {wl[0]:.0f}-{wl[-1]:.0f} nm, "
      f"median spacing {np.median(np.diff(wl)):.2f} nm\n")

TARGETS = {"blue": 490, "green": 570, "red": 665, "rededge": 705,
           "nir": 860, "swir1": 1650, "swir2": 2200}
idx = {}
for name, t in TARGETS.items():
    i = int(np.argmin(np.abs(wl - t)))
    idx[name] = i
    print(f"  {name:8} target {t:>5} nm -> band {i:3d} ({wl[i]:.1f} nm)")
""")

# ──────────────────────── step 6: load + indices ──────────────────────────
md(r"""
## Step 6 — Load bands and compute indices

Each index asks the same question — *how different are these two colours?* —
normalised so illumination cancels out:

$$\text{index} = \frac{A - B}{A + B}$$

| Index | Compares | High means |
|---|---|---|
| **NDVI** | NIR vs red | healthy green growth |
| **NDRE** | NIR vs red edge | plenty of chlorophyll *inside* the leaf |
| **NDMI** | NIR vs SWIR | leaves full of water |
| **BSI** | soil colours vs plant colours | exposed bare soil |

**NDMI matters most for salinity.** Salt stops a plant taking up water even when
the soil is wet, so salt stress shows as falling canopy water — distinguishable
from drought, where the soil is dry too.

### Published salinity indices

We also compute four indices from the soil-salinity literature. Their formulas
were verified against published sources (`SALINITY_INDEX_VERIFICATION.md`), and
two corrections came out of that check:

1. **Names in this literature are ambiguous.** "SI1" refers to at least three
   different formulas across different papers, and band numbers mean different
   wavelengths on different sensors. We therefore name each index for its source
   rather than by number.
2. **These formulas were designed for broadband sensors.** A single 5 nm Tanager
   band is not the same measurement as a 34 nm Sentinel-2 green band. We
   integrate the Tanager spectrum through ESA's measured Sentinel-2 response
   functions first, so the values mean what the literature means.

> ℹ️ These indices are **descriptive context, not the prioritisation mechanism**.
> The ranking is driven by the NDVI–NDRE residual in Step 9 and does not depend
> on any of them. None is calibrated to dS/m.
""")

code(r"""
import s2_srf   # ESA measured Sentinel-2 spectral response functions

vnir_sel = np.where((wl >= 430) & (wl <= 920))[0]      # for broadband synthesis

with h5py.File(local_path, "r") as f:
    sr  = f[f"{ROOT}/surface_reflectance"]
    R = {n: sr[i, :, :].astype("float32") for n, i in idx.items()}
    vnir = np.stack([sr[i, :, :].astype("float32") for i in vnir_sel])
    # full scene carries the per-band uncertainty cube; the bundled sample
    # carries the 2-D mean over the bands this pipeline reads
    if f"{ROOT}/surface_reflectance_uncertainty" in f:
        unc = f[f"{ROOT}/surface_reflectance_uncertainty"]
        U = np.nanmean(np.stack([unc[i, :, :].astype("float32")
                                 for i in idx.values()]), axis=0)
    else:
        U = f[f"{ROOT}/surface_reflectance_uncertainty_mean"][:].astype("float32")

wl_vnir = wl[vnir_sel]
for a in R.values():
    a[~valid] = np.nan
    a[a < 0] = np.nan
U[~valid] = np.nan
vnir[:, ~valid] = np.nan
vnir[vnir < 0] = np.nan

b, g, r_, re_, nir, sw1, sw2 = (R["blue"], R["green"], R["red"], R["rededge"],
                                R["nir"], R["swir1"], R["swir2"])

# --- canopy indices, on narrow bands (as in 01_agriculture, for comparability)
ix = {}
ix["NDVI"] = (nir - r_)  / (nir + r_  + EPS)
ix["NDRE"] = (nir - re_) / (nir + re_ + EPS)
ix["NDMI"] = (nir - sw1) / (nir + sw1 + EPS)
ix["BSI"]  = ((sw1 + r_) - (nir + g)) / ((sw1 + r_) + (nir + g) + EPS)

# --- published salinity indices, on BROADBAND-equivalent reflectance
def broadband(band):
    # integrate Tanager through ESA's measured Sentinel-2 band response
    w = s2_srf.weights_on_grid(band, wl_vnir, "S2A")
    return np.tensordot(w, vnir, axes=(0, 0))

bb = {k: broadband(v) for k, v in
      {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8A"}.items()}
B_, G_, R_, N_ = bb["blue"], bb["green"], bb["red"], bb["nir"]

with np.errstate(invalid="ignore"):
    ix["SI_Khan"]    = np.sqrt(np.clip(G_ * R_, 0, None))            # Khan et al.
    ix["SI_Douaoui"] = np.sqrt(np.clip(G_**2 + R_**2 + N_**2, 0, None))  # Douaoui 2006
    ix["BI"]         = np.sqrt(np.clip(R_**2 + N_**2, 0, None))      # Brightness Index
    ix["CRSI"]       = np.sqrt(np.clip((N_*R_ - G_*B_) / (N_*R_ + G_*B_ + EPS),
                                       0, None))                     # Scudiero 2014/15

for k, v in ix.items():
    print(f"  {k:11} mean {np.nanmean(v):7.4f}   "
          f"range {np.nanmin(v):7.4f} to {np.nanmax(v):7.4f}")
""")

# ──────────────────────── step 7: parcel segmentation ─────────────────────
md(r"""
## Step 7 — Segment the scene into field parcels

**This step is the method, not a technicality.**

Scene-wide averages are useless here. The scene is a patchwork of small
individually managed fields, so a whole-scene mean blends thriving crop with bare
ground and describes neither. Our customer allocates budget *per farm block*, so
the unit of analysis must be the parcel.

### Why not simple connected components

Adjacent fields are spectrally contiguous, so naive labelling merges whole
farming districts into one blob — in testing it produced a single 866 ha
"parcel". Instead we use a **watershed on the NDVI gradient**: field interiors
are spectrally flat, boundaries are edges. Markers are seeded in flat interiors
and grown to the edges.

> ⚠️ **Limitation to state in any write-up:** these parcels are segmented from
> imagery, **not** from a cadastral registry. They approximate management units.
> Parcels far above the median size are flagged `needs_split` as probably several
> fields merged.
""")

code(r"""
NDVI_CROP_MIN = 0.35   # a pixel counts as actively cropped above this
MIN_PIXELS    = 20     # smallest parcel kept, ~1.8 ha at 30 m
GRAD_PCT      = 50     # gradient percentile defining "field interior"
MIN_MARKER    = 4      # smallest seed marker, px
BIG_PARCEL_HA = 50     # above this, flag as probably merged fields

full = np.ones((3, 3), bool)
crop   = (np.nan_to_num(ix["NDVI"], nan=-9.0) > NDVI_CROP_MIN) & valid
eroded = ndimage.binary_erosion(crop, structure=full)   # drop mixed edge pixels

sm   = ndimage.uniform_filter(np.nan_to_num(ix["NDVI"], nan=0.0), size=3)
grad = np.hypot(ndimage.sobel(sm, axis=0), ndimage.sobel(sm, axis=1))
g_in = grad[eroded]

flat = (grad < np.percentile(g_in, GRAD_PCT)) & eroded
mk, n_mk = ndimage.label(flat, structure=full)
mk_sz   = ndimage.sum(np.ones_like(mk), mk, index=np.arange(1, n_mk + 1))
mk_keep = np.where(mk_sz >= MIN_MARKER)[0] + 1
mk_remap = np.zeros(n_mk + 1, np.int32)
mk_remap[mk_keep] = np.arange(1, len(mk_keep) + 1)
mk = mk_remap[mk]

g8 = np.clip(grad / np.percentile(g_in, 99) * 255, 0, 255).astype(np.uint8)
labels = ndimage.watershed_ift(g8, mk.astype(np.int32))
labels = np.where(eroded, labels, 0)
labels[labels < 0] = 0                                   # watershed ridge lines


# --- split spatially disconnected pieces ---------------------------------
# watershed_ift can give one label to regions nowhere near each other. Left
# alone, one "parcel" was 395 fragments spread over 29 km and its mean NDVI
# described nothing real. 4-connectivity matches how rasterio vectorises the
# labels later, so one parcel ends up as exactly one polygon.
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

n_raw = int(labels.max())
sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n_raw + 1))
keep  = np.where(sizes >= MIN_PIXELS)[0] + 1
remap = np.zeros(n_raw + 1, np.int32)
remap[keep] = np.arange(1, len(keep) + 1)
labels = remap[labels]
n = len(keep)

print(f"{len(mk_keep)} seed markers -> {n_raw} regions -> {n} parcels")
""")

# ───────────────────────── step 8: per-parcel table ───────────────────────
md(r"""
## Step 8 — Build the per-parcel table

For every parcel we record each index (mean and standard deviation), the area, a
centroid in latitude/longitude, and three quality measures: valid-pixel fraction,
cloud fraction, and the **provider's own reflectance uncertainty**.

Carrying uncertainty through is deliberate. A system that allocates public money
should never present a ranking without saying how much to trust each row.
""")

code(r"""
# Grid geometry is READ FROM THE FILE, never hardcoded. The bundled sample is
# a crop of the full scene, so it has a different origin; hardcoding the full
# scene's corner would place every sample parcel 14.75 km from where it is.
import re as _re
with h5py.File(local_path, "r") as f:
    _meta = f["HDFEOS INFORMATION/StructMetadata.0"][()].decode("utf-8", "replace")
    _epsg = int(f["HDFEOS/GRIDS/HYP"].attrs["epsg_code"])
_ul = _re.search(r"UpperLeftPointMtrs=\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", _meta)
_lr = _re.search(r"LowerRightMtrs=\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", _meta)
if not (_ul and _lr):
    raise RuntimeError("Could not read grid corners from StructMetadata.0")
ulx, uly = float(_ul.group(1)), float(_ul.group(2))
lrx, lry = float(_lr.group(1)), float(_lr.group(2))
ny_px, nx_px = ix["NDVI"].shape
res_x, res_y = (lrx - ulx) / nx_px, (uly - lry) / ny_px
transform = Affine(res_x, 0.0, ulx, 0.0, -res_y, uly)
to_wgs84  = Transformer.from_crs(f"EPSG:{_epsg}", "EPSG:4326", always_xy=True)
print(f"Grid: {res_x:.0f} m, EPSG:{_epsg}, upper-left ({ulx:.0f}, {uly:.0f})")
ids = np.arange(1, n + 1)

def parcel_stat(arr, fn=ndimage.mean):
    return np.array(fn(arr, labels, index=ids), dtype="float64")

rows = {"parcel_id": ids}
px = parcel_stat(np.ones_like(labels, dtype="float32"), ndimage.sum)
rows["n_pixels"] = px.astype(int)
rows["area_ha"]  = px * 30 * 30 / 10_000.0

for name, arr in ix.items():
    ok   = np.isfinite(arr).astype("float32")
    tot  = parcel_stat(np.where(np.isfinite(arr), arr, 0.0), ndimage.sum)
    cnt  = parcel_stat(ok, ndimage.sum)
    mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    sq   = parcel_stat(np.where(np.isfinite(arr), arr**2, 0.0), ndimage.sum)
    var  = np.where(cnt > 1, sq / np.maximum(cnt, 1) - mean**2, np.nan)
    rows[f"{name}_mean"] = mean
    rows[f"{name}_std"]  = np.sqrt(np.clip(var, 0, None))

# Parcels are BUILT from valid pixels, so validity measured inside a parcel is
# 1.0 for every parcel by construction — which tells you nothing. Measure the
# neighbourhood instead: the parcel's bounding box includes the masked pixels
# around it, so it actually varies.
_sl = ndimage.find_objects(labels)
_lv, _lc = np.full(n, np.nan), np.full(n, np.nan)
for _i, _s in enumerate(_sl):
    if _s is None:
        continue
    _lv[_i] = float(valid[_s].mean())
    _lc[_i] = float((cloud[_s] != 0).mean())
rows["local_valid_frac"] = _lv
rows["local_cloud_frac"] = _lc
u_ok = np.isfinite(U).astype("float32")
rows["refl_uncertainty"] = (parcel_stat(np.where(np.isfinite(U), U, 0.0), ndimage.sum)
                            / np.maximum(parcel_stat(u_ok, ndimage.sum), 1))

cy, cx = zip(*ndimage.center_of_mass(np.ones_like(labels), labels, ids))
east, north = transform * (np.array(cx) + 0.5, np.array(cy) + 0.5)
rows["lon"], rows["lat"] = to_wgs84.transform(east, north)

df = pd.DataFrame(rows)
print(f"{len(df)} parcels, {df['area_ha'].sum():.0f} ha total, "
      f"median {df['area_ha'].median():.1f} ha")
""")

# ───────────────────────── step 9: stress score ───────────────────────────
md(r"""
## Step 9 — The hidden-stress score

This is the core of the product, so it is worth being precise.

NDVI is roughly *what a human eye sees*. NDRE reflects *what is happening inside
the leaf*. A crop under early stress keeps a full-looking canopy while its
chlorophyll is already failing — so **the gap between the two is the early
warning**, opening weeks before anything is visible.

NDVI and NDRE sit on different scales, so subtracting them directly is
meaningless. Instead we fit the scene-wide relationship between them and measure
each parcel's **residual**:

$$\text{stress} = \text{NDRE}_{\text{observed}} - \big(a \cdot \text{NDVI} + b\big)$$

A parcel whose NDRE sits **below** what its own greenness predicts is losing
chlorophyll without losing canopy. More negative = more hidden stress.
""")

code(r"""
m = df["NDVI_mean"].notna() & df["NDRE_mean"].notna()
slope, intercept = np.polyfit(df.loc[m, "NDVI_mean"], df.loc[m, "NDRE_mean"], 1)
df["NDRE_expected"]   = slope * df["NDVI_mean"] + intercept
df["stress_residual"] = df["NDRE_mean"] - df["NDRE_expected"]
df["stress_rank"]     = df["stress_residual"].rank(method="min").astype("Int64")

# Confidence combines the three things that genuinely vary: how many pixels a
# parcel averages over, how clean the imagery around it is, and the provider's
# reflectance uncertainty. It is NOT a claim about in-parcel validity.
_u = df["refl_uncertainty"]
_u_hi = _u.quantile(0.90) if _u.notna().any() else np.inf
df["confidence"] = np.where(
    (df["n_pixels"] >= 50) & (df["local_valid_frac"] > 0.70) & (_u <= _u_hi),
    "high",
    np.where((df["n_pixels"] >= 25) & (df["local_valid_frac"] > 0.40),
             "medium", "low"))
df["needs_split"] = df["area_ha"] > BIG_PARCEL_HA

# Identify the actual geometry, independently of table order and rounding.
from run_identity import compute_run_id
shapes = {}
for geom, val in features.shapes(labels.astype("int32"), mask=labels > 0,
                                 transform=transform):
    pid = int(val)
    poly = [[list(to_wgs84.transform(x, y)) for x, y in ring]
            for ring in geom["coordinates"]]
    shapes.setdefault(pid, []).append(poly)

_identity_features = [
    {"type": "Feature", "properties": {"parcel_id": pid},
     "geometry": ({"type": "Polygon", "coordinates": polys[0]} if len(polys) == 1
                  else {"type": "MultiPolygon", "coordinates": polys})}
    for pid, polys in shapes.items()
]
RUN_ID = compute_run_id(
    SCENE_ID, _identity_features, source_crs=f"EPSG:{_epsg}",
    parameters={"NDVI_CROP_MIN": NDVI_CROP_MIN, "MIN_PIXELS": MIN_PIXELS,
                "GRAD_PCT": GRAD_PCT, "MIN_MARKER": MIN_MARKER,
                "BIG_PARCEL_HA": BIG_PARCEL_HA})
df.insert(1, "run_id", RUN_ID)
print("run id:", RUN_ID)

df = df.sort_values("stress_residual").reset_index(drop=True)
num = df.select_dtypes("number").columns
df[num] = df[num].round(4)

print(f"Scene-wide fit: NDRE = {slope:.3f} x NDVI + {intercept:.3f}")
print(f"Confidence: {(df.confidence=='high').sum()} high, "
      f"{(df.confidence=='medium').sum()} medium, {(df.confidence=='low').sum()} low")
print(f"Flagged needs_split: {int(df.needs_split.sum())}\n")
print("TOP 10 PARCELS BY HIDDEN STRESS")
print(df.head(10)[["parcel_id", "area_ha", "NDVI_mean", "NDRE_mean",
                   "stress_residual", "NDMI_mean", "BSI_mean",
                   "confidence"]].to_string(index=False))
""")

# ─────────────────────── step 10: is it signal? ───────────────────────────
md(r"""
## Step 10 — Is this signal, or noise?

A ranking is worthless if it is random. Two independent checks.

**1. Spatial coherence.** Measurement noise is spatially random. A real
field-level process clusters. We correlate each parcel's residual with the mean
of its five nearest neighbours.

**2. Physical consistency.** If the score captures real stress, the most-stressed
parcels should also be drier and barer — measured by indices that are *not* part
of the score.
""")

code(r"""
from scipy.spatial import cKDTree

work = df[(df.confidence != "low") & (~df.needs_split)].copy()
print(f"Working set: {len(work)} parcels (excludes low-confidence and merged)\n")

xy = np.c_[work.lon.values * np.cos(np.radians(work.lat.mean())), work.lat.values]
_, nn = cKDTree(xy).query(xy, k=6)
res = work["stress_residual"].values
nb_mean = res[nn[:, 1:]].mean(axis=1)
print(f"1. Spatial coherence  r = {np.corrcoef(res, nb_mean)[0,1]:+.3f}")
print("   (> 0 means clustered, i.e. a real field-level process)\n")

q = work["stress_residual"].quantile([0.1, 0.9])
lo = work[work.stress_residual <= q.iloc[0]]
hi = work[work.stress_residual >= q.iloc[1]]
print(f"2. Most-stressed decile (n={len(lo)}) vs healthiest (n={len(hi)})")
print(f"   {'index':12} {'stressed':>10} {'healthy':>10} {'diff':>10}")
for c in ["NDMI_mean", "BSI_mean", "SI_Khan_mean", "NDVI_mean"]:
    print(f"   {c:12} {lo[c].mean():10.4f} {hi[c].mean():10.4f} "
          f"{lo[c].mean()-hi[c].mean():+10.4f}")
""")

md(r"""
**Reading the result.** Stressed parcels come out drier (lower NDMI), barer
(higher BSI) and more saline-looking (higher SI1) — but only slightly less green.
NDVI separates the two groups least of all.

That is precisely the claim the project rests on: **standard greenness misses
this, and the sub-visible indicators catch it.**
""")

# ─────────────────────────── step 11: the map ─────────────────────────────
md(r"""
## Step 11 — The priority map
""")

code(r"""
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPoly

# Parcel polygons were prepared in Step 9 for the shared run identity.

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, (col, cmap, title) in zip(axes, [
        ("NDVI_mean", "RdYlGn", "Per-parcel NDVI — greenness"),
        ("stress_residual", "RdBu",
         "Hidden stress — red = NDRE below predicted")]):
    vals = df.set_index("parcel_id")[col]
    pats, cs = [], []
    for pid, polys in shapes.items():
        for poly in polys:
            pats.append(MplPoly(np.array(poly[0]), closed=True))
            cs.append(vals.get(pid, np.nan))
    pc = PatchCollection(pats, cmap=cmap, edgecolor="0.35", linewidth=0.3)
    v = np.array(cs, dtype=float)
    if col == "stress_residual":
        lim = np.nanmax(np.abs(v)); pc.set_clim(-lim, lim)
    pc.set_array(v)
    ax.add_collection(pc); ax.autoscale_view(); ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    plt.colorbar(pc, ax=ax, shrink=0.78)
plt.suptitle(f"Palm2Soil — {len(df)} parcels, {p.get('location_description','')[:40]}",
             fontsize=13)
plt.tight_layout(); plt.show()
""")

# ──────────────────── step 12: hyperspectral argument ─────────────────────
md(r"""
## Step 12 — Why hyperspectral is necessary

The hackathon guidance is to start with Sentinel-2 and reach for hyperspectral
only when it solves a **documented limitation**. This step documents it.

### First, the physics — stated correctly

> **Pure halite (NaCl) has no diagnostic absorption in the VNIR–SWIR.** Its
> crystal lattice produces no vibrational features in this range. **We do not
> claim to detect salt directly.**

What *is* detectable:
- halite is **hygroscopic**, so salt crusts produce secondary features in the
  water-absorption regions — reported near 1442, 1851, 1958 and 2226 nm
- associated evaporites such as **gypsum** have genuinely diagnostic features —
  near 1023, 1225, 1457, 1757, 1800 and 2336 nm
- reported salt-correlated windows: **1931–2123 nm** and **2153–2254 nm**

The inference is indirect, through hydration and associated minerals.

### Second, where Sentinel-2 cannot follow

We simulate Sentinel-2 by integrating this Tanager spectrum through **ESA's
measured spectral response functions** — identical ground, instant and
atmosphere, so spectral sampling is the only variable.
""")

code(r"""
import s2_srf   # loads ESA S2-SRF v5 from s2_srf_esa_v5.xlsx

def count(a, b):
    return int(((wl >= a) & (wl <= b)).sum())

wl_srf, srf = s2_srf.load("S2A")

def extent(band, thr=0.01):
    rr = srf[band] / srf[band].max()
    nz = wl_srf[rr > thr]
    return float(nz.min()), float(nz.max())

def response_in(lo, hi, bands=("B11", "B12")):
    return sum(srf[bd][(wl_srf >= lo) & (wl_srf <= hi)].sum() / srf[bd].sum()
               for bd in bands)

b11_lo, b11_hi = extent("B11")
b12_lo, b12_hi = extent("B12")

for bd in ("B11", "B12"):
    st = s2_srf.band_stats(bd, "S2A")
    print(f"  S-2 {bd}: centre {st['centre']:7.1f} nm   FWHM {st['fwhm']:3.0f} nm   "
          f"span {extent(bd)[0]:.0f}-{extent(bd)[1]:.0f} nm")

print(f"\n  {'window':26} {'Tanager':>8}   Sentinel-2")
print(f"  {'1931-2123 salt window':26} {count(1931,2123):>8}   "
      f"{response_in(1931,2123)*100:.1f}% of its SWIR response")
print(f"  {'2153-2254 salt window':26} {count(2153,2254):>8}   "
      f"{response_in(2153,2254)*100:.1f}%, all inside one band")
print(f"  {f'{b11_hi:.0f}-{b12_lo:.0f}':26} {count(b11_hi,b12_lo):>8}   "
      f"ZERO - a {b12_lo-b11_hi:.0f} nm blind gap")
print(f"  {f'{b12_hi:.0f}-2450':26} {count(b12_hi,2450):>8}   "
      f"ZERO - response has ended")
print(f"  {'1500-2500 total':26} {count(1500,2500):>8}   2")
""")

md(r"""
### Third, measured on bare soil

**Salt crusts are a bare-soil phenomenon.** Over a closed canopy the soil is
hidden and the SWIR carries leaf water, not soil chemistry. An earlier version of
this analysis measured these features over *cropped* parcels and found a weak,
largely artefactual result — a useful reminder that the physics has to drive the
pixel selection.

This has a **product consequence**: soil chemistry and crop stress must be read
from different pixels within the same farm block. It is also operationally
convenient — bare ground is most exposed between crops, which is exactly when an
intervention could be scheduled.
""")

code(r"""
sel = np.where((wl >= 1450) & (wl <= 2450))[0]
with h5py.File(local_path, "r") as f:
    srd = f[f"{ROOT}/surface_reflectance"]
    cube = np.stack([srd[i, :, :].astype("float32") for i in sel])
wls = wl[sel]
cube[:, ~valid] = np.nan
cube[cube < 0] = np.nan

bare = (np.nan_to_num(ix["NDVI"], nan=9.0) < 0.15) & valid
bare = ndimage.binary_erosion(bare, structure=full)
blab, bn = ndimage.label(bare, structure=full)
bsz = ndimage.sum(np.ones_like(blab), blab, index=np.arange(1, bn + 1))
bkeep = np.where(bsz >= 20)[0] + 1
bmap = np.zeros(bn + 1, np.int32); bmap[bkeep] = np.arange(1, len(bkeep) + 1)
blab = bmap[blab]; nb_ = len(bkeep)
bids = np.arange(1, nb_ + 1)
print(f"{nb_} bare-soil patches\n")

bspec = np.empty((nb_, len(wls)))
for i in range(len(wls)):
    a = cube[i]
    tot = ndimage.sum(np.where(np.isfinite(a), a, 0.0), blab, index=bids)
    cnt = ndimage.sum(np.isfinite(a).astype("float32"), blab, index=bids)
    bspec[:, i] = np.array(tot) / np.maximum(np.array(cnt), 1)

wv = np.zeros(len(wls), bool)                      # water-vapour bands
for a_, b_ in [(1340, 1460), (1790, 1960)]:
    wv |= (wls >= a_) & (wls <= b_)

def continuum_remove(x, y):
    hull = [0]
    for j in range(1, len(x)):
        while len(hull) >= 2:
            a_, b_ = hull[-2], hull[-1]
            if (y[b_]-y[a_])*(x[j]-x[a_]) <= (y[j]-y[a_])*(x[b_]-x[a_]):
                hull.pop()
            else:
                break
        hull.append(j)
    return y / np.maximum(np.interp(x, x[hull], y[hull]), EPS)

s2sim = {bd: bspec @ s2_srf.weights_on_grid(bd, wls, "S2A") for bd in ("B11", "B12")}
X2 = np.column_stack([s2sim["B11"], s2sim["B12"], np.ones(nb_)])

for label, (lo, hi) in {"2216 nm clay Al-OH / halite crust": (2150, 2270),
                        "2340 nm carbonate / gypsum":        (2280, 2400)}.items():
    msk = (wls >= lo) & (wls <= hi) & ~wv
    x = wls[msk]
    depth = np.array([1.0 - continuum_remove(x, bspec[i, msk]).min()
                      for i in range(nb_)])
    beta, *_ = np.linalg.lstsq(X2, depth, rcond=None)
    r2 = 1 - ((depth - X2 @ beta)**2).sum() / ((depth - depth.mean())**2).sum()
    print(f"  {label}")
    print(f"     depth {depth.mean():.4f} +- {depth.std():.4f}   "
          f"Sentinel-2 reconstructs R2={r2:.3f}  "
          f"({(1-max(r2,0))*100:.0f}% unrecoverable)")
""")

md(r"""
### What we tested and did *not* find

We also tested the **red-edge position** argument, since that is the usual
justification for hyperspectral in crop work. It did not hold up as well:
Sentinel-2 reconstructs red-edge position at R² 0.68, and its NDRE separated an
independent grouping *better* than either red-edge-position metric.

We report that because it is true. The SWIR result above is the stronger and more
honest basis for the hyperspectral claim, and the argument rests on sensor design
— band widths and gaps — which is verifiable independently of this scene.
""")

# ───────────────────────────── step 13: outputs ───────────────────────────
md(r"""
## Step 13 — Write the deliverables
""")

code(r"""
df.to_csv("parcel_table.csv", index=False)

attrs = df.set_index("parcel_id").to_dict("index")
feats = []
for pid, polys in sorted(shapes.items()):
    if pid not in attrs:
        continue
    props = {"parcel_id": pid, "run_id": RUN_ID}
    props.update({k: (None if pd.isna(v) else v) for k, v in attrs[pid].items()})
    geom = ({"type": "Polygon", "coordinates": polys[0]} if len(polys) == 1
            else {"type": "MultiPolygon", "coordinates": polys})
    feats.append({"type": "Feature", "geometry": geom, "properties": props})

with open("parcels.geojson", "w", encoding="utf-8") as f:
    json.dump({"type": "FeatureCollection", "name": "palm2soil_parcels",
               "crs": {"type": "name",
                       "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
               "features": feats}, f)

print(f"parcel_table.csv   {len(df)} rows")
print(f"parcels.geojson    {len(feats)} features")
assert len(feats) == len(df), "feature/row mismatch"
""")

md(r"""
## Provenance

| Field | Value |
|---|---|
| Product | Tanager-1 orthorectified surface reflectance (HDF5) |
| Provider | Planet Labs, open data via STAC |
| Scene | `20250608_091605_90_4001` |
| Acquired | 2025-06-08 |
| Processing | Provider surface reflectance; cloud, cirrus and no-data masks applied |
| Grid | 30 m, EPSG:32636 (UTM 36N) |
| Licence | CC-BY-4.0 |
| Sentinel-2 responses | ESA COPE-GSEG-EOPG-TN-15-0007, S2-SRF v5 |

---

## What we claim, and what we do not

**We claim**
- A spatially coherent crop-stress signal is measurable at parcel level
  (neighbour correlation r ≈ +0.23)
- It aligns with independent moisture and soil-exposure indicators, while being
  nearly invisible to NDVI
- Soil absorption features exist at 2216 and 2340 nm that Sentinel-2 cannot
  resolve — 49% and 97% of their variance is unrecoverable from its two SWIR bands
- The method is reproducible from open data end to end

**We do not claim**
- ❌ Absolute salinity in dS/m. **All ranking is relative within this scene.**
  Calibration requires paired ground measurements and is the first objective of
  the incubation phase.
- ❌ Cadastral accuracy. Parcels are segmented from imagery.
- ❌ Direct salt detection. Halite has no diagnostic absorption here.
- ❌ Transferable mineralogy. The sampling argument transfers to Al Ain because
  it concerns sensor design; the specific features present will differ.
- ❌ Calibrated salinity indices. The four published indices are verified as
  formulas (see `SALINITY_INDEX_VERIFICATION.md`) but none is tied to a measured
  dS/m value here. They are descriptive context, not the ranking mechanism.
- ❌ Temporal change. This is a single date. The verification loop — treated
  versus untreated over time — is designed but not yet demonstrated.

### Why the analysis is in Turkey

No usable UAE hyperspectral scene exists in the open Tanager archive; we
established this by auditing all 276 scenes across nine collections. The
Konya–Çumra Plain is a documented groundwater-salinisation analogue — the same
mechanism we target in Al Ain. Demonstrating on an analogue where imagery exists
is a deliberate scoping decision, stated up front.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}

for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell{i:03d}"

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"WROTE {OUT}  ({len(cells)} cells)")
