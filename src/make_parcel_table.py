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
Per-parcel index table for the Palm2Soil proof of concept.

Takes the Tanager surface-reflectance scene the starter notebook uses, segments
it into cropped field parcels, and writes one row per parcel with vegetation and
salinity indices, a hidden-stress score, and per-parcel data-quality figures.

Outputs
  parcel_table.csv        one row per parcel, ranked by hidden-stress score
  parcels.geojson         same attributes with polygons, WGS84, for the dashboard
  parcel_summary.txt      a short text digest

IMPORTANT LIMITATION: parcels are derived by segmenting the vegetation mask in
the imagery itself, NOT from a cadastral field-boundary dataset. They approximate
real management units but are not registered farm blocks. State this in the
submission.
"""
import json

import h5py
import numpy as np
import pandas as pd
import requests
from affine import Affine
from pyproj import Transformer
from rasterio import features
from scipy import ndimage

import s2_srf   # ESA measured Sentinel-2 response functions

SCENE_ID = "20250608_091605_90_4001"
COLLECTION = "agriculture"
H5 = os.path.join(REPO, f"{SCENE_ID}_ortho_sr_hdf5.h5")
ROOT = "HDFEOS/GRIDS/HYP/Data Fields"

# segmentation controls
NDVI_CROP_MIN = 0.35   # a pixel counts as actively cropped above this
MIN_PIXELS = 20        # drop parcels smaller than this AFTER erosion (~1.8 ha)
GRAD_PCT = 50          # gradient percentile below which a pixel is "field interior"
MIN_MARKER = 4         # smallest seed marker, in pixels
BIG_PARCEL_HA = 50     # above this, flag as probably several fields merged
EPS = 1e-6


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- wavelengths
def get_wavelengths():
    """Band centres in nm, from the STAC item (same source the notebook uses)."""
    url = (f"https://www.planet.com/data/stac/tanager-core-imagery/"
           f"{COLLECTION}/{SCENE_ID}/{SCENE_ID}.json")
    # Prefer the bundled copy so this works offline and makes no needless
    # network call; fall back to Planet's catalogue if it is absent.
    _local = os.path.join(REPO, "examples", "input",
                          f"stac_item_{SCENE_ID}.json")
    if os.path.exists(_local):
        with open(_local, encoding="utf-8") as _f:
            item = json.load(_f)
    else:
        item = requests.get(url, timeout=120).json()
    key = "ortho_sr_hdf5" if "ortho_sr_hdf5" in item["assets"] else "basic_sr_hdf5"
    bands = [b for b in item["assets"][key].get("bands", [])
             if "eo:center_wavelength" in b]
    wl = np.array([b["eo:center_wavelength"] for b in bands], dtype=float) * 1000.0
    return wl, item


def pick(wl, target):
    i = int(np.argmin(np.abs(wl - target)))
    return i, float(wl[i])


# ---------------------------------------------------------------------- main
log("1/7  reading wavelengths from STAC")
wl, item = get_wavelengths()
log(f"     {len(wl)} bands, {wl[0]:.0f}-{wl[-1]:.0f} nm")

# Targets: the first five match the starter notebook exactly so the numbers are
# comparable; blue and swir2 are added for the salinity indices.
TARGETS = {"blue": 490, "green": 570, "red": 665, "rededge": 705,
           "nir": 860, "swir1": 1650, "swir2": 2200}
idx = {}
log("2/7  selecting bands")
for name, t in TARGETS.items():
    i, actual = pick(wl, t)
    idx[name] = i
    log(f"     {name:8} target {t:>5} nm -> band {i:3d} ({actual:.1f} nm)")

log("3/7  reading masks and reflectance")
# The VNIR window is needed as a block so the published salinity indices can be
# computed on BROADBAND-equivalent reflectance -- see note below.
vnir_sel = np.where((wl >= 430) & (wl <= 920))[0]
with h5py.File(H5, "r") as f:
    cloud = f[f"{ROOT}/beta_cloud_mask"][:]
    nodata = f[f"{ROOT}/nodata_pixels"][:]
    cirrus = f[f"{ROOT}/beta_cirrus_mask"][:]
    sr = f[f"{ROOT}/surface_reflectance"]
    unc = f[f"{ROOT}/surface_reflectance_uncertainty"]
    R = {n: sr[i, :, :].astype("float32") for n, i in idx.items()}
    vnir = np.stack([sr[i, :, :].astype("float32") for i in vnir_sel])
    # provider uncertainty, averaged over the bands actually used
    U = np.nanmean(np.stack([unc[i, :, :].astype("float32")
                             for i in idx.values()]), axis=0)
wl_vnir = wl[vnir_sel]
vnir[:, (nodata != 0) | (cloud != 0) | (cirrus != 0)] = np.nan
vnir[vnir < 0] = np.nan

valid = (nodata == 0) & (cloud == 0) & (cirrus == 0)
log(f"     scene {cloud.shape},  valid {valid.mean()*100:.1f}%,  "
    f"cloud {(cloud != 0).mean()*100:.1f}%")

for a in R.values():
    a[~valid] = np.nan
    a[a < 0] = np.nan
U[~valid] = np.nan

log("4/7  computing indices")
b, g, r = R["blue"], R["green"], R["red"]
re_, nir = R["rededge"], R["nir"]
sw1, sw2 = R["swir1"], R["swir2"]

ix = {}
# vegetation / moisture / soil - identical definitions to the starter notebook
ix["NDVI"] = (nir - r) / (nir + r + EPS)
ix["NDRE"] = (nir - re_) / (nir + re_ + EPS)
ix["NDMI"] = (nir - sw1) / (nir + sw1 + EPS)
ix["BSI"] = ((sw1 + r) - (nir + g)) / ((sw1 + r) + (nir + g) + EPS)

# --- published salinity indices -------------------------------------------
# VERIFIED. Formulas and attributions cross-checked against the CRAN
# SoilSaltIndex package documentation and Rafik et al. (Remote Sensing, 2022).
# See SALINITY_INDEX_VERIFICATION.md for sources and for why the names used in
# the literature are ambiguous.
#
# These indices were all developed for BROADBAND sensors (Landsat, SPOT,
# IRS-1B, Sentinel-2). A single 5 nm Tanager band is not the same measurement
# as a 30-90 nm satellite band, so we integrate Tanager through ESA's measured
# Sentinel-2 response functions first. Difference from naive single-band use is
# small (r >= 0.996) but this version is the one that can be cited.
def broadband(band):
    w = s2_srf.weights_on_grid(band, wl_vnir, "S2A")
    return np.tensordot(w, vnir, axes=(0, 0))


bb = {k: broadband(v) for k, v in
      {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8A"}.items()}
bg, bb_, br, bn = bb["green"], bb["blue"], bb["red"], bb["nir"]

with np.errstate(invalid="ignore"):
    # SI_Khan  = sqrt(Green x Red)             Khan et al.
    ix["SI_Khan"] = np.sqrt(np.clip(bg * br, 0, None))
    # SI_Douaoui = sqrt(G^2 + R^2 + NIR^2)     Douaoui et al. 2006
    ix["SI_Douaoui"] = np.sqrt(np.clip(bg ** 2 + br ** 2 + bn ** 2, 0, None))
    # BI = sqrt(R^2 + NIR^2)                   Brightness Index
    ix["BI"] = np.sqrt(np.clip(br ** 2 + bn ** 2, 0, None))
    # CRSI = sqrt(((NIR x R) - (G x B)) / ((NIR x R) + (G x B)))
    #                                          Scudiero et al. 2014/2015
    ix["CRSI"] = np.sqrt(np.clip((bn * br - bg * bb_) / (bn * br + bg * bb_ + EPS),
                                 0, None))

log("5/7  segmenting cropped parcels")
# Adjacent fields are spectrally contiguous, so simple connected components
# merges whole farming districts into one blob (a 866 ha "parcel" in testing).
# Split on the NDVI gradient instead: field interiors are flat, boundaries are
# edges. Markers come from flat interiors; watershed grows them to the edges.
full = np.ones((3, 3), bool)
crop = (np.nan_to_num(ix["NDVI"], nan=-9.0) > NDVI_CROP_MIN) & valid
eroded = ndimage.binary_erosion(crop, structure=full)   # drop mixed edge pixels

nd = np.nan_to_num(ix["NDVI"], nan=0.0)
sm = ndimage.uniform_filter(nd, size=3)
grad = np.hypot(ndimage.sobel(sm, axis=0), ndimage.sobel(sm, axis=1))
g_in = grad[eroded]

flat = (grad < np.percentile(g_in, GRAD_PCT)) & eroded
mk, n_mk = ndimage.label(flat, structure=full)
mk_sz = ndimage.sum(np.ones_like(mk), mk, index=np.arange(1, n_mk + 1))
mk_keep = np.where(mk_sz >= MIN_MARKER)[0] + 1
mk_remap = np.zeros(n_mk + 1, dtype=np.int32)
mk_remap[mk_keep] = np.arange(1, len(mk_keep) + 1)
mk = mk_remap[mk]
log(f"     {len(mk_keep)} seed markers from flat field interiors")

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
keep = np.where(sizes >= MIN_PIXELS)[0] + 1
log(f"     {n_raw} regions -> {len(keep)} parcels at >= {MIN_PIXELS} px")

# renumber kept parcels 1..N
remap = np.zeros(n_raw + 1, dtype=np.int32)
remap[keep] = np.arange(1, len(keep) + 1)
labels = remap[labels]
n = len(keep)

log("6/7  building per-parcel statistics")
# Read the grid from the file rather than hardcoding it: a cropped scene has a
# different origin, and a hardcoded corner would misplace every parcel.
import re as _re
with h5py.File(H5, "r") as _f:
    _meta = _f["HDFEOS INFORMATION/StructMetadata.0"][()].decode("utf-8", "replace")
    _epsg = int(_f["HDFEOS/GRIDS/HYP"].attrs["epsg_code"])
_ul = _re.search(r"UpperLeftPointMtrs=\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", _meta)
_lr = _re.search(r"LowerRightMtrs=\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", _meta)
if not (_ul and _lr):
    raise RuntimeError("Could not read grid corners from StructMetadata.0")
_ulx, _uly = float(_ul.group(1)), float(_ul.group(2))
_lrx, _lry = float(_lr.group(1)), float(_lr.group(2))
_ny, _nx = labels.shape
transform = Affine((_lrx - _ulx) / _nx, 0.0, _ulx,
                   0.0, -(_uly - _lry) / _ny, _uly)
log(f"     grid {(_lrx-_ulx)/_nx:.0f} m, EPSG:{_epsg}, UL ({_ulx:.0f}, {_uly:.0f})")
to_wgs84 = Transformer.from_crs(f"EPSG:{_epsg}", "EPSG:4326", always_xy=True)
lab_ids = np.arange(1, n + 1)


def per_parcel(arr, fn=ndimage.mean):
    return np.array(fn(arr, labels, index=lab_ids), dtype="float64")


rows = {"parcel_id": lab_ids}
px = per_parcel(np.ones_like(labels, dtype="float32"), ndimage.sum)
rows["n_pixels"] = px.astype(int)
rows["area_ha"] = px * 30 * 30 / 10_000.0

for name, arr in ix.items():
    a = np.where(np.isfinite(arr), arr, 0.0)
    ok = np.isfinite(arr).astype("float32")
    tot = per_parcel(a, ndimage.sum)
    cnt = per_parcel(ok, ndimage.sum)
    mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    rows[f"{name}_mean"] = mean
    sq = per_parcel(np.where(np.isfinite(arr), arr ** 2, 0.0), ndimage.sum)
    var = np.where(cnt > 1, sq / np.maximum(cnt, 1) - mean ** 2, np.nan)
    rows[f"{name}_std"] = np.sqrt(np.clip(var, 0, None))

# data quality per parcel
# Data quality, measured honestly.
#
# Parcels are BUILT from valid pixels, so validity measured inside a parcel is
# 1.0 by construction for every parcel -- which is what the earlier version
# reported, and it was meaningless. Measure the neighbourhood instead: the
# parcel's bounding box, which includes the masked pixels around it and so
# actually varies.
sl = ndimage.find_objects(labels)
loc_valid = np.full(n, np.nan)
loc_cloud = np.full(n, np.nan)
for i, sel_ in enumerate(sl):
    if sel_ is None:
        continue
    loc_valid[i] = float(valid[sel_].mean())
    loc_cloud[i] = float((cloud[sel_] != 0).mean())
rows["local_valid_frac"] = loc_valid      # valid share of the bounding box
rows["local_cloud_frac"] = loc_cloud      # cloud share of the bounding box
u = np.where(np.isfinite(U), U, 0.0)
uok = np.isfinite(U).astype("float32")
rows["refl_uncertainty"] = (per_parcel(u, ndimage.sum)
                            / np.maximum(per_parcel(uok, ndimage.sum), 1))

# centroids -> lon/lat
cy, cx = zip(*ndimage.center_of_mass(np.ones_like(labels), labels, lab_ids))
east, north = transform * (np.array(cx) + 0.5, np.array(cy) + 0.5)
lon, lat = to_wgs84.transform(east, north)
rows["lon"] = lon
rows["lat"] = lat

df = pd.DataFrame(rows)

# ---- hidden-stress score -------------------------------------------------
# NDVI and NDRE live on different scales, so a raw difference is meaningless.
# Fit the scene-wide NDRE~NDVI relationship and take the residual: a parcel
# whose NDRE sits BELOW what its greenness predicts is losing chlorophyll while
# the canopy still looks full. More negative = more hidden stress.
m = df["NDVI_mean"].notna() & df["NDRE_mean"].notna()
slope, intercept = np.polyfit(df.loc[m, "NDVI_mean"], df.loc[m, "NDRE_mean"], 1)
expected = slope * df["NDVI_mean"] + intercept
df["NDRE_expected"] = expected
df["stress_residual"] = df["NDRE_mean"] - expected
df["stress_rank"] = df["stress_residual"].rank(method="min").astype("Int64")

# Confidence combines the three things that genuinely vary per parcel: how
# many pixels it averages over, how clean the surrounding imagery is, and the
# provider's own reflectance uncertainty. It is NOT a statement about
# in-parcel validity, which is 1.0 for every parcel by construction.
_u = df["refl_uncertainty"]
_u_hi = _u.quantile(0.90) if _u.notna().any() else np.inf
df["confidence"] = np.where(
    (df["n_pixels"] >= 50) & (df["local_valid_frac"] > 0.70) & (_u <= _u_hi),
    "high",
    np.where((df["n_pixels"] >= 25) & (df["local_valid_frac"] > 0.40),
             "medium", "low"))

# Honest flag: without cadastral boundaries some neighbouring fields stay merged.
# Anything far above the median parcel size is probably several fields.
df["needs_split"] = df["area_ha"] > BIG_PARCEL_HA

df = df.sort_values("stress_residual").reset_index(drop=True)

num = df.select_dtypes("number").columns
df[num] = df[num].round(4)

# Identify the same actual geometry as the notebook, independent of row order.
from run_identity import compute_run_id
# Polygons for the dashboard choropleth. rasterio vectorises with 4-connectivity
# while the labels are 8-connected, so one parcel can yield several rings --
# collect them per id and emit a single MultiPolygon, or the GeoJSON ends up
# with duplicate parcel_ids and the dashboard double-counts.
rings_by_id = {}
for geom, val in features.shapes(labels.astype("int32"), mask=labels > 0,
                                 transform=transform):
    pid = int(val)
    poly = [[list(to_wgs84.transform(x, y)) for x, y in ring]
            for ring in geom["coordinates"]]
    rings_by_id.setdefault(pid, []).append(poly)

_identity_features = [
    {"type": "Feature", "properties": {"parcel_id": pid},
     "geometry": ({"type": "Polygon", "coordinates": polys[0]} if len(polys) == 1
                  else {"type": "MultiPolygon", "coordinates": polys})}
    for pid, polys in rings_by_id.items()
]
RUN_ID = compute_run_id(
    SCENE_ID, _identity_features, source_crs=f"EPSG:{_epsg}",
    parameters={"NDVI_CROP_MIN": NDVI_CROP_MIN, "MIN_PIXELS": MIN_PIXELS,
                "GRAD_PCT": GRAD_PCT, "MIN_MARKER": MIN_MARKER,
                "BIG_PARCEL_HA": BIG_PARCEL_HA})
df.insert(1, "run_id", RUN_ID)
log(f"     run id {RUN_ID}")

log("7/7  writing outputs")
csv_path = os.path.join(REPO, "parcel_table.csv")
df.to_csv(csv_path, index=False)


geoms = []
attrs = df.set_index("parcel_id").to_dict("index")
for pid, polys in sorted(rings_by_id.items()):
    if pid not in attrs:
        continue
    props = {"parcel_id": pid, "run_id": RUN_ID}
    props.update({k: (None if pd.isna(v) else v) for k, v in attrs[pid].items()})
    geometry = ({"type": "Polygon", "coordinates": polys[0]} if len(polys) == 1
                else {"type": "MultiPolygon", "coordinates": polys})
    geoms.append({"type": "Feature", "geometry": geometry, "properties": props})
assert len(geoms) == len(df), f"{len(geoms)} features vs {len(df)} parcels"
gj_path = os.path.join(REPO, "parcels.geojson")
with open(gj_path, "w", encoding="utf-8") as f:
    json.dump({"type": "FeatureCollection",
               "name": "palm2soil_parcels",
               "crs": {"type": "name",
                       "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
               "features": geoms}, f)

p = item["properties"]
summary = f"""Palm2Soil - per-parcel index table
{'=' * 58}
Run id       {RUN_ID}
Scene        {SCENE_ID}
Location     {p.get('location_description')}
Acquired     {p.get('datetime')}
Sensor       Planet Tanager-1, {len(wl)} bands, {wl[0]:.0f}-{wl[-1]:.0f} nm
Grid         30 m, EPSG:32636 (UTM 36N)
Licence      {p.get('license')}
Scene cloud  {p.get('cloud_percent')}%   valid pixels {valid.mean()*100:.1f}%

Parcels      {n} cropped parcels >= {MIN_PIXELS} px ({MIN_PIXELS*0.09:.1f} ha)
Total area   {df['area_ha'].sum():.0f} ha
Parcel size  median {df['area_ha'].median():.1f} ha, \
range {df['area_ha'].min():.1f}-{df['area_ha'].max():.1f} ha

Confidence   {(df['confidence'] == 'high').sum()} high, \
{(df['confidence'] == 'medium').sum()} medium, {(df['confidence'] == 'low').sum()} low
Needs split  {int(df['needs_split'].sum())} parcels over {BIG_PARCEL_HA} ha \
(probably several fields merged - exclude or split before use)

Scene-wide NDRE ~ NDVI fit: NDRE = {slope:.3f} * NDVI + {intercept:.3f}
Hidden-stress residual: mean {df['stress_residual'].mean():.4f}, \
sd {df['stress_residual'].std():.4f}, \
range {df['stress_residual'].min():.4f} to {df['stress_residual'].max():.4f}

Index means across parcels
"""
for name in ix:
    c = f"{name}_mean"
    summary += (f"  {name:5} mean {df[c].mean():7.4f}   "
                f"range {df[c].min():7.4f} to {df[c].max():7.4f}\n")

summary += f"""
Top 10 parcels by hidden stress (NDRE below the level its NDVI predicts)
{'-' * 58}
"""
top = df.head(10)[["parcel_id", "area_ha", "NDVI_mean", "NDRE_mean",
                   "stress_residual", "NDMI_mean", "BSI_mean", "confidence"]]
summary += top.to_string(index=False)
summary += f"""

CAVEATS
  Parcels are segmented from the imagery, not from cadastral boundaries.
  Salinity indices are VERIFIED against published sources and computed on
  broadband-equivalent reflectance - see SALINITY_INDEX_VERIFICATION.md.
  Severity is RELATIVE within this scene - no absolute dS/m calibration
  exists without ground truth.
"""
sum_path = os.path.join(REPO, "parcel_summary.txt")
with open(sum_path, "w", encoding="utf-8") as f:
    f.write(summary)

print()
print(summary)
log(f"wrote {csv_path}")
log(f"wrote {gj_path}  ({len(geoms)} polygons)")
log(f"wrote {sum_path}")
