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
Sentinel-2 seasonal trajectories over the Palm2Soil parcels.

Tanager imaged Çumra exactly once (8 June 2025) — the whole open archive has no
second date over this footprint, so the verification half of the detect →
prioritise → verify loop cannot be shown with hyperspectral alone.

Sentinel-2 can: 5-day revisit, free, and 104 scenes under 20% cloud over this
footprint in 2025. This script tracks the SAME 375 parcels through the season
and asks whether the parcels Tanager flagged as stressed on 8 June behave
differently across the year.

Division of labour this demonstrates:
  Sentinel-2 — frequent broadband monitoring, detects change
  Tanager    — periodic hyperspectral, characterises soil chemistry S-2 cannot see

Produces temporal_trajectories.png and temporal_stats.txt
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio.features
import stackstac
from pyproj import Transformer

FIGDIR = os.path.join(REPO, "executed", "figures")
os.makedirs(FIGDIR, exist_ok=True)
BBOX = [32.442, 37.514, 32.706, 37.695]
EPSG = 32636
RES = 20                      # m — native for S-2 red edge and B8A
TANAGER_DATE = pd.Timestamp("2025-06-08")
WINDOW_DAYS = 10              # keep the clearest scene per window
SEASON = "2025-03-01/2025-10-31"
EPS = 1e-6
SCL_KEEP = {4, 5, 7}          # vegetation, bare soil, unclassified

INK, GREEN, ORANGE, GREY = "#14261f", "#0d3b2e", "#c2571a", "#5d6b64"
CRIM = "#a02020"


def log(m):
    print(m, flush=True)


log("1/6  searching Sentinel-2")
cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace)
items = sorted(cat.search(collections=["sentinel-2-l2a"], bbox=BBOX,
                          datetime=SEASON,
                          query={"eo:cloud_cover": {"lt": 20}}).item_collection(),
               key=lambda i: i.datetime)
log(f"     {len(items)} scenes under 20% cloud")

# thin to the clearest scene per window, so runtime stays sane
picked, bucket = [], {}
for it in items:
    key = (it.datetime.date() - pd.Timestamp("2025-03-01").date()).days // WINDOW_DAYS
    cc = it.properties["eo:cloud_cover"]
    if key not in bucket or cc < bucket[key].properties["eo:cloud_cover"]:
        bucket[key] = it
picked = [bucket[k] for k in sorted(bucket)]
log(f"     thinned to {len(picked)} dates (clearest per {WINDOW_DAYS}-day window)")

log("2/6  building the cube")
stack = stackstac.stack(picked, assets=["B03", "B04", "B05", "B8A", "SCL"],
                        resolution=RES, epsg=EPSG, bounds_latlon=BBOX,
                        chunksize=2048, rescale=False,
                        fill_value=np.float32("nan"),
                        dtype=np.dtype("float32"))
log(f"     {stack.shape}  (time, band, y, x)")

log("3/6  rasterising the parcels onto the S-2 grid")
gj = json.load(open(os.path.join(REPO, "parcels.geojson")))
to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)


def reproject(geom):
    def ring(r):
        return [list(to_utm.transform(x, y)) for x, y in r]
    if geom["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [ring(r) for r in geom["coordinates"]]}
    return {"type": "MultiPolygon",
            "coordinates": [[ring(r) for r in poly] for poly in geom["coordinates"]]}


shapes = [(reproject(f["geometry"]), f["properties"]["parcel_id"])
          for f in gj["features"]]
# Build the transform from the cube's own coordinates. `hasattr(stack,"rio")`
# is always true once rioxarray is imported, so the careful fallback below was
# never reached; stackstac's x/y are pixel CENTRES, and treating them as the
# grid corner shifted every parcel half a pixel (10 m at 20 m resolution).
import affine
xs, ys = stack.x.values.astype(float), stack.y.values.astype(float)
res_x = float(xs[1] - xs[0]) if xs.size > 1 else float(RES)
res_y = float(ys[0] - ys[1]) if ys.size > 1 else float(RES)
transform = affine.Affine(res_x, 0.0, xs[0] - res_x / 2.0,
                          0.0, -res_y, ys[0] + res_y / 2.0)
log(f"     grid {res_x:.0f} m, origin "
    f"({xs[0] - res_x / 2.0:.0f}, {ys[0] + res_y / 2.0:.0f})")
labels = rasterio.features.rasterize(
    shapes, out_shape=(stack.sizes["y"], stack.sizes["x"]),
    transform=transform, fill=0, dtype="int32")
present = np.unique(labels)
present = present[present > 0]
log(f"     {len(present)} of {len(shapes)} parcels land on the S-2 grid")

log("4/6  reading dates and computing per-parcel means")
# Processing baseline >= 04.00 carries a -1000 reflectance offset. Ignoring it
# would bias every normalised index, so it is applied explicitly per scene.
bands = list(stack.band.values)
bi = {b: bands.index(b) for b in ("B03", "B04", "B05", "B8A", "SCL")}
rows = []
for t in range(stack.sizes["time"]):
    it = picked[t]
    date = pd.Timestamp(it.datetime.date())
    baseline = str(it.properties.get("s2:processing_baseline", "05.00"))
    offset = -1000.0 if float(baseline) >= 4.0 else 0.0
    arr = stack.isel(time=t).compute().values
    scl = arr[bi["SCL"]]
    ok = np.isin(scl.astype("int16"), list(SCL_KEEP))
    def refl(b):
        v = (arr[bi[b]] + offset) / 10000.0
        v[~ok] = np.nan
        v[v <= 0] = np.nan
        return v
    g, r, re_, nir = refl("B03"), refl("B04"), refl("B05"), refl("B8A")
    ndvi = (nir - r) / (nir + r + EPS)
    ndre = (nir - re_) / (nir + re_ + EPS)
    for name, a in (("NDVI", ndvi), ("NDRE", ndre)):
        fin = np.isfinite(a)
        tot = np.bincount(labels[fin], weights=a[fin], minlength=present.max() + 1)
        cnt = np.bincount(labels[fin], minlength=present.max() + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        for pid in present:
            rows.append({"parcel_id": int(pid), "date": date, "index": name,
                         "value": float(mean[pid]), "n_px": int(cnt[pid])})
    log(f"     {date.date()}  cloud {it.properties['eo:cloud_cover']:4.1f}%  "
        f"valid {ok.mean()*100:5.1f}%")

ts = pd.DataFrame(rows)
ts = ts[ts.n_px >= 5]
wide = ts.pivot_table(index=["parcel_id", "date"], columns="index",
                      values="value").reset_index()
log(f"     {len(wide)} parcel-date observations")

log("5/6  joining the Tanager stress ranking")
tab = pd.read_csv(os.path.join(REPO, "parcel_table.csv"))
work = tab[(tab.confidence != "low") & (~tab.needs_split)]
q = work["stress_residual"].quantile([0.2, 0.8])
grp = {}
for _, r_ in work.iterrows():
    if r_.stress_residual <= q.iloc[0]:
        grp[int(r_.parcel_id)] = "flagged stressed"
    elif r_.stress_residual >= q.iloc[1]:
        grp[int(r_.parcel_id)] = "flagged healthy"
wide["group"] = wide.parcel_id.map(grp)
sub = wide.dropna(subset=["group"])
log(f"     {sub.parcel_id.nunique()} parcels in the two groups")

curves = sub.groupby(["group", "date"])[["NDVI", "NDRE"]].agg(["mean", "sem"])

# season summary per parcel
per = (wide.groupby("parcel_id")
       .agg(peak_NDVI=("NDVI", "max"),
            mean_NDVI=("NDVI", "mean"),
            n_obs=("NDVI", "size")).reset_index())
late = wide[wide.date >= pd.Timestamp("2025-08-15")]
per = per.merge(late.groupby("parcel_id")["NDVI"].mean()
                .rename("late_NDVI").reset_index(), on="parcel_id", how="left")
per["senescence_drop"] = per.peak_NDVI - per.late_NDVI
per = per.merge(tab[["parcel_id", "stress_residual", "confidence", "needs_split"]],
                on="parcel_id", how="left")
pw = per[(per.confidence != "low") & (~per.needs_split)].dropna(
    subset=["stress_residual", "peak_NDVI", "senescence_drop"])

corr = {c: pw["stress_residual"].corr(pw[c])
        for c in ("peak_NDVI", "mean_NDVI", "late_NDVI", "senescence_drop")}
for k, v in corr.items():
    log(f"     corr(Tanager stress residual, {k:16}) = {v:+.3f}")

log("6/6  drawing")
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.34, wspace=0.26)

ax = fig.add_subplot(gs[0, :])
for g_, col in (("flagged stressed", CRIM), ("flagged healthy", GREEN)):
    if g_ not in curves.index.get_level_values(0):
        continue
    c = curves.loc[g_]
    m, e = c[("NDVI", "mean")], c[("NDVI", "sem")]
    ax.plot(c.index, m, "-o", color=col, ms=4, lw=1.8, label=f"{g_} (NDVI)")
    ax.fill_between(c.index, m - 1.96 * e, m + 1.96 * e, color=col, alpha=0.18)
ax.axvline(TANAGER_DATE, color=ORANGE, lw=2, ls="--")
ax.annotate("Tanager acquisition\n8 June 2025 — the only one",
            (TANAGER_DATE, ax.get_ylim()[1] * 0.97), color=ORANGE, fontsize=9,
            ha="center", va="top",
            bbox=dict(fc="white", ec=ORANGE, alpha=0.9, boxstyle="round,pad=0.3"))
ax.set_ylabel("NDVI")
ax.set_xlabel("")
ax.set_title("A  Seasonal NDVI from Sentinel-2, split by the Tanager stress ranking "
             "(shaded = 95% CI)", fontsize=12, color=INK, loc="left")
ax.legend(frameon=False, fontsize=9.5)
ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[1, 0])
for g_, col in (("flagged stressed", CRIM), ("flagged healthy", GREEN)):
    if g_ not in curves.index.get_level_values(0):
        continue
    c = curves.loc[g_]
    ax.plot(c.index, c[("NDRE", "mean")], "-o", color=col, ms=3.5, lw=1.6, label=g_)
ax.axvline(TANAGER_DATE, color=ORANGE, lw=1.6, ls="--")
ax.set_ylabel("NDRE")
ax.set_title("B  NDRE through the season", fontsize=11.5, color=INK, loc="left")
ax.tick_params(axis="x", rotation=30)
ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[1, 1])
ax.scatter(pw.stress_residual, pw.peak_NDVI, s=14, color=GREY, alpha=0.75)
if len(pw) > 2:
    z = np.polyfit(pw.stress_residual, pw.peak_NDVI, 1)
    xs_ = np.linspace(pw.stress_residual.min(), pw.stress_residual.max(), 20)
    ax.plot(xs_, np.polyval(z, xs_), color=CRIM, lw=1.6)
ax.set_xlabel("Tanager stress residual (8 June)")
ax.set_ylabel("Sentinel-2 peak NDVI (season)")
ax.set_title(f"C  Does one date predict the season?\nr = {corr['peak_NDVI']:+.3f}",
             fontsize=11.5, color=INK, loc="left")
ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[1, 2])
names = list(corr)
vals = [corr[k] for k in names]
cols = [CRIM if abs(v) > 0.3 else GREY for v in vals]
ax.barh(range(len(names)), vals, color=cols, alpha=0.9)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9)
ax.axvline(0, color="k", lw=0.8)
for i, v in enumerate(vals):
    ax.text(v + (0.015 if v >= 0 else -0.015), i, f"{v:+.2f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=9)
ax.set_xlim(-max(0.5, max(abs(v) for v in vals) * 1.4),
            max(0.5, max(abs(v) for v in vals) * 1.4))
ax.set_xlabel("correlation with Tanager stress residual")
ax.set_title("D  Seasonal behaviour vs the single-date score",
             fontsize=11.5, color=INK, loc="left")
ax.grid(alpha=0.25, axis="x")

fig.suptitle("Closing the loop: Sentinel-2 supplies the time axis Tanager cannot",
             fontsize=14.5, color=INK, x=0.012, ha="left", y=0.985)
fig.text(0.012, 0.945,
         f"{len(picked)} Sentinel-2 dates, Mar–Oct 2025, over the same "
         f"{sub.parcel_id.nunique()} parcels. Tanager imaged this footprint once "
         "in the entire open archive.", fontsize=9.4, color=GREY, ha="left")
out = os.path.join(FIGDIR, "temporal_trajectories.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")

ts.to_csv(os.path.join(REPO, "parcel_timeseries.csv"), index=False)

txt = f"""Sentinel-2 seasonal trajectories over the Palm2Soil parcels
{'=' * 66}
Why this exists
  Tanager imaged Çumra ONCE (8 June 2025). Across all 276 scenes in the open
  Tanager archive, no second acquisition covers this footprint. The
  verification stage of the detect-prioritise-verify loop therefore cannot be
  demonstrated with hyperspectral data alone.

  Sentinel-2 can: {len(items)} scenes under 20% cloud over this footprint between
  March and October 2025, thinned here to {len(picked)} dates (clearest per
  {WINDOW_DAYS}-day window).

Data
  Sentinel-2 L2A via Microsoft Planetary Computer, tile 36SVG, {RES} m
  Bands B03, B04, B05, B8A; SCL used for masking (kept classes {sorted(SCL_KEEP)})
  Reflectance offset applied for processing baseline >= 04.00
  Parcels: the same {len(shapes)} polygons, {len(present)} of which land on the grid
  Observations: {len(wide)} parcel-date records

Do the Tanager-flagged parcels behave differently across the season?
  Groups: lowest and highest quintile of the 8 June stress residual
  ({sub.parcel_id.nunique()} parcels total)

  Correlation of the single-date Tanager stress residual with:
"""
for k, v in corr.items():
    txt += f"    {k:18} r = {v:+.3f}\n"
txt += f"""
INTERPRETATION
  A positive correlation with peak or mean NDVI means parcels scored as more
  stressed on 8 June were also less vigorous across the whole season -- i.e.
  the single hyperspectral date carries information about the season, not just
  about that morning.
  A correlation near zero would mean the 8 June score is specific to that date
  and should not be read as a seasonal property. Either result is useful and
  both are reported here without selection.

WHAT THIS DEMONSTRATES
  The same parcels can be tracked through time from freely available imagery.
  Change detection works at parcel level.
  The monitoring loop is operational, not hypothetical.

WHAT IT DOES NOT DEMONSTRATE
  No treatment occurred, so this is not a treatment effect. It is a capability
  demonstration of the verification stage.
  Sentinel-2 cannot measure the soil absorption features that justify
  hyperspectral (see swir_stats.txt). The two sensors do different jobs:
  Sentinel-2 provides the time axis, Tanager the soil chemistry.

CAVEATS
  Sentinel-2 is {RES} m where the parcels were cut from a 30 m grid, so parcel
  edges are resampled and small parcels carry fewer pixels. Observations with
  fewer than 5 valid pixels are dropped.
  Cloud masking relies on the provider's SCL layer.
  One season, one region, no ground truth.
"""
with open(os.path.join(REPO, "temporal_stats.txt"), "w") as f:
    f.write(txt)
print()
print(txt)
log(f"wrote {out}")
log(f"wrote parcel_timeseries.csv ({len(ts)} rows)")
