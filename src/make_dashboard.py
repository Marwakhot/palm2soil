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
Build Palm2Soil_Dashboard.html from the real parcel data.

Replaces the prototype's six invented Al Ain blocks and CSS-illustration map
with the 375 parcels actually measured, drawn as a geographic choropleth.

Design decisions worth knowing:
  * Self-contained. Data is embedded and the map is inline SVG, so the file
    opens from disk with no network and no CDN. A submission reviewer may be
    offline.
  * No fabricated records. The prototype carried invented Emirati names, phone
    numbers and email addresses for six farms. Those are gone; a provenance and
    data-quality panel takes that slot, which the onboarding checklist asks for
    anyway.
  * Choropleth, not heatmap. A heatmap implies a smooth interpolated surface.
    The finding is that variation is per-parcel with hard boundaries.
  * Relative ranking only. Nothing is labelled in dS/m, because nothing is
    calibrated to dS/m.
"""
import argparse
import json
import math

import pandas as pd

from dataset import resolve_dataset

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--data-dir", default=None,
                help="folder holding parcel_table.csv and parcels.geojson "
                     "(default: repo root, else examples/output)")
ap.add_argument("-o", "--out", default=None, help="output HTML path")
args = ap.parse_args()

CSV_PATH, GJ_PATH, DATASET = resolve_dataset(args.data_dir, REPO, need_geojson=True)
OUT = args.out or os.path.join(REPO, "Palm2Soil_Dashboard.html")
print(f"dataset: {DATASET}  ({os.path.dirname(CSV_PATH)})")

SCENE = {
    "id": "20250608_091605_90_4001",
    "place": "Çumra, Konya, Türkiye",
    "date": "8 June 2025",
    "sensor": "Planet Tanager-1 · 426 bands · 376–2499 nm",
    "grid": "30 m · EPSG:32636 (UTM 36N)",
    "licence": "CC-BY-4.0",
    "cloud": "2%",
}

LAYERS = [
    ("stress_residual", "Review priority", "Canopy–chlorophyll divergence on the "
     "date observed. Lower (red) = NDRE further below the level this parcel's "
     "own greenness predicts.", "diverging", False),
    ("NDVI_mean", "Greenness (NDVI)", "General vegetation vigour.", "green", False),
    ("NDRE_mean", "Chlorophyll (NDRE)", "Red-edge index; chlorophyll condition "
     "inside the leaf.", "green", False),
    ("NDMI_mean", "Canopy water (NDMI)", "Leaf water content. Salt stress limits "
     "water uptake even in wet soil.", "water", False),
    ("BSI_mean", "Bare soil (BSI)", "Exposed soil surface. Salt crusts read as "
     "bright bare soil.", "soil", False),
    ("SI_Khan_mean", "Salinity index", "Khan et al. salinity index on "
     "broadband-equivalent reflectance. Not calibrated to dS/m.", "soil", False),
]


def load():
    gj = json.load(open(GJ_PATH, encoding="utf-8"))
    tab = pd.read_csv(CSV_PATH)
    keep = ["parcel_id", "area_ha", "n_pixels", "lon", "lat", "confidence",
            "needs_split", "local_valid_frac", "local_cloud_frac", "refl_uncertainty",
            "stress_residual", "stress_rank", "run_id"] + [c for c, *_ in LAYERS
                                                 if c != "stress_residual"]
    tab = tab[[c for c in keep if c in tab.columns]]
    attrs = {int(r.parcel_id): r for r in tab.itertuples()}

    def rnd(o, n=5):
        if isinstance(o, list):
            return [rnd(x, n) for x in o]
        if isinstance(o, float):
            return round(o, n)
        return o

    feats = []
    for f in gj["features"]:
        pid = int(f["properties"]["parcel_id"])
        if pid not in attrs:
            continue
        a = attrs[pid]
        # Keep EVERY ring, including interior holes. Taking only r[0] from a
        # MultiPolygon dropped 16 holes across 11 parcels.
        if f["geometry"]["type"] == "Polygon":
            rings = list(f["geometry"]["coordinates"])
        else:
            rings = [r for poly in f["geometry"]["coordinates"] for r in poly]
        rings = [rnd(r) for r in rings]
        props = {"id": pid}
        for c in tab.columns:
            if c == "parcel_id":
                continue
            v = getattr(a, c)
            props[c] = (None if pd.isna(v) else
                        (bool(v) if isinstance(v, bool) else
                         (round(float(v), 4) if isinstance(v, (int, float)) else v)))
        feats.append({"p": props, "r": rings})
    return feats, tab


feats, tab = load()
RUN_ID = str(tab["run_id"].iloc[0]) if "run_id" in tab.columns else "unknown"
IS_SAMPLE = len(tab) < 100
DATA_NOTE = ("bundled sample — 3.8 km crop" if IS_SAMPLE else "full scene")
work = tab[(tab.confidence != "low") & (~tab.needs_split)]
# Keep rejected parcels out of the eligible count and priority thresholds.
# Their measurements can still be inspected on the other map layers.
NO_ELIGIBLE = work.empty
if NO_ELIGIBLE:
    print("  warning: no parcel passes the quality filter; ranking is unavailable")
DATA = json.dumps(feats, separators=(",", ":"))
LAYERS_JS = json.dumps([{"key": k, "label": l, "note": n, "ramp": r, "rev": rv}
                        for k, l, n, r, rv in LAYERS], separators=(",", ":"))

def _finite(v, default=None):
    """Never let NaN or inf reach the generated JavaScript."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


stats = {
    "total": len(tab),
    "usable": len(work),
    "area": f"{tab.area_ha.sum():,.0f}",
    "median": f"{tab.area_ha.median():.1f}",
    "high": int((tab.confidence == "high").sum()),
    "med": int((tab.confidence == "medium").sum()),
    "low": int((tab.confidence == "low").sum()),
    "split": int(tab.needs_split.sum()),
    "p20": _finite(work.stress_residual.quantile(0.2)),
    "p40": _finite(work.stress_residual.quantile(0.4)),
}

DATASET_LABEL = DATA_NOTE
RANKING_DESCRIPTION = ("Inspection ranking is unavailable until parcels pass the quality checks."
                       if NO_ELIGIBLE else "Ranked for inspection priority.")
NO_ELIGIBLE_BANNER = ("" if not NO_ELIGIBLE else
    '<div class="banner" style="border-left-color:#a02020;background:#fdeeeb">'
    '<b>No eligible parcels.</b>&nbsp;Every parcel in this dataset was filtered '
    'out as low-confidence or probably-merged. Inspection ranking is unavailable. '
    'You can inspect measurements on the other map layers; review data quality '
    'or parcel boundaries before requesting a new ranking.'
    '</div>')
HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0d3d37">
<title>Palm2Soil — Remediation Priority Dashboard</title>
<style>
:root {{
  --ink:#14332e; --deep:#0a3731; --green:#168061; --mint:#aee2c5;
  --mint-soft:#ecfaf1; --page:#f2f6f4; --paper:#fff; --line:#d9e2dd;
  --muted:#6c7d76; --coral:#e86955; --amber:#e7a541; --blue:#3385a3;
  --shadow:0 12px 34px rgba(23,58,51,.09);
}}
*{{box-sizing:border-box}}
body{{margin:0;min-width:320px;color:var(--ink);background:var(--page);
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
button{{font:inherit;cursor:pointer}} h1,h2,h3,p{{margin:0}}
.dashboard{{display:grid;grid-template-columns:238px minmax(0,1fr);min-height:100vh}}
.side-nav{{display:flex;flex-direction:column;min-height:100vh;padding:22px 15px;
  color:#e7f4ed;background:linear-gradient(180deg,var(--deep),#0d5447)}}
.brand-row{{display:flex;align-items:center;gap:10px;padding:0 9px 22px}}
.brand-mark{{display:grid;place-items:center;width:34px;height:34px;border-radius:11px;
  color:var(--deep);background:var(--mint);font-size:19px;font-weight:900}}
.brand{{font-size:20px;font-weight:800;letter-spacing:-.05em}}
.brand small{{display:block;margin-top:2px;color:#a8c9bd;font-size:10px;font-weight:700;
  letter-spacing:.08em;text-transform:uppercase}}
.nav-label{{padding:0 11px;color:#a8c9bd;font-size:10px;font-weight:750;
  letter-spacing:.11em;text-transform:uppercase}}
.layer-list{{display:grid;gap:4px;margin:10px 0 18px}}
.layer-btn{{display:block;width:100%;padding:9px 11px;border:0;border-radius:11px;
  color:#d2e7de;background:transparent;text-align:left;font-size:13px;line-height:1.3}}
.layer-btn:hover{{background:rgba(255,255,255,.1)}}
.layer-btn[aria-pressed=true]{{color:#fff;background:rgba(255,255,255,.15);font-weight:700}}
.side-bottom{{margin-top:auto;padding:14px;border:1px solid rgba(199,241,215,.23);
  border-radius:14px;background:rgba(255,255,255,.08)}}
.side-bottom b{{display:block;color:var(--mint);font-size:11px}}
.side-bottom p{{margin-top:7px;color:#c5dfd5;font-size:11px;line-height:1.45}}
.page-main{{min-width:0}}
.topbar{{display:flex;align-items:center;justify-content:space-between;gap:20px;
  min-height:70px;padding:14px 28px;border-bottom:1px solid var(--line);
  background:rgba(255,255,255,.92)}}
.crumb{{color:var(--muted);font-size:13px}} .crumb strong{{color:var(--ink)}}
.scene-pill{{display:inline-flex;align-items:center;gap:7px;padding:7px 11px;
  border:1px solid #bce5cd;border-radius:999px;color:#167350;background:var(--mint-soft);
  font-size:12px;font-weight:700}}
.content{{max-width:1560px;margin:0 auto;padding:24px 28px 32px}}
.heading-row{{display:flex;align-items:flex-end;justify-content:space-between;
  gap:20px;margin-bottom:16px;flex-wrap:wrap}}
.eyebrow{{color:var(--green);font-size:11px;font-weight:800;letter-spacing:.1em;
  text-transform:uppercase}}
h1{{margin-top:4px;font-size:28px;letter-spacing:-.05em}}
.heading-row p{{max-width:560px;color:var(--muted);font-size:13.5px;line-height:1.45}}
.banner{{display:flex;gap:11px;margin-bottom:17px;padding:12px 14px;
  border-left:3px solid var(--amber);border-radius:8px;background:#fdf6e8;
  color:#6a5120;font-size:12.5px;line-height:1.5}}
.metric-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;
  margin-bottom:17px}}
.metric{{padding:15px;border:1px solid var(--line);border-radius:16px;
  background:var(--paper)}}
.metric-label{{color:var(--muted);font-size:12px}}
.metric strong{{display:block;margin-top:8px;font-size:26px;letter-spacing:-.05em}}
.metric-note{{margin-top:3px;color:var(--muted);font-size:11px}}
.dashboard-grid{{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:17px}}
.card{{border:1px solid var(--line);border-radius:18px;background:var(--paper);
  box-shadow:var(--shadow)}}
.map-card{{overflow:hidden}}
.map-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;
  padding:16px 18px;border-bottom:1px solid var(--line);flex-wrap:wrap}}
.map-top h2{{font-size:16px;letter-spacing:-.025em}}
.map-top p{{margin-top:4px;max-width:620px;color:var(--muted);font-size:12px;line-height:1.45}}
.filters{{display:flex;gap:6px;flex-wrap:wrap}}
.chip{{min-height:30px;padding:5px 9px;border:1px solid var(--line);border-radius:8px;
  color:var(--muted);background:#f4f7f5;font-size:11px;font-weight:700}}
.chip[aria-pressed=true]{{border-color:#93d2b1;color:var(--green);background:var(--mint-soft)}}
.map-wrap{{position:relative;background:#f7faf8}}
svg.map{{display:block;width:100%;height:auto}}
.parcel{{stroke:#fff;stroke-width:.35;cursor:pointer;transition:opacity .15s}}
.parcel:hover{{stroke:#0a3731;stroke-width:1.1}}
.parcel.sel{{stroke:#0a3731;stroke-width:1.6}}
.parcel.dim{{opacity:.12;pointer-events:none}}
.scalebar{{position:absolute;left:14px;bottom:12px;padding:6px 9px;border-radius:9px;
  background:rgba(255,255,255,.9);border:1px solid var(--line);font-size:10px;
  color:var(--muted);font-weight:700}}
.legend{{position:absolute;right:14px;bottom:12px;padding:9px 11px;border-radius:11px;
  background:rgba(255,255,255,.93);border:1px solid var(--line)}}
.legend b{{display:block;margin-bottom:6px;font-size:10px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted)}}
.ramp{{display:flex;height:11px;width:168px;border-radius:3px;overflow:hidden}}
.ramp i{{flex:1}}
.ramp-ends{{display:flex;justify-content:space-between;margin-top:4px;
  font-size:10px;color:var(--muted)}}
.map-foot{{display:flex;justify-content:space-between;gap:10px;padding:11px 17px;
  border-top:1px solid var(--line);color:var(--muted);font-size:11px;flex-wrap:wrap}}
.right-col{{display:grid;align-content:start;gap:17px}}
.pad{{padding:17px}}
.kicker{{color:var(--muted);font-size:11px;font-weight:750;letter-spacing:.08em;
  text-transform:uppercase}}
.title-row{{display:flex;align-items:flex-start;justify-content:space-between;
  gap:10px;margin-top:5px}}
.title-row h2{{font-size:17px}}
.badge{{padding:4px 8px;border-radius:999px;font-size:10.5px;font-weight:800;
  white-space:nowrap}}
.b-high{{color:#17704f;background:#e6f5ed}} .b-med{{color:#8a5b10;background:#fff3d8}}
.b-low{{color:#8a3a2c;background:#fde7e3}}
.rows{{display:grid;gap:1px;margin:14px 0;background:var(--line);
  border:1px solid var(--line);border-radius:11px;overflow:hidden}}
.row{{display:flex;justify-content:space-between;gap:10px;padding:8px 11px;
  background:#fff;font-size:12px}}
.row span{{color:var(--muted)}} .row b{{font-variant-numeric:tabular-nums}}
.note{{padding:11px;border-left:3px solid var(--blue);border-radius:6px;
  background:#eef6f9;color:#2c5767;font-size:11.5px;line-height:1.5}}
.prov{{display:grid;gap:1px;background:var(--line);border:1px solid var(--line);
  border-radius:11px;overflow:hidden}}
.prov div{{display:flex;justify-content:space-between;gap:10px;padding:7px 11px;
  background:#fff;font-size:11px}}
.prov span{{color:var(--muted)}}
.disclaim{{margin-top:13px;padding:11px;border-left:3px solid var(--coral);
  border-radius:6px;background:#fdeeeb;color:#7d3528;font-size:11.5px;line-height:1.5}}
@media(max-width:1180px){{.dashboard-grid{{grid-template-columns:1fr}}
  .metric-grid{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:900px){{.dashboard{{grid-template-columns:1fr}}
  .side-nav{{min-height:auto;padding:14px 16px}}
  .layer-list{{grid-template-columns:repeat(2,1fr);display:grid}}
  .side-bottom{{display:none}} .content{{padding:20px 14px}}}}
</style>
</head>
<body>
<div class="dashboard">
  <aside class="side-nav">
    <div class="brand-row">
      <div class="brand-mark">P</div>
      <div class="brand">Palm2Soil<small>Priority dashboard</small></div>
    </div>
    <div class="nav-label">Map layer</div>
    <div class="layer-list" id="layers"></div>
    <div class="side-bottom">
      <b>Proof of concept</b>
      <p>Real measurements from one Tanager-1 acquisition. Ranking is relative
      within this scene — no absolute salinity calibration exists yet.</p>
    </div>
  </aside>

  <main class="page-main">
    <header class="topbar">
      <div class="crumb">Pilot area / <strong>{SCENE['place']}</strong></div>
      <div class="scene-pill">{SCENE['date']} · {SCENE['cloud']} cloud · {DATASET_LABEL}</div>
    </header>

    <div class="content">
      <div class="heading-row">
        <div>
          <div class="eyebrow">Decision workspace</div>
          <h1>Remediation priority map</h1>
        </div>
        <p>{stats['total']} field parcels segmented from a single hyperspectral
        acquisition. {RANKING_DESCRIPTION} Select a parcel to see its
        measurements, data quality and provenance.</p>
      </div>

      {NO_ELIGIBLE_BANNER}
      <div class="banner"><b>Demonstration area.</b>&nbsp;No usable UAE hyperspectral
      scene exists in the open Tanager archive — we audited all 276 scenes across
      nine collections. This is the Konya–Çumra Plain, a documented
      groundwater-salinisation analogue with the same mechanism as Al Ain. The
      method transfers; local mineralogy will differ.</div>

      <section class="metric-grid">
        <article class="metric"><div class="metric-label">Parcels mapped</div>
          <strong>{stats['total']}</strong>
          <div class="metric-note">{stats['area']} ha · median {stats['median']} ha</div></article>
        <article class="metric"><div class="metric-label">Usable for ranking</div>
          <strong>{stats['usable']}</strong>
          <div class="metric-note">excludes low-confidence and merged parcels</div></article>
        <article class="metric"><div class="metric-label">High confidence</div>
          <strong>{stats['high']}</strong>
          <div class="metric-note">{stats['med']} medium · {stats['low']} low</div></article>
        <article class="metric"><div class="metric-label">Flagged for review</div>
          <strong>{stats['split']}</strong>
          <div class="metric-note">probably several fields merged</div></article>
      </section>

      <section class="dashboard-grid">
        <article class="card map-card">
          <div class="map-top">
            <div><h2 id="map-h">Review priority</h2><p id="map-p">…</p></div>
            <div class="filters">
              <button class="chip" id="f-high" aria-pressed="false" type="button">High confidence only</button>
              <button class="chip" id="f-split" aria-pressed="true" type="button">Hide merged parcels</button>
            </div>
          </div>
          <div class="map-wrap">
            <svg class="map" id="map" viewBox="0 0 1000 700" role="img"
                 aria-label="Choropleth of field parcels"></svg>
            <div class="scalebar" id="scalebar"></div>
            <div class="legend"><b id="leg-t">Scale</b>
              <div class="ramp" id="leg-ramp"></div>
              <div class="ramp-ends"><span id="leg-lo"></span><span id="leg-hi"></span></div>
            </div>
          </div>
          <div class="map-foot">
            <span><strong id="shown">—</strong> parcels shown</span>
            <span>Choropleth of measured parcels — not an interpolated surface</span>
          </div>
        </article>

        <div class="right-col">
          <aside class="card pad" aria-live="polite">
            <div class="kicker">Selected parcel</div>
            <div class="title-row"><h2 id="d-name">—</h2><span class="badge" id="d-conf">—</span></div>
            <div class="rows" id="d-rows"></div>
            <div class="note" id="d-note"></div>
          </aside>

          <aside class="card pad">
            <div class="kicker">Data provenance</div>
            <div class="prov" style="margin-top:10px">
              <div><span>Product</span><b>Tanager-1 ortho SR</b></div>
              <div><span>Scene</span><b>{SCENE['id']}</b></div>
              <div><span>Run id</span><b>{RUN_ID}</b></div>
              <div><span>Dataset</span><b>{DATA_NOTE}</b></div>
              <div><span>Acquired</span><b>{SCENE['date']}</b></div>
              <div><span>Sensor</span><b>426 bands, 376–2499 nm</b></div>
              <div><span>Grid</span><b>{SCENE['grid']}</b></div>
              <div><span>Processing</span><b>cloud, cirrus, no-data masked</b></div>
              <div><span>Licence</span><b>{SCENE['licence']}</b></div>
            </div>
            <div class="disclaim"><b>Not a salinity measurement.</b> Values are
            relative within this scene. Converting to dS/m requires paired ground
            measurements, which we do not have. Tiering is a designed mechanism,
            not a calibrated one.</div>
          </aside>
        </div>
      </section>
    </div>
  </main>
</div>

<script>
const F = {DATA};
const LAYERS = {LAYERS_JS};
const P20 = {json.dumps(stats['p20'])}, P40 = {json.dumps(stats['p40'])};
const RANKING_AVAILABLE = {json.dumps(not NO_ELIGIBLE)};
let layer = LAYERS[0], sel = null, onlyHigh = false, hideSplit = true;

const RAMPS = {{
  diverging: ['#b2182b','#ef8a62','#fddbc7','#f7f7f7','#d1e5f0','#67a9cf','#2166ac'],
  green:     ['#f7fcf5','#c7e9c0','#74c476','#31a354','#006d2c'],
  water:     ['#f6e8c3','#dfc27d','#c7eae5','#5ab4ac','#01665e'],
  soil:      ['#f6f4ef','#e6d8b5','#d8b365','#a6761d','#7a4f10']
}};

function lerp(a,b,t){{return a+(b-a)*t}}
function hex2rgb(h){{return [1,3,5].map(i=>parseInt(h.slice(i,i+2),16))}}
function ramp(name,t){{
  const c=RAMPS[name]; t=Math.max(0,Math.min(1,t));
  const x=t*(c.length-1), i=Math.min(Math.floor(x),c.length-2), f=x-i;
  const a=hex2rgb(c[i]), b=hex2rgb(c[i+1]);
  return `rgb(${{a.map((v,k)=>Math.round(lerp(v,b[k],f))).join(',')}})`;
}}

// geographic extent -> svg
const W=1000,H=700,PAD=16;
let lo=[1e9,1e9], hi=[-1e9,-1e9];
F.forEach(f=>f.r.forEach(r=>r.forEach(([x,y])=>{{
  lo[0]=Math.min(lo[0],x); lo[1]=Math.min(lo[1],y);
  hi[0]=Math.max(hi[0],x); hi[1]=Math.max(hi[1],y);
}})));
const latm=(lo[1]+hi[1])/2, kx=Math.cos(latm*Math.PI/180);
const gw=(hi[0]-lo[0])*kx, gh=hi[1]-lo[1];
const s=Math.min((W-2*PAD)/gw,(H-2*PAD)/gh);
const ox=(W-gw*s)/2, oy=(H-gh*s)/2;
const px=(x,y)=>[ox+(x-lo[0])*kx*s, oy+(hi[1]-y)*s];

function visible(p){{
  if(hideSplit && p.needs_split) return false;
  if(onlyHigh && p.confidence!=='high') return false;
  return true;
}}

function values(){{
  return F.filter(f=>visible(f.p)).map(f=>f.p[layer.key]).filter(v=>v!=null);
}}

function draw(){{
  if(!RANKING_AVAILABLE && layer.key==='stress_residual'){{
    document.getElementById('map').innerHTML =
      '<text x="500" y="340" text-anchor="middle" font-size="19" fill="#6c7d76">'
      + 'No parcels eligible for ranking</text>'
      + '<text x="500" y="368" text-anchor="middle" font-size="13" fill="#6c7d76">'
      + 'Choose another layer to inspect the available measurements</text>';
    document.getElementById('shown').textContent = '0';
    document.getElementById('leg-ramp').innerHTML = '';
    document.getElementById('leg-lo').textContent = '—';
    document.getElementById('leg-hi').textContent = '—';
    document.getElementById('leg-t').textContent = 'Ranking unavailable';
    return;
  }}
  const vals=values();
  if(!vals.length){{
    document.getElementById('map').innerHTML =
      `<text x="500" y="340" text-anchor="middle" font-size="19" fill="#6c7d76">`
      + `No parcels match the current filters</text>`
      + `<text x="500" y="368" text-anchor="middle" font-size="13" fill="#6c7d76">`
      + `Turn off a filter to see the rest</text>`;
    document.getElementById('shown').textContent = '0';
    document.getElementById('leg-ramp').innerHTML = '';
    document.getElementById('leg-lo').textContent = '—';
    document.getElementById('leg-hi').textContent = '—';
    document.getElementById('leg-t').textContent = layer.label;
    return;
  }}
  const mn=Math.min(...vals), mx=Math.max(...vals);
  const svg=document.getElementById('map');
  svg.innerHTML = F.map((f,i)=>{{
    const p=f.p, v=p[layer.key];
    const vis=visible(p);
    let t = (v==null||mx===mn) ? .5 : (v-mn)/(mx-mn);
    if(layer.rev) t = 1-t;
    const d = f.r.map(r=>'M'+r.map(([x,y])=>px(x,y).map(n=>n.toFixed(1)).join(',')).join('L')+'Z').join(' ');
    return `<path class="parcel${{vis?'':' dim'}}${{sel===i?' sel':''}}" d="${{d}}"
      fill="${{v==null?'#ccc':ramp(layer.ramp,t)}}" data-i="${{i}}">
      <title>Parcel ${{p.id}} — ${{layer.label}}: ${{v==null?'n/a':v.toFixed(4)}}</title></path>`;
  }}).join('');
  svg.querySelectorAll('.parcel').forEach(el=>
    el.addEventListener('click',()=>select(+el.dataset.i)));
  document.getElementById('shown').textContent = F.filter(f=>visible(f.p)).length;
  document.getElementById('leg-ramp').innerHTML =
    Array.from({{length:28}},(_,k)=>{{
      let t=k/27; if(layer.rev) t=1-t;
      return `<i style="background:${{ramp(layer.ramp,t)}}"></i>`;
    }}).join('');
  document.getElementById('leg-t').textContent = layer.label;
  document.getElementById('leg-lo').textContent = mn.toFixed(3);
  document.getElementById('leg-hi').textContent = mx.toFixed(3);
  document.getElementById('scalebar').textContent =
    `scene extent ${{(gw*111.32).toFixed(1)}} × ${{(gh*111.32).toFixed(1)}} km  ·  30 m pixels`;
}}

function band(r){{
  if(!RANKING_AVAILABLE) return ['Unavailable',''];
  if(r<=P20) return ['Priority 1','b-low'];
  if(r<=P40) return ['Priority 2','b-med'];
  return ['Priority 3','b-high'];
}}

function select(i){{
  sel=i; const p=F[i].p;
  // Excluded parcels must not be handed a priority band: they are not in the
  // ranking, so showing one invents a result.
  const eligible = !p.needs_split && p.confidence !== 'low';
  document.getElementById('d-name').textContent = 'Parcel '+p.id;
  const c=p.confidence;
  const badge=document.getElementById('d-conf');
  badge.textContent = c==='high'?'High confidence':c==='medium'?'Medium confidence':'Low confidence';
  badge.className = 'badge '+(c==='high'?'b-high':c==='medium'?'b-med':'b-low');
  const [pb] = band(p.stress_residual);
  const rows = [
    ...(RANKING_AVAILABLE ? [
      ['Inspection priority', eligible ? pb+' of 3' : 'Not ranked — excluded'],
      ['Rank', eligible ? p.stress_rank+' of {stats['total']}' : '—'],
    ] : []),
    ['Divergence score', p.stress_residual.toFixed(4)],
    ['Area', p.area_ha.toFixed(2)+' ha ('+p.n_pixels+' px)'],
    ['NDVI', p.NDVI_mean.toFixed(4)],
    ['NDRE', p.NDRE_mean.toFixed(4)],
    ['NDMI — canopy water', p.NDMI_mean.toFixed(4)],
    ['BSI — bare soil', p.BSI_mean.toFixed(4)],
    ['Salinity index', p.SI_Khan_mean.toFixed(4)],
    ['Clear imagery nearby', (p.local_valid_frac*100).toFixed(1)+'%'],
    ['Cloud nearby', (p.local_cloud_frac*100).toFixed(1)+'%'],
    ['Reflectance uncertainty', p.refl_uncertainty.toFixed(4)],
    ['Centre', p.lat.toFixed(4)+'°N, '+p.lon.toFixed(4)+'°E'],
  ];
  document.getElementById('d-rows').innerHTML =
    rows.map(([k,v])=>`<div class="row"><span>${{k}}</span><b>${{v}}</b></div>`).join('');
  let note = eligible
    ? 'Priority reflects how far this parcel\\'s chlorophyll signal sits below '
      + 'the level its own greenness predicts, on 8 June 2025. It indicates '
      + 'where to inspect, not what the soil contains.'
    : 'This parcel is excluded from the ranking, so it has no priority. Its '
      + 'measurements are shown for reference only.';
  if(!RANKING_AVAILABLE) note = 'No parcels in this dataset pass the quality '
    + 'filter. These measurements are shown for inspection only; no rank or '
    + 'inspection priority has been assigned.';
  if(p.needs_split) note += ' ⚠ This parcel is unusually large and probably '
    + 'contains several fields; treat its averages with caution.';
  if(p.confidence==='low') note += ' ⚠ Low confidence: few valid pixels.';
  document.getElementById('d-note').textContent = note;
  draw();
}}

document.getElementById('layers').innerHTML = LAYERS.map((l,i)=>
  `<button class="layer-btn" aria-pressed="${{i===0}}" data-i="${{i}}" type="button">${{l.label}}</button>`
).join('');
document.querySelectorAll('.layer-btn').forEach(b=>b.addEventListener('click',()=>{{
  layer = LAYERS[+b.dataset.i];
  document.querySelectorAll('.layer-btn').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));
  document.getElementById('map-h').textContent = layer.label;
  document.getElementById('map-p').textContent = layer.note;
  draw(); if(sel!=null) select(sel);
}}));
function reselectIfHidden(){{
  // A parcel filtered off the map must not keep its details on screen.
  if(sel!=null && visible(F[sel].p)) {{ draw(); return; }}
  const next = F.map((f,i)=>[i,f.p]).filter(([,p])=>visible(p))
                .sort((a,b)=>(a[1].stress_rank||1e9)-(b[1].stress_rank||1e9))[0];
  if(next) select(next[0]);
  else {{
    sel=null;
    document.getElementById('d-name').textContent='No parcel selected';
    document.getElementById('d-conf').textContent='—';
    document.getElementById('d-conf').className='badge';
    document.getElementById('d-rows').innerHTML='';
    document.getElementById('d-note').textContent=
      'No parcel matches the current filters. Turn one off to select a parcel.';
    draw();
  }}
}}
document.getElementById('f-high').addEventListener('click',e=>{{
  onlyHigh=!onlyHigh; e.currentTarget.setAttribute('aria-pressed',String(onlyHigh));
  reselectIfHidden();
}});
document.getElementById('f-split').addEventListener('click',e=>{{
  hideSplit=!hideSplit; e.currentTarget.setAttribute('aria-pressed',String(hideSplit));
  reselectIfHidden();
}});

document.getElementById('map-p').textContent = layer.note;
draw();
// open on the highest-priority parcel that is actually usable
const first = F.map((f,i)=>[i,f.p]).filter(([,p])=>visible(p) && p.confidence!=='low')
               .sort((a,b)=>a[1].stress_rank-b[1].stress_rank)[0];
select(first ? first[0] : 0);
</script>
</body>
</html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"WROTE {OUT}  {os.path.getsize(OUT)//1024} KB  ({len(feats)} parcels embedded)")
