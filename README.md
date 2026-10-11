# Palm2Soil — Proof of Concept

**Satellite-guided salinity remediation using date palm waste**
Arab Youth Space Hackathon 2026 · The 813 Challenge · Precision Agriculture & Crop Intelligence

---

## What this is

A pipeline that turns one hyperspectral acquisition into a **ranked inspection
priority list** for farm blocks, so an authority with a fixed remediation budget
has an evidence base for deciding where to look first.

Input is open satellite data. Output is one row per field parcel, ranked, with
data-quality figures attached to every row.

---

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook 07_palm2soil_salinity_prioritisation.ipynb
```

Then **Run All**. It executes end to end in about two minutes on the bundled
sample, with **no network access and no downloads**.

Tested on Python 3.12.

---

## What you get

Running the notebook writes `parcel_table.csv` and `parcels.geojson` to the
working directory, and draws the parcel maps inline.

To see the results without running anything, open `examples/output/`.

---

## Contents

```
README.md                                 this file
requirements.txt                          pinned dependencies
smoke_test.py                             end-to-end regression check
dataset.py                                dataset selection and id validation
run_identity.py                           shared geometry-based run fingerprint
import_observations.py                    joins field records back to measurements
tests/                                    dataset, importer, app and pipeline regressions
07_palm2soil_salinity_prioritisation.ipynb    the pipeline, end to end
s2_srf.py                                 ESA Sentinel-2 response loader
s2_srf_esa_v5.xlsx                        ESA S2-SRF v5 (COPE-GSEG-EOPG-TN-15-0007)

examples/input/
  sample_cumra_128px_ortho_sr.h5          6.8 MB — 128×128 px crop, runs offline
  stac_item_20250608_091605_90_4001.json  scene metadata and band wavelengths
  sample_field_records.json               8 ground observations from the field app
  sample_field_records_with_errors.json   deliberately broken records, for the importer

examples/output/                          produced from the FULL scene
  parcel_table.csv                        409 parcels × 30 columns
  parcels.geojson                         same, with polygons (WGS84)
  parcel_summary.txt                      text digest
  parcel_check.png                        parcel maps
  swir_absorption_argument.png            why hyperspectral is necessary
  swir_stats.txt                          the numbers behind it
  rededge_sentinel2_vs_tanager.png        a test that went against us
  rededge_stats.txt
  temporal_trajectories.png               25 Sentinel-2 dates, same parcels
  temporal_stats.txt
  sample_run_executed.ipynb               the notebook run on the bundled sample

Palm2Soil_Dashboard.html                  authority dashboard (open in a browser)
Palm2Soil_FieldApp.html                   field capture app (mobile view)
```

Both HTML files are self-contained — no server, no internet.

---

## Example input

Two files in `examples/input/`.

**`sample_cumra_128px_ortho_sr.h5`** — a 128 × 128 pixel crop (3.8 × 3.8 km) of
the Tanager-1 acquisition, keeping the original HDF5 layout exactly. The
426-band axis is preserved so band indices still match the STAC wavelength
list, but bands the pipeline never reads are stored as zeros to keep the file
small. The crop was
chosen to contain both actively cropped parcels and exposed bare soil, because
the pipeline needs both.

**`stac_item_…json`** — the SpatioTemporal Asset Catalog record. This is how the
pipeline discovers the scene and reads the wavelength of each of the 426 bands.

> **Sample numbers differ from the full-scene numbers.** The sample is stored at
> reduced precision and covers well under 1% of the scene area, so it yields
> far fewer parcels than 409 and the statistics differ. This is expected. To reproduce the
> figures in `examples/output/`, set `DOWNLOAD_FULL = True` in Step 3 of the
> notebook; it will fetch the full 860 MB acquisition from Planet's open archive.

---

## Example output

`parcel_table.csv` — one row per parcel:

| Column | Meaning |
|---|---|
| `parcel_id` | identifier, matches `parcels.geojson` |
| `area_ha`, `n_pixels` | size |
| `lon`, `lat` | centroid, WGS84 |
| `NDVI_mean` / `_std` | greenness |
| `NDRE_mean` / `_std` | chlorophyll, red edge |
| `NDMI_mean` / `_std` | canopy water |
| `BSI_mean` / `_std` | bare soil |
| `SI_Khan`, `SI_Douaoui`, `BI`, `CRSI` | published salinity indices |
| `stress_residual` | **the ranking quantity** — see below |
| `stress_rank` | 1 = most divergent |
| `run_id` | identifies which pipeline run produced these ids |
| `local_valid_frac`, `local_cloud_frac` | clear / cloudy share of the area around the parcel |
| `refl_uncertainty` | provider's own reflectance uncertainty |
| `confidence` | high / medium / low — from parcel size, surrounding image quality and reflectance uncertainty |
| `needs_split` | true if probably several merged fields |

For analysis, filter to `confidence != 'low'` and `needs_split == False`
— 345 of 409 parcels.

---

## How the ranking works

NDVI approximates what the eye sees. NDRE reflects chlorophyll condition inside
the leaf. Where the two diverge, canopy cover and chlorophyll are telling
different stories.

They sit on different scales, so a raw difference is meaningless. Instead the
scene-wide relationship is fitted —

```
NDRE = 0.902 × NDVI − 0.119
```

— and each parcel is scored by how far its NDRE falls **below the level its own
greenness predicts**. More negative means more divergence.

**Two checks that it is measuring something real**, both computed in the notebook:

1. **Spatially coherent.** Neighbouring parcels have correlated scores
   (r = +0.23). Noise would be spatially random.
2. **Physically consistent.** The most-divergent decile is drier (NDMI −0.045),
   barer (BSI +0.036) and more saline-looking (+0.016) than the least — while
   being only marginally less green (NDVI −0.018). **Standard greenness
   separates the two groups least of all.**

---

## Why hyperspectral

Soil salinity is not directly visible to any optical sensor. **Pure halite has
no diagnostic absorption in the VNIR–SWIR.** It is inferred from hydration
features and associated evaporite minerals, whose absorptions are tens of
nanometres wide and sit near 1450, 1850, 1950, 2200 and 2340 nm.

Sentinel-2 was simulated by integrating the Tanager spectrum through **ESA's
measured spectral response functions** — same ground, same instant, same
atmosphere, so sampling is the only variable.

| | Tanager-1 | Sentinel-2 |
|---|---|---|
| Bands, 1500–2500 nm | **201** | 2 |
| 1931–2123 nm salt window | 39 bands | 6.2% of its SWIR response |
| 1667–2092 nm | 86 bands | **zero — 425 nm blind gap** |
| 2316–2450 nm | 27 bands | **zero — response has ended** |

Measured on bare soil, Sentinel-2's two SWIR bands reconstruct only **51%** of
the variance in the 2216 nm feature depth and **4%** of the 2340 nm feature.

---

## Closing the loop

The field app exports its records as JSON; `import_observations.py` reads them
back, validates each one against the parcel table, and writes
`observations_joined.csv` — one row per observation with that parcel's
satellite measurements attached. Try it:

```bash
python import_observations.py examples/input/sample_field_records.json \
       --data-dir examples/output
```

Records that cannot be matched are **never discarded quietly**. An unknown
parcel id, a record from a different scene, a malformed timestamp or a missing
field are each reported individually and written to
`observations_unmatched.csv`, and the command exits non-zero. An unmatched
record almost always means a real mistake — a stale export, or a table
regenerated since the observation was made — and silence would hide it. The
second example file exercises all four cases:

```bash
python import_observations.py examples/input/sample_field_records_with_errors.json \
       --data-dir examples/output
```

**This joins; it does not calibrate.** Producing a dS/m model needs laboratory
soil values and far more records than a prototype collects. The joined table is
the input that work would start from — which is precisely the gap named under
Limitations below.

---

## A note on data quality columns

Parcels are **built from valid pixels**, so validity measured *inside* a parcel
is 1.0 for every parcel by construction. An earlier version reported exactly
that and it was meaningless. The columns now measure the parcel's
**surrounding area** instead — `local_valid_frac` and `local_cloud_frac` cover
its bounding box, which includes the masked pixels around it and therefore
actually varies (0.71–1.00 on the full scene).

`confidence` combines the three things that genuinely differ between parcels:
how many pixels it averages over, how clean the imagery around it is, and the
provider's own reflectance uncertainty. It is **not** a claim about in-parcel
validity.

Every output also carries a `run_id`. Parcel ids are only unique within one
run — a sample run numbers 1..20 and so does the start of a full-scene run —
so anything referring to a parcel must also say which run it meant. The
importer rejects records whose run id is absent or does not match.

Run fingerprints use the parcel geometries, coordinate reference systems and
segmentation settings through one shared function. The notebook and standalone
pipeline use the same rule, independent of table row order or rounded display
measurements. The importer checks scene identity from the run identifier even
when no `parcel_summary.txt` is present.

If no parcels pass the quality filter, both apps show that ranking is
unavailable. They do not assign priorities to excluded parcels.

---

## Limitations

Stated plainly, because a proof of concept that overstates its reach is worth
less than one that knows its boundaries.

**No absolute salinity.** All ranking is **relative within a scene**. Converting
to dS/m needs paired ground measurements at known locations; the 2009 EAD
assessment is available to us only as regional percentages, and a per-pixel map
cannot be validated against a summary statistic. Obtaining georeferenced ground
truth is the first objective of the incubation phase. Until then, intervention
tiering is a designed mechanism, not a calibrated one.

**No seasonal prediction.** We tested whether the single-date score predicts
seasonal performance across 25 Sentinel-2 dates. **It does not** (|r| ≤ 0.25,
signs opposite to a stress interpretation). On one acquisition,
canopy–chlorophyll divergence cannot be cleanly separated from crop type and
growth stage. The score indicates **where to inspect**, not what will happen.

**Parcels are not cadastral.** Spatially disconnected regions are split, so one parcel is exactly one contiguous polygon. They are segmented from the imagery by watershed
on the NDVI gradient, approximating management units. 5 of 409 are flagged
`needs_split`.

**Single acquisition.** Tanager imaged this footprint **once** in the entire open
archive — a real operational constraint of the sensor. Seasonal tracking is
demonstrated with Sentinel-2 instead.

**Salinity indices are not calibrated.** The four published indices are verified
as formulas and computed on broadband-equivalent reflectance, but none is tied
to a measured dS/m value. They are descriptive context; the ranking does not
depend on them.

---

## Why the demonstration is in Türkiye, not the UAE

**There is no usable UAE hyperspectral scene in the open Tanager archive.** We
established that by auditing all 276 scenes across all nine collections rather
than assuming it. Three fall in MENA, all above 50% cloud; the agriculture
collection contains none on the Arabian Peninsula; the single UAE scene is in an
unrelated collection at 51% cloud and 25.8° off-nadir, over Al Dhafrah rather
than Al Ain.

We therefore demonstrate on the **Konya–Çumra Plain**, a documented
secondary-salinisation case and a close structural analogue to Al Ain: a
semi-arid endorheic basin where roughly 90% of irrigation water is drawn from
groundwater, with about 1 m per year of water-table decline and published
water–salt balance studies.

Reported root-zone salinity there is milder than the UAE bands, so this is
framed as detecting the condition *before* it reaches UAE severity. The sampling
argument transfers unchanged because it concerns instrument design; the specific
mineral features present will differ with local soil chemistry.

---

## Two results that went against us

Both are kept in the submission, and both are reproducible from the notebook.

**The red-edge justification did not hold.** Our application argued hyperspectral
on red-edge position shifts. Sentinel-2 reconstructs red-edge position at
R² 0.68, and its NDRE separated an independent grouping better than either
red-edge metric. The hyperspectral case was rebuilt on SWIR absorption, where
the evidence is far stronger.

**The score does not predict the season.** As above.

Each led to a better-specified design — the complementary-sensor architecture
and the SWIR argument respectively. A justification that survives its own test
is worth more than one never tested.

---

## Platform note

If `pip install` tries to **build rasterio or pyproj from source** and fails
asking for GDAL or PROJ:

You are on Intel macOS 14. Recent releases ship x86_64 wheels tagged
`macosx_15_0`, which macOS 14 cannot install. The pins in `requirements.txt`
(`rasterio==1.4.3`, `pyproj==3.7.2`) are the newest versions with a usable
wheel. Apple Silicon and Linux are unaffected.

---

## Data and attribution

| | |
|---|---|
| Imagery | Planet Tanager-1, orthorectified surface reflectance |
| Scene | `20250608_091605_90_4001` — Çumra, Konya, Türkiye, 8 June 2025 |
| Sensor | 426 bands, 376–2499 nm, 30 m |
| Access | Planet open STAC catalogue |
| Licence | **CC-BY-4.0** |
| Seasonal comparison | Sentinel-2 L2A via Microsoft Planetary Computer, tile 36SVG |
| Sentinel-2 responses | ESA/Copernicus, COPE-GSEG-EOPG-TN-15-0007 S2-SRF v5 |

All 2009 soil salinity figures are from Abdelfattah et al., *Soil Survey
Horizons*, Soil Science Society of America, 2009, commissioned by the
Environment Agency – Abu Dhabi.

---

## Reproducing everything

**The notebook is the thing to run** — it works out of the box on the bundled
sample. Everything in `src/` is included so every published number is
traceable, but the scripts have different requirements.

**Run in this package as shipped:**

```bash
python smoke_test.py              # runs the notebook, then checks everything
python -m unittest discover -s tests -v  # edge-case regression tests
python src/make_dashboard.py      # rebuilds the authority dashboard
python src/make_farmer_app.py     # rebuilds the field app
python import_observations.py examples/input/sample_field_records.json \
       --data-dir examples/output        # joins field records to measurements
```

Both app builders take `--data-dir` to say which dataset to use, and `-o` for
the output path. With neither, they look in the package root first, then
`examples/output/`, and print which one they chose. The dashboard reads its
table and GeoJSON from the **same** directory and requires matching parcel
and run identities. Both builders reject duplicate parcel ids and missing or
inconsistent run metadata. This prevents measurements from being attached to
shapes from a different run.

```bash
python src/make_dashboard.py --data-dir examples/output
```

`smoke_test.py` is the regression guard: it executes the notebook on the
bundled sample, checks the ids match between table and GeoJSON, verifies the
geometries are valid and both centroids and polygons fall inside the footprint
read from the input HDF5 metadata,
rebuilds both apps from those outputs, and confirms every field they feature
actually exists in the dataset. It exits non-zero on any failure.

The tests also cover empty eligible subsets, mismatched runs, scene checks
without a summary file, malformed field ids, stable run fingerprints and
geographic displacement. Run both commands before repackaging the submission.
Node.js is optional: when available on `PATH`, the regression suite also
executes isolated checks of the generated JavaScript. No browser or network
connection is needed for those checks.

**Need the full 860 MB scene** (set `DOWNLOAD_FULL = True` in the notebook, or
fetch `ortho_sr_hdf5` from the STAC item in `examples/input/`, and place it at
the package root):

```bash
python src/make_parcel_table.py   # parcel table + geojson
python src/make_swir_figure.py    # the hyperspectral argument
python src/make_rededge_figure.py # the test that went against us
python src/make_example_input.py  # regenerates the bundled sample
```

**Needs the full scene and network access:**

```bash
python src/make_temporal_figure.py  # 25 Sentinel-2 dates via Planetary Computer
```

The notebook itself is generated by `src/build_palm2soil_notebook.py`, so its
content stays reviewable as source rather than hand-edited JSON.
