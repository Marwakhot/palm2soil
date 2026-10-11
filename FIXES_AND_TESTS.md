# Palm2Soil submission — corrected

Builds on the 11 October corrected package; eight further issues fixed and
verified on 11 October 2026.

## Issues fixed in this round

| # | Issue | Fix |
|---|---|---|
| 1 | **Disconnected patches shared one parcel id.** One "parcel" was 395 fragments spread over 29.2 km and labelled high confidence; two more (107 and 112 pieces) were featured in the field app unflagged. Their mean measurements described nothing real. | Disconnected regions are split before anything is measured, using 4-connectivity so one parcel is exactly one polygon. 375 → 409 parcels; max pieces per parcel 395 → **1**, max span 29.2 km → **3.03 km**. |
| 2 | Parcels excluded from the ranking were still shown a priority band and a rank. | Eligibility now drives the detail panel: excluded parcels read "Not ranked — excluded", rank "—", with an explanatory note. |
| 3 | The map dropped 16 interior holes across 11 parcels, filling areas the segmentation had excluded. | Every ring is carried, including interior rings of MultiPolygons. All 426 rings now reach the page. |
| 4 | Filtering a selected parcel off the map left its details on screen. | Selection moves to the highest-priority visible parcel, or clears with an explanatory message. |
| 5 | Sentinel-2 sampling was offset half a pixel (10 m at 20 m). `hasattr(stack,"rio")` is always true once rioxarray is imported, so the correct fallback never ran. | The transform is built from the cube's own coordinates, which are pixel centres. |
| 6 | Joined observations lost every quality column: the importer still listed `valid_frac`/`cloud_frac`, renamed `local_*` two rounds earlier. | Column list updated; quality and uncertainty now survive the join. |
| 7 | One badly encoded JSON aborted a whole import batch. | `UnicodeDecodeError` is caught per file, reported, and the remaining files still import. |
| 8 | A failed storage **read** showed an empty checklist next to "Saved on this device". | Read failures are reported explicitly and distinguished from "nothing recorded yet". |

Run IDs changed again because parcel geometry changed. The apps, example
records, full-scene dataset and executed sample notebook were all regenerated
together. Older exports are rejected rather than silently reassigned.

## Verification

- **31/31 regression tests passed** (`python -m unittest discover -s tests`).
- **33/33 end-to-end smoke checks passed** (`python smoke_test.py`), which
  executes the bundled notebook on the sample and rebuilds both apps.
- Both apps were driven in a real browser: no console errors, all six map
  layers, both filters, excluded-parcel handling, record capture, and the
  storage-failure path were exercised directly.
- Fragmentation, hole preservation, importer columns and encoding handling
  were each verified against the shipped artefacts rather than the source.

Not rerun: the 860 MB full-scene download and the Sentinel-2 seasonal network
workflow. Scientific calibration remains outside this proof of concept.
