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
Build examples/input/ -- a small spatial subset of the Tanager scene.

The full acquisition is 860 MB, which is a lot to ask of a reviewer who just
wants to confirm the notebook runs. This writes a 192 x 192 pixel crop
(5.8 x 5.8 km) that keeps the file layout of the original exactly, so the
notebook reads it with no code changes.

The crop was chosen to contain both actively cropped parcels and exposed bare
soil, because the pipeline needs both: canopy indices come from the cropped
parcels, soil absorption features from the bare ground.

Reflectance is stored float16 to keep the file small. That is ~3 significant
figures, which is below the noise floor of the measurement but does shift
index values in the 4th decimal place -- so numbers from the sample will not
match the full-scene outputs exactly. That is expected and documented.
"""
import shutil

import h5py
import numpy as np

SRC = os.path.join(REPO, "20250608_091605_90_4001_ortho_sr_hdf5.h5")
OUTDIR = os.path.join(REPO, "examples", "input")
OUT = os.path.join(OUTDIR, "sample_cumra_128px_ortho_sr.h5")
ROOT = "HDFEOS/GRIDS/HYP/Data Fields"

# 128 px (3.8 km) centred in the window originally chosen for its mix of
# cropped and bare ground. Smaller than ideal, but the archive has to be small
# enough to actually transfer and upload.
SZ = 128
_Y0, _X0, _SRC_SZ = 191, 453, 192
_OFF = (_SRC_SZ - SZ) // 2
Y0, X0 = _Y0 + _OFF, _X0 + _OFF
UL_X, UL_Y, RES = 450750.0, 4172100.0, 30.0

os.makedirs(OUTDIR, exist_ok=True)
nx, ny = UL_X + X0 * RES, UL_Y - Y0 * RES
lx, ly = nx + SZ * RES, ny - SZ * RES

struct = f"""GROUP=SwathStructure
END_GROUP=SwathStructure
GROUP=GridStructure
\tGROUP=GRID_1
\t\tGridName="HYP"
\t\tBand=426
\t\tXDim={SZ}
\t\tYDim={SZ}
\t\tUpperLeftPointMtrs=({nx:.2f},{ny:.2f})
\t\tLowerRightMtrs=({lx:.2f},{ly:.2f})
\t\tProjection=HE5_GCTP_UTM
\t\tZoneCode=36
\t\tSphereCode=12
\t\tPixelRegistration=HE5_HDFE_CORNER
\t\tGridOrigin=HE5_HDFE_GD_UL
\tEND_GROUP=GRID_1
END_GROUP=GridStructure
END
"""

print(f"cropping y{Y0}:{Y0+SZ} x{X0}:{X0+SZ} from {os.path.basename(SRC)}")
with h5py.File(SRC, "r") as src, h5py.File(OUT, "w") as dst:
    g = dst.create_group(ROOT)
    for name in ("beta_cloud_mask", "nodata_pixels", "beta_cirrus_mask"):
        a = src[f"{ROOT}/{name}"][Y0:Y0 + SZ, X0:X0 + SZ]
        g.create_dataset(name, data=a, compression="gzip", compression_opts=6)
        print(f"  {name:32} {a.shape} {a.dtype}")
    # The 426-band axis is preserved so band indices still line up with the
    # STAC wavelength list, but bands the pipeline never reads are stored as
    # zeros -- they compress to almost nothing and save a third of the file.
    # Use the bundled STAC copy when present; only reach for the network if
    # it is genuinely missing.
    import json as _json
    _stac = os.path.join(OUTDIR, "stac_item_20250608_091605_90_4001.json")
    if os.path.exists(_stac):
        with open(_stac, encoding="utf-8") as _sf:
            _item = _json.load(_sf)
    else:
        import requests as _rq
        _item = _rq.get("https://www.planet.com/data/stac/tanager-core-imagery/"
                        "agriculture/20250608_091605_90_4001/"
                        "20250608_091605_90_4001.json", timeout=120).json()
    _wl = np.array([b["eo:center_wavelength"] for b in
                    _item["assets"]["ortho_sr_hdf5"]["bands"]
                    if "eo:center_wavelength" in b]) * 1000.0
    USE = ((_wl >= 430) & (_wl <= 920)) | ((_wl >= 1450) & (_wl <= 2450))
    for _t in (490, 570, 665, 705, 860, 1650, 2200):
        USE[int(np.argmin(np.abs(_wl - _t)))] = True

    d = src[f"{ROOT}/surface_reflectance"]
    out = g.create_dataset("surface_reflectance", shape=(d.shape[0], SZ, SZ),
                           dtype="float16", compression="gzip",
                           compression_opts=9, chunks=(16, SZ, SZ))
    for i in range(0, d.shape[0], 16):
        blk = d[i:i + 16, Y0:Y0 + SZ, X0:X0 + SZ].astype("float16")
        blk[~USE[i:i + 16]] = 0.0          # bands the pipeline never reads
        out[i:i + 16] = blk
    print(f"  surface_reflectance              {out.shape} float16 "
          f"({int(USE.sum())} bands carried, {int((~USE).sum())} zeroed)")

    # The pipeline only ever averages provider uncertainty over the seven bands
    # it reads, so the sample carries that 2-D mean rather than the full
    # 426-band cube, which would double the file for no analytical gain.
    USED = [23, 39, 58, 66, 97, 254, 365]      # blue..swir2, see notebook Step 5
    u = src[f"{ROOT}/surface_reflectance_uncertainty"]
    um = np.nanmean(np.stack([u[i, Y0:Y0 + SZ, X0:X0 + SZ].astype("float32")
                              for i in USED]), axis=0)
    g.create_dataset("surface_reflectance_uncertainty_mean", data=um.astype("float32"),
                     compression="gzip", compression_opts=9)
    print(f"  uncertainty_mean (2-D)           {um.shape} float32")
    hyp = dst[ "HDFEOS/GRIDS/HYP" ]
    for k, v in src["HDFEOS/GRIDS/HYP"].attrs.items():
        hyp.attrs[k] = v
    info = dst.create_group("HDFEOS INFORMATION")
    info.create_dataset("StructMetadata.0", data=np.bytes_(struct))
    dst.attrs["note"] = np.bytes_(
        "192x192 px subset of Tanager-1 scene 20250608_091605_90_4001 "
        "(Cumra, Konya, Turkiye, 2025-06-08). Reflectance stored float16 to "
        "reduce size; index values differ from the full scene in the 4th "
        "decimal. Bands outside 430-920 nm and 1450-2450 nm are stored as "
        "zeros because the pipeline never reads them; the 426-band axis is "
        "kept so indices still match the STAC wavelength list. Per-band "
        "uncertainty is reduced to a 2-D mean over the seven bands read. "
        "Source data CC-BY-4.0, Planet Labs.")

mb = os.path.getsize(OUT) / 1e6
print(f"\nwrote {OUT}  {mb:.1f} MB")
print(f"  extent {SZ*RES/1000:.1f} x {SZ*RES/1000:.1f} km  ·  UL ({nx:.0f}, {ny:.0f}) UTM 36N")

# the STAC item is the other half of "example input": it is how the pipeline
# discovers the scene and reads band wavelengths.
import json
import requests
url = ("https://www.planet.com/data/stac/tanager-core-imagery/agriculture/"
       "20250608_091605_90_4001/20250608_091605_90_4001.json")
sj = os.path.join(OUTDIR, "stac_item_20250608_091605_90_4001.json")
if os.path.exists(sj):
    print(f"kept  {sj}  (already bundled)")
else:
    item = requests.get(url, timeout=120).json()
    with open(sj, "w", encoding="utf-8") as f:
        json.dump(item, f, indent=1)
print(f"wrote {sj}  {os.path.getsize(sj)/1e3:.0f} KB")
