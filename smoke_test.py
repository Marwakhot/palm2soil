# -*- coding: utf-8 -*-
"""
End-to-end smoke test for the Palm2Soil submission package.

Catches the regressions that actually happened during development:
  * a hardcoded geotransform putting sample parcels 14.75 km from the truth
  * hardcoded parcel ids crashing the app builders on a different dataset
  * the table and the geojson drifting out of step
  * rebuild commands that do not run from the package layout

    python smoke_test.py                 # run the notebook, then check
    python smoke_test.py --no-notebook   # check existing outputs only

Exit status is 0 only if every check passes.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(REPO) == "src":
    REPO = os.path.dirname(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

NB = "07_palm2soil_salinity_prioritisation.ipynb"

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    return ok


def src(script):
    """Scripts live at the repo root, or under src/ in the package."""
    for p in (os.path.join(REPO, script), os.path.join(REPO, "src", script)):
        if os.path.exists(p):
            return p
    return None


def input_footprint(h5_path):
    """Return the input's own projected CRS and metre bounds from HDF5."""
    import re
    import h5py
    from pyproj import CRS
    with h5py.File(h5_path, "r") as handle:
        metadata = handle["HDFEOS INFORMATION/StructMetadata.0"][()].decode("utf-8")
        epsg = int(handle["HDFEOS/GRIDS/HYP"].attrs["epsg_code"])
    number = r"([-+\d.eE]+)"
    corners = []
    for name in ("UpperLeftPointMtrs", "LowerRightMtrs"):
        match = re.search(name + r"=\(\s*" + number + r"\s*,\s*" + number + r"\s*\)", metadata)
        if not match:
            raise ValueError(f"Missing {name} in {h5_path}")
        corners.append(tuple(map(float, match.groups())))
    crs = CRS.from_epsg(epsg)
    if not crs.is_projected or any(axis.unit_conversion_factor != 1 for axis in crs.axis_info):
        raise ValueError("Footprint validation requires a projected CRS in metres")
    (x1, y1), (x2, y2) = corners
    return crs, (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def footprint_violations(table, geojson, h5_path, tolerance_m=0.05):
    """Check both table centroids and all polygon bounds in the input CRS.

    Five centimetres allows numerical projection noise, not displaced pixels.
    """
    import math
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform
    crs, (left, bottom, right, top) = input_footprint(h5_path)
    project = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform

    def inside(bounds):
        return (all(math.isfinite(v) for v in bounds)
                and left - tolerance_m <= bounds[0]
                and bottom - tolerance_m <= bounds[1]
                and bounds[2] <= right + tolerance_m
                and bounds[3] <= top + tolerance_m)

    bad_centroids, bad_polygons = [], []
    for row in table.itertuples():
        x, y = project(row.lon, row.lat)
        if not inside((x, y, x, y)):
            bad_centroids.append(int(row.parcel_id))
    for feature in geojson["features"]:
        try:
            geometry = transform(project, shape(feature["geometry"]))
            valid = not geometry.is_empty and inside(geometry.bounds)
        except Exception:
            valid = False
        if not valid:
            bad_polygons.append(int(feature["properties"]["parcel_id"]))
    return bad_centroids, bad_polygons


def input_path(directory):
    """Follow the notebook's full-scene-first input selection."""
    import glob
    candidates = glob.glob(os.path.join(directory, "examples", "input", "stac_item_*.json"))
    if len(candidates) == 1:
        with open(candidates[0], encoding="utf-8") as handle:
            item = json.load(handle)
        key = "ortho_sr_hdf5" if "ortho_sr_hdf5" in item["assets"] else "basic_sr_hdf5"
        full_path = os.path.join(directory, os.path.basename(item["assets"][key]["href"]))
        if os.path.exists(full_path):
            return full_path
    return os.path.join(directory, "examples", "input", "sample_cumra_128px_ortho_sr.h5")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-notebook", action="store_true",
                    help="skip executing the notebook; check existing outputs")
    a = ap.parse_args()
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8")

    work = tempfile.mkdtemp(prefix="palm2soil_smoke_")
    print(f"scratch: {work}\n")

    # ---- 1. run the notebook on the bundled sample ----------------------
    if a.no_notebook:
        print("1. notebook execution SKIPPED")
        out_dir = REPO
    else:
        print("1. running the notebook on the bundled sample")
        nb_path = os.path.join(REPO, NB)
        if not check("notebook present", os.path.exists(nb_path), nb_path):
            return finish()
        # run in a copy so the repo's own outputs are not overwritten
        for item in ("examples", "s2_srf.py", "s2_srf_esa_v5.xlsx", "run_identity.py", NB):
            s = os.path.join(REPO, item)
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(work, item))
            elif os.path.exists(s):
                shutil.copy2(s, work)
        r = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--output", os.path.join(work, "run.ipynb"), NB],
            cwd=work, capture_output=True, text=True, encoding="utf-8",
            env=child_env, timeout=3600)
        if not check("notebook ran without error", r.returncode == 0,
                     (r.stderr or "").strip().splitlines()[-1][:90] if r.returncode else ""):
            return finish()
        out_dir = work

    csv_p = os.path.join(out_dir, "parcel_table.csv")
    gj_p = os.path.join(out_dir, "parcels.geojson")

    # ---- 2. outputs: ids, geometry, coordinates -------------------------
    print("\n2. checking outputs")
    import pandas as pd
    if not check("parcel_table.csv written", os.path.exists(csv_p)):
        return finish()
    if not check("parcels.geojson written", os.path.exists(gj_p)):
        return finish()

    df = pd.read_csv(csv_p)
    gj = json.load(open(gj_p, encoding="utf-8"))
    ids_csv = set(df.parcel_id.astype(int))
    ids_gj = [int(f["properties"]["parcel_id"]) for f in gj["features"]]

    check("table is not empty", len(df) > 0, f"{len(df)} parcels")
    check("ids match between table and geojson", ids_csv == set(ids_gj))
    check("no duplicate ids in geojson", len(ids_gj) == len(set(ids_gj)))
    check("no duplicate ids in the table",
          len(df) == df.parcel_id.nunique(),
          f"{len(df) - df.parcel_id.nunique()} duplicate(s)")
    check("run_id present", "run_id" in df.columns and df.run_id.nunique() == 1,
          str(df.run_id.iloc[0]) if "run_id" in df.columns else "missing")
    # quality columns must actually vary -- constant 1.0 / 0.0 was the bug
    for col in ("local_valid_frac", "local_cloud_frac"):
        if col in df.columns:
            check(f"{col} is informative", df[col].nunique() > 1,
                  f"{df[col].nunique()} distinct values")

    # Real topology check, not just "is it shaped like a polygon"
    try:
        from shapely.geometry import shape
        bad_geom = []
        for f in gj["features"]:
            try:
                g = shape(f["geometry"])
                if g.is_empty or not g.is_valid or g.area <= 0:
                    bad_geom.append(f["properties"]["parcel_id"])
            except Exception:
                bad_geom.append(f["properties"]["parcel_id"])
        check("all geometries topologically valid", not bad_geom,
              f"{len(bad_geom)} invalid" if bad_geom else "shapely is_valid")
    except ImportError:
        bad_geom = [f["properties"]["parcel_id"] for f in gj["features"]
                    if f["geometry"]["type"] not in ("Polygon", "MultiPolygon")
                    or not f["geometry"]["coordinates"]]
        check("all geometries well-formed (shapely unavailable)",
              not bad_geom, f"{len(bad_geom)} bad")

    # Validate against metadata in the input itself, in projected metres.
    bad_centroids, bad_polygons = footprint_violations(df, gj, input_path(out_dir))
    check("centroids inside the actual input footprint", not bad_centroids,
          f"{len(bad_centroids)} outside")
    check("polygons inside the actual input footprint", not bad_polygons,
          f"{len(bad_polygons)} outside")
    if "run_id" in df.columns and len(df):
        check("table and geojson have the same run identity",
              all(f["properties"].get("run_id") == df.run_id.iloc[0]
                  for f in gj["features"]))

    # ---- 3. rebuild both apps from those outputs ------------------------
    print("\n3. rebuilding the apps from those outputs")
    for script, out_name in (("make_dashboard.py", "Palm2Soil_Dashboard.html"),
                             ("make_farmer_app.py", "Palm2Soil_FieldApp.html")):
        sp = src(script)
        if not check(f"{script} found", sp is not None):
            continue
        dest = os.path.join(work, out_name)
        r = subprocess.run(
            [sys.executable, sp, "--data-dir", out_dir, "-o", dest],
            capture_output=True, text=True, encoding="utf-8", env=child_env, timeout=900)
        ok = check(f"{script} runs", r.returncode == 0,
                   (r.stderr or "").strip().splitlines()[-1][:90] if r.returncode else "")
        if not ok:
            continue
        check(f"{out_name} written", os.path.exists(dest),
              f"{os.path.getsize(dest)//1024} KB" if os.path.exists(dest) else "")

        # ---- 4. every featured field must exist in the dataset ----------
        html = open(dest, encoding="utf-8").read()
        import re
        m = re.search(r'const (?:F|FIELDS) = (\[.*?\]);', html, re.S)
        if not m:
            check(f"{out_name}: embedded data found", False, "no data array")
            continue
        try:
            feats = json.loads(m.group(1))
            parsed = True
        except json.JSONDecodeError as e:
            feats, parsed = [], False
            check(f"{out_name}: embedded data parses", False, str(e)[:60])
        if parsed:
            # an empty list must not be mistaken for "nothing to check"
            if not check(f"{out_name}: embedded data is non-empty",
                         len(feats) > 0, f"{len(feats)} entries"):
                continue
            used = {int(f.get("id", f.get("p", {}).get("id", -1))) for f in feats}
            used.discard(-1)
            check(f"{out_name}: ids parsed from embedded data", bool(used),
                  f"{len(used)} ids")
            missing = used - ids_csv
            check(f"{out_name}: featured ids all exist", not missing,
                  f"{len(used)} ids referenced" if not missing
                  else f"missing {sorted(missing)[:5]}")

    # ---- 5. importer round-trip ----------------------------------------
    print("\n5. observation importer")
    imp = src("import_observations.py")
    if check("import_observations.py found", imp is not None):
        good = os.path.join(REPO, "examples", "input", "sample_field_records.json")
        bad_f = os.path.join(REPO, "examples", "input",
                             "sample_field_records_with_errors.json")
        joined = os.path.join(work, "observations_joined.csv")
        if check("sample records present", os.path.exists(good)):
            # records reference full-scene parcel ids, so join against the
            # shipped full-scene table rather than the sample run
            full_dir = os.path.join(REPO, "examples", "output")
            r = subprocess.run(
                [sys.executable, imp, good, "--data-dir", full_dir,
                 "--out", joined],
                capture_output=True, text=True, encoding="utf-8", env=child_env, timeout=600)
            check("valid records import cleanly", r.returncode == 0,
                  (r.stdout or "").strip().splitlines()[-1][:70] if r.returncode else "")
            if os.path.exists(joined):
                import pandas as _pd
                j = _pd.read_csv(joined)
                check("joined table has measurements attached",
                      "NDVI_mean" in j.columns and len(j) > 0,
                      f"{len(j)} rows")
        if check("error-case records present", os.path.exists(bad_f)):
            r2 = subprocess.run(
                [sys.executable, imp, bad_f, "--data-dir",
                 os.path.join(REPO, "examples", "output"),
                 "--out", os.path.join(work, "j2.csv")],
                capture_output=True, text=True, encoding="utf-8", env=child_env, timeout=600)
            check("bad records rejected, not silently dropped",
                  r2.returncode != 0 and "UNMATCHED" in r2.stdout,
                  "exits non-zero and lists them")

    return finish()


def finish():
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print(f"\n{'=' * 56}\n{passed}/{total} checks passed")
    failed = [n for n, ok, _ in checks if not ok]
    if failed:
        print("FAILED:")
        for n in failed:
            print(f"  - {n}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
