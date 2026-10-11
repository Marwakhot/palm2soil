# -*- coding: utf-8 -*-
"""
Join ground observations captured in the field app back onto the satellite
measurements they belong to.

This is the return leg of the loop. The field app exports JSON; this reads it,
checks every record against the parcel table, and writes one row per record
with the satellite measurements for that parcel attached. Records that cannot
be matched are reported individually and written to their own file -- never
dropped quietly, because an unmatched record usually means a real mistake
(wrong scene, stale export, field deleted by a re-run) and silence would hide
it.

    python import_observations.py exports/*.json
    python import_observations.py rec.json --data-dir examples/output

Writes observations_joined.csv and, if anything failed to match,
observations_unmatched.csv. Exit status is non-zero if any record was rejected.

WHAT THIS DOES NOT DO
  It joins; it does not calibrate. Turning observations into a dS/m model needs
  laboratory soil measurements and far more records than a prototype collects.
  The joined table is the input that work would start from.
"""
import os
import sys

# Resolve the repository root and make it importable BEFORE any local import.
REPO = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(REPO) == "src":
    REPO = os.path.dirname(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import argparse
import glob
import json
from datetime import datetime

import pandas as pd

from dataset import (parcel_id, resolve_dataset, scene_from_run_id,
                     validate_table_identity)

MEASURE_COLS = ["area_ha", "lon", "lat", "NDVI_mean", "NDRE_mean", "NDMI_mean",
                "BSI_mean", "SI_Khan_mean", "CRSI_mean", "stress_residual",
                "stress_rank", "confidence", "needs_split", "local_valid_frac",
                "local_cloud_frac", "refl_uncertainty"]
REQUIRED = ("task", "field", "scene", "run_id", "at")


def iso_ok(s):
    try:
        datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return True
    except (ValueError, TypeError):
        return False


def read_export(path):
    """-> (records, file-level scene, error or None)"""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except UnicodeDecodeError as e:
        # One badly encoded file must not take the whole batch down.
        return [], None, f"not valid UTF-8: {e}"
    except (OSError, json.JSONDecodeError) as e:
        return [], None, f"unreadable: {e}"
    if isinstance(doc, list):                      # bare list of records
        return doc, None, None
    if not isinstance(doc, dict):
        return [], None, "top level is neither an object nor a list"
    recs = doc.get("records")
    if recs is None:
        return [], None, "no 'records' key"
    if not isinstance(recs, list):
        return [], None, "'records' is not a list"
    return recs, doc.get("scene"), None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="JSON export(s) from the field app")
    ap.add_argument("--data-dir", default=None,
                    help="folder holding parcel_table.csv "
                         "(default: repo root, else examples/output)")
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument("--allow-scene-mismatch", action="store_true",
                    help="join records whose scene or run id differs from the "
                         "table's — unsafe, parcel ids repeat between runs; "
                         "valid table identity and coherent record identity "
                         "are still required")
    a = ap.parse_args()

    paths = []
    for pat in a.files:
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else [pat])
    if not paths:
        raise SystemExit("No input files.")

    csv_path, label = resolve_dataset(a.data_dir, REPO)
    tab = pd.read_csv(csv_path, dtype={"parcel_id": str})
    known, table_run, table_scene = validate_table_identity(tab, csv_path)
    tab["parcel_id"] = tab["parcel_id"].map(parcel_id)
    print(f"parcel table : {label}  ({len(tab)} parcels)")
    print(f"exports      : {len(paths)} file(s)\n")

    rows, bad = [], []
    for p in paths:
        recs, file_scene, err = read_export(p)
        if err:
            bad.append({"source": os.path.basename(p), "reason": err})
            print(f"  {os.path.basename(p)}: {err}")
            continue
        print(f"  {os.path.basename(p)}: {len(recs)} record(s)")
        for i, r in enumerate(recs):
            src = f"{os.path.basename(p)}#{i}"
            if not isinstance(r, dict):
                bad.append({"source": src, "reason": "record is not an object"})
                continue
            missing = [k for k in REQUIRED if k not in r or r[k] in (None, "")]
            if missing:
                bad.append({"source": src, "field": r.get("field"),
                            "reason": "missing field(s): " + ", ".join(missing)})
                continue
            if not iso_ok(r["at"]):
                bad.append({"source": src, "field": r.get("field"),
                            "reason": f"timestamp not ISO-8601: {r['at']!r}"})
                continue
            raw_fid = r["field"]
            try:
                fid = parcel_id(raw_fid)
            except ValueError as exc:
                bad.append({"source": src, "field": raw_fid,
                            "reason": str(exc)})
                continue
            if not isinstance(r["task"], str) or not r["task"].strip():
                bad.append({"source": src, "field": fid,
                            "reason": "task must be a nonempty string"})
                continue
            scene = r.get("scene") or file_scene
            run = r["run_id"]
            try:
                run_scene = scene_from_run_id(run)
            except ValueError as exc:
                bad.append({"source": src, "field": fid, "scene": scene,
                            "reason": str(exc)})
                continue
            if scene != run_scene:
                bad.append({"source": src, "field": fid, "scene": scene,
                            "reason": f"scene {scene!r} disagrees with "
                                      f"record run {run}"})
                continue
            if not a.allow_scene_mismatch and (run != table_run or scene != table_scene):
                bad.append({"source": src, "field": fid, "scene": scene,
                            "reason": f"record run {run} != table run {table_run} "
                                      "(use --allow-scene-mismatch to override)"})
                continue
            if fid not in known:
                bad.append({"source": src, "field": fid, "scene": scene,
                            "reason": f"parcel {fid} not in this table "
                                      f"(ids 1..{max(known)})"})
                continue
            rows.append({"source": os.path.basename(p), "parcel_id": fid,
                         "task": r["task"], "observed_at": r["at"],
                         "record_scene": scene,
                         "record_run_id": run,
                         "scan_date": r.get("scan_date", "")})

    if not rows and not bad:
        bad.append({"source": "input exports",
                    "reason": "No records found in any input file."})

    out = pd.DataFrame(rows)

    # --- duplicates ------------------------------------------------------
    if not out.empty:
        key = ["parcel_id", "task", "observed_at", "record_run_id"]
        dupes = out.duplicated(subset=key, keep="first")
        if dupes.any():
            print(f"\n  {int(dupes.sum())} duplicate record(s) collapsed "
                  "(same parcel, task and timestamp)")
            out = out[~dupes]

    joined = pd.DataFrame()
    if not out.empty:
        cols = ["parcel_id"] + [c for c in MEASURE_COLS if c in tab.columns]
        joined = out.merge(tab[cols], on="parcel_id", how="left")
        joined = joined.sort_values(["parcel_id", "task"]).reset_index(drop=True)

    out_path = a.out or os.path.join(REPO, "observations_joined.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    bad_path = os.path.join(os.path.dirname(os.path.abspath(out_path)),
                            "observations_unmatched.csv")
    # Both files are rewritten on every completed import, empty or not, so a
    # previous run's results can never be mistaken for this one's.
    (joined if not joined.empty else
     pd.DataFrame(columns=["source", "parcel_id", "task", "observed_at"])
     ).to_csv(out_path, index=False)
    (pd.DataFrame(bad) if bad else
     pd.DataFrame(columns=["source", "field", "scene", "reason"])
     ).to_csv(bad_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"matched   : {len(joined)} record(s) across "
          f"{joined.parcel_id.nunique() if not joined.empty else 0} parcel(s)")
    print(f"unmatched : {len(bad)}")
    print(f"written   : {out_path}")
    if not joined.empty:
        print(f"by task   : {joined.task.value_counts().to_dict()}")

    if bad:
        print(f"\nUNMATCHED RECORDS — not discarded, written to {bad_path}")
        for b in bad:
            print(f"  {b.get('source','?'):28} {b['reason']}")

    if not joined.empty:
        print("\nNOTE  This table joins observations to measurements. It is not "
              "a calibration.\n      Producing a dS/m model needs laboratory "
              "soil values and far more\n      records than a prototype "
              "collects. This is where that work starts.")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
