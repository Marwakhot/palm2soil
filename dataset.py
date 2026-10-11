# -*- coding: utf-8 -*-
"""Select a parcel dataset and validate its parcel, run and scene identities."""
import json
import os
import re
from decimal import Decimal, InvalidOperation


def parcel_id(value):
    """Parse a positive whole-number identifier without truncation or rounding."""
    if isinstance(value, bool) or type(value).__name__ == "bool_":
        raise ValueError("field id is a boolean, not a parcel id")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError("field id is not a number") from None
    if not number.is_finite():
        raise ValueError("field id is not finite")
    if number != number.to_integral_value():
        raise ValueError("field id is not a whole number")
    if number < 1:
        raise ValueError("field id is not positive")
    return int(number)


def scene_from_run_id(run_id):
    """Read the scene from the pipeline's ``scene.hex-digest`` run identity."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id is missing or empty")
    if run_id != run_id.strip():
        raise ValueError("run_id must not contain surrounding whitespace")
    match = re.fullmatch(r"([^\s]+)\.([0-9a-fA-F]{10,64})", run_id)
    if not match:
        raise ValueError("run_id must have the pipeline's scene.hex-digest format")
    return match.group(1)


def _validate_identity(records, source):
    """Validate every row/feature, returning (parcel IDs, run ID, scene)."""
    if not records:
        raise SystemExit(f"{source} contains no parcels.")
    ids, runs = [], set()
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise SystemExit(f"{source}, parcel {index}: properties are not an object.")
        try:
            ids.append(parcel_id(record.get("parcel_id")))
            run = record.get("run_id")
            scene = scene_from_run_id(run)
        except ValueError as exc:
            raise SystemExit(f"{source}, parcel {index}: {exc}.") from None
        if "scene_id" in record and record["scene_id"] != scene:
            raise SystemExit(
                f"{source}, parcel {index}: scene_id disagrees with run_id.")
        runs.add(run)
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{source} contains duplicate parcel IDs.")
    if len(runs) != 1:
        raise SystemExit(f"{source} mixes run IDs; use one pipeline run per dataset.")
    run = next(iter(runs))
    return set(ids), run, scene_from_run_id(run)


def validate_table_identity(table, source="parcel table"):
    """Validate all CSV rows, not just the first run ID."""
    return _validate_identity(table.to_dict("records"), source)


def _label(directory, repo):
    rel = os.path.relpath(directory, repo)
    if rel in (".", ""):
        return "repository root"
    if rel.replace(os.sep, "/") == "examples/output":
        return "bundled example output"
    return rel


def resolve_dataset(data_dir, repo, need_geojson=False):
    """Return paths and label after checking one complete dataset's identity.

    An explicit data_dir is used alone. Otherwise search the repository root,
    then examples/output. CSV-only consumers still require valid run identity.
    """
    import pandas as pd

    candidates = ([os.path.abspath(data_dir)] if data_dir else
                  [repo, os.path.join(repo, "examples", "output")])
    for directory in candidates:
        csv_path = os.path.join(directory, "parcel_table.csv")
        geojson_path = os.path.join(directory, "parcels.geojson")
        if not os.path.exists(csv_path):
            continue
        if need_geojson:
            if not os.path.exists(geojson_path):
                raise SystemExit(
                    f"{csv_path} exists but parcels.geojson does not.\n"
                    "Both must come from the same pipeline run.")
            _check_ids(csv_path, geojson_path)
            return csv_path, geojson_path, _label(directory, repo)
        validate_table_identity(pd.read_csv(csv_path, dtype={"parcel_id": str}), csv_path)
        return csv_path, _label(directory, repo)
    raise SystemExit(
        "No parcel_table.csv found. Looked in:\n  " + "\n  ".join(candidates) +
        "\n\nRun the notebook first, or pass --data-dir explicitly.")


def _check_ids(csv_path, geojson_path):
    import pandas as pd

    ids_csv, run_csv, _ = validate_table_identity(
        pd.read_csv(csv_path, dtype={"parcel_id": str}), csv_path)
    with open(geojson_path, encoding="utf-8") as handle:
        document = json.load(handle)
    features = document.get("features") if isinstance(document, dict) else None
    if not isinstance(features, list):
        raise SystemExit(f"{geojson_path} has no GeoJSON feature list.")
    properties = [feature.get("properties") if isinstance(feature, dict) else None
                  for feature in features]
    ids_geojson, run_geojson, _ = _validate_identity(properties, geojson_path)
    if run_csv != run_geojson:
        raise SystemExit(
            "parcel_table.csv and parcels.geojson have different run IDs.\n"
            "Regenerate both, or select a matching pair with --data-dir.")
    if ids_csv != ids_geojson:
        raise SystemExit(
            "parcel_table.csv and parcels.geojson disagree:\n"
            f"  {len(ids_csv)} IDs in the table, {len(ids_geojson)} in the GeoJSON\n"
            f"  {len(ids_csv - ids_geojson)} only in the table, "
            f"{len(ids_geojson - ids_csv)} only in the GeoJSON\n"
            "Regenerate both, or select a matching pair with --data-dir.")
