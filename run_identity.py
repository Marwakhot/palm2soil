"""Deterministic parcel-run identity shared by every pipeline entry point.

Only parcel geometry, its coordinate systems, scene and processing parameters
define identity. Display order, measured values and table rounding do not.
"""
import hashlib
import json
import math
from numbers import Integral


def _ring(coordinates):
    points = []
    for point in coordinates:
        if len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
            raise ValueError("Run identity requires finite two-dimensional coordinates")
        # All output polygons use WGS84. Eight decimal places preserve about
        # one millimetre while removing insignificant reprojection noise.
        xy = tuple(round(float(v), 8) or 0.0 for v in point)
        if not points or points[-1] != xy:
            points.append(xy)
    if points and points[0] == points[-1]:
        points.pop()
    if len(set(points)) < 3:
        raise ValueError("Run identity requires non-degenerate polygon rings")
    # Ring starting vertex and winding do not change the parcel.
    variants = []
    for direction in (points, list(reversed(points))):
        anchor = min(direction)
        for i, xy in enumerate(direction):
            if xy == anchor:
                variants.append(tuple(direction[i:] + direction[:i]))
    return min(variants)


def _polygon(rings):
    if not rings:
        raise ValueError("Run identity requires non-empty polygons")
    return (_ring(rings[0]), tuple(sorted(_ring(ring) for ring in rings[1:])))


def compute_run_id(scene_id, features, *, crs="EPSG:4326", source_crs=None,
                   parameters=None):
    """Return SCENE.<20 hex characters>, independent of row/ring ordering.

    ``features`` is a GeoJSON feature iterable or FeatureCollection. Each
    feature must have an integer ``properties.parcel_id`` and polygon geometry.
    Pass the same source CRS and processing parameters when migrating stored
    outputs as when producing them from the raster.
    """
    if isinstance(features, dict):
        features = features["features"]
    parcels = []
    seen = set()
    for feature in features:
        pid = feature["properties"]["parcel_id"]
        if isinstance(pid, bool) or not isinstance(pid, Integral):
            raise ValueError("parcel_id must be an integer")
        pid = int(pid)
        if pid in seen:
            raise ValueError(f"Duplicate parcel_id {pid}")
        seen.add(pid)
        geom = feature["geometry"]
        if geom["type"] == "Polygon":
            polygons = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            polygons = geom["coordinates"]
        else:
            raise ValueError("Run identity requires Polygon or MultiPolygon")
        if not polygons:
            raise ValueError("Run identity requires non-empty geometries")
        parcels.append((pid, tuple(sorted(_polygon(poly) for poly in polygons))))
    if not parcels:
        raise ValueError("Cannot identify an empty parcel dataset")
    payload = {
        "identity_version": 1,
        "scene": str(scene_id),
        "crs": str(crs).upper(),
        "source_crs": str(source_crs).upper() if source_crs is not None else None,
        "parameters": parameters or {},
        "parcels": sorted(parcels),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return f"{scene_id}.{hashlib.sha256(encoded).hexdigest()[:20]}"
