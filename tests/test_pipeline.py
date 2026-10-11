"""Regressions for run identity, input georeferencing and notebook generation."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from run_identity import compute_run_id
from smoke_test import footprint_violations


def feature(pid, coordinates):
    return {"type": "Feature", "properties": {"parcel_id": pid},
            "geometry": {"type": "Polygon", "coordinates": [coordinates]}}


class RunIdentityTests(unittest.TestCase):
    def setUp(self):
        self.square = feature(1, [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
        self.other = feature(2, [[3, 3], [4, 3], [4, 4], [3, 4], [3, 3]])

    def identity(self, features, **kwargs):
        return compute_run_id("scene", features, **kwargs)

    def test_same_geometry_has_same_id_despite_row_ring_and_property_order(self):
        changed = copy.deepcopy(self.square)
        ring = list(reversed(changed["geometry"]["coordinates"][0][:-1]))
        ring = ring[2:] + ring[:2]
        changed["geometry"]["coordinates"] = [ring + [ring[0]]]
        changed["properties"].update(lon=123.4567, area_ha=987.6,
                                     stress_residual=-0.2, run_id="old")
        self.assertEqual(self.identity([self.square, self.other]),
                         self.identity([self.other, changed]))

    def test_same_centroid_area_but_different_shape_has_different_id(self):
        rectangle = feature(1, [[-2, -.5], [2, -.5], [2, .5], [-2, .5], [-2, -.5]])
        self.assertNotEqual(self.identity([self.square]), self.identity([rectangle]))

    def test_crs_parameters_scene_and_parcel_id_are_part_of_identity(self):
        baseline = self.identity([self.square], source_crs="EPSG:32636", parameters={"min_pixels": 20})
        self.assertNotEqual(baseline, self.identity([self.square], source_crs="EPSG:32637", parameters={"min_pixels": 20}))
        self.assertNotEqual(baseline, self.identity([self.square], source_crs="EPSG:32636", parameters={"min_pixels": 25}))
        self.assertNotEqual(baseline, compute_run_id("another-scene", [self.square], source_crs="EPSG:32636", parameters={"min_pixels": 20}))
        moved_id = copy.deepcopy(self.square)
        moved_id["properties"]["parcel_id"] = 3
        self.assertNotEqual(self.identity([self.square]), self.identity([moved_id]))

    def test_geometry_shift_changes_id(self):
        shifted = copy.deepcopy(self.square)
        shifted["geometry"]["coordinates"][0] = [[x + .005, y] for x, y in shifted["geometry"]["coordinates"][0]]
        self.assertNotEqual(self.identity([self.square]), self.identity([shifted]))

    def test_duplicate_ids_and_nonfinite_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            self.identity([self.square, self.square])
        malformed = copy.deepcopy(self.square)
        malformed["geometry"]["coordinates"][0][0][0] = float("nan")
        with self.assertRaises(ValueError):
            self.identity([malformed])

    def test_bundled_full_scene_identity_matches_shared_helper(self):
        with (REPO / "examples/output/parcels.geojson").open(encoding="utf-8") as handle:
            gj = json.load(handle)
        expected = compute_run_id(
            "20250608_091605_90_4001", gj, source_crs="EPSG:32636",
            parameters={"NDVI_CROP_MIN": .35, "MIN_PIXELS": 20,
                        "GRAD_PCT": 50, "MIN_MARKER": 4, "BIG_PARCEL_HA": 50})
        table = pd.read_csv(REPO / "examples/output/parcel_table.csv")
        self.assertEqual(set(table.run_id), {expected})
        self.assertEqual({f["properties"]["run_id"] for f in gj["features"]}, {expected})
        with (REPO / "examples/input/sample_field_records.json").open(encoding="utf-8") as handle:
            records = json.load(handle)
        self.assertEqual(records["run_id"], expected)
        self.assertEqual({record["run_id"] for record in records["records"]}, {expected})


class FootprintTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.h5 = Path(self.directory.name) / "input.h5"
        with h5py.File(self.h5, "w") as handle:
            handle.create_group("HDFEOS/GRIDS/HYP").attrs["epsg_code"] = 32636
            info = handle.create_group("HDFEOS INFORMATION")
            info.create_dataset("StructMetadata.0", data=np.bytes_(
                "UpperLeftPointMtrs=(500000.0,4100000.0)\n"
                "LowerRightMtrs=(500600.0,4099400.0)"))
        self.to_wgs = Transformer.from_crs(32636, 4326, always_xy=True).transform
        lon, lat = self.to_wgs(500300, 4099700)
        self.table = pd.DataFrame({"parcel_id": [1], "lon": [lon], "lat": [lat]})

    def geometries(self, dx=0):
        ring = [self.to_wgs(x + dx, y) for x, y in
                [(500450, 4099450), (500550, 4099450), (500550, 4099550),
                 (500450, 4099550), (500450, 4099450)]]
        return {"type": "FeatureCollection", "features": [feature(1, ring)]}

    def test_unmodified_geometry_uses_h5_bounds_instead_of_hardcoded_origin(self):
        self.assertEqual(footprint_violations(self.table, self.geometries(), self.h5), ([], []))

    def test_displaced_polygons_rejected_when_table_centroids_are_unchanged(self):
        self.assertEqual(footprint_violations(self.table, self.geometries(dx=440), self.h5), ([], [1]))

    def test_displaced_centroids_rejected_when_polygons_are_unchanged(self):
        moved = self.table.copy()
        moved.loc[0, ["lon", "lat"]] = self.to_wgs(500740, 4099700)
        self.assertEqual(footprint_violations(moved, self.geometries(), self.h5), ([1], []))


class NotebookSourceTests(unittest.TestCase):
    def test_generator_reproduces_all_shipped_cell_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            script = root / "src/build_palm2soil_notebook.py"
            shutil.copy2(REPO / "src/build_palm2soil_notebook.py", script)
            subprocess.run([sys.executable, str(script)], check=True,
                           capture_output=True, encoding="utf-8")
            name = "07_palm2soil_salinity_prioritisation.ipynb"
            generated = json.loads((root / name).read_text(encoding="utf-8"))
            shipped = json.loads((REPO / name).read_text(encoding="utf-8"))
        self.assertEqual(len(shipped["cells"]), 33)
        self.assertEqual([(c["cell_type"], c["source"]) for c in generated["cells"]],
                         [(c["cell_type"], c["source"]) for c in shipped["cells"]])


if __name__ == "__main__":
    unittest.main()
