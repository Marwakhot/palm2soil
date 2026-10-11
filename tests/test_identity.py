"""Regression tests for dataset identity and the field-observation import loop."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dataset
import import_observations

SCENE = "scene_A"
RUN = SCENE + ".0123456789abcdef01234"
OTHER = SCENE + ".fedcba9876543210abcd"


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="palm2soil_identity_")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.table = pd.DataFrame([
            {"parcel_id": 1, "run_id": RUN, "area_ha": 2.5},
            {"parcel_id": 2, "run_id": RUN, "area_ha": 3.0},
        ])
        self.features = [
            {"type": "Feature", "properties": {"parcel_id": row.parcel_id,
              "run_id": row.run_id}, "geometry": None}
            for row in self.table.itertuples(index=False)
        ]
        self.write_pair()

    def write_pair(self):
        self.table.to_csv(self.directory / "parcel_table.csv", index=False)
        (self.directory / "parcels.geojson").write_text(json.dumps({
            "type": "FeatureCollection", "features": self.features}), encoding="utf-8")

    def resolve(self, geojson=True):
        return dataset.resolve_dataset(str(self.directory), str(ROOT), geojson)

    def record(self, **changes):
        value = {"field": 1, "task": "photo", "scene": SCENE, "run_id": RUN,
                 "at": "2026-10-11T09:00:00Z"}
        value.update(changes)
        return value

    def do_import(self, records, flags=()):
        export = self.directory / "records.json"
        export.write_text(json.dumps({"records": records}), encoding="utf-8")
        args = ["import_observations.py", str(export), "--data-dir",
                str(self.directory), "--out", str(self.directory / "joined.csv"), *flags]
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            result = import_observations.main()
        return (result, pd.read_csv(self.directory / "joined.csv"),
                pd.read_csv(self.directory / "observations_unmatched.csv"))

    def test_valid_pair_and_legacy_run_id(self):
        self.assertEqual(len(self.resolve()), 3)
        self.assertEqual(len(self.resolve(False)), 2)
        self.table["run_id"] = SCENE + ".0123456789"
        for feature in self.features:
            feature["properties"]["run_id"] = SCENE + ".0123456789"
        self.write_pair()
        self.resolve()

    def test_matching_ids_with_different_runs_are_rejected(self):
        for feature in self.features:
            feature["properties"]["run_id"] = OTHER
        self.write_pair()
        with self.assertRaisesRegex(SystemExit, "different run IDs"):
            self.resolve()

    def test_csv_run_identity_required_for_every_row(self):
        original = self.table.copy()
        for invalid in (None, "", " ", OTHER, "invalid-run"):
            with self.subTest(value=invalid):
                self.table = original.copy()
                self.table.loc[1, "run_id"] = invalid
                self.write_pair()
                with self.assertRaises(SystemExit):
                    self.resolve(False)
        self.table = original.drop(columns="run_id")
        self.write_pair()
        with self.assertRaises(SystemExit):
            self.resolve(False)

    def test_geojson_run_identity_required_for_every_feature(self):
        original = copy.deepcopy(self.features)
        for invalid in (None, "", " ", OTHER, "invalid-run"):
            with self.subTest(value=invalid):
                self.features = copy.deepcopy(original)
                self.features[1]["properties"]["run_id"] = invalid
                self.write_pair()
                with self.assertRaises(SystemExit):
                    self.resolve()
        self.features = copy.deepcopy(original)
        del self.features[1]["properties"]["run_id"]
        self.write_pair()
        with self.assertRaises(SystemExit):
            self.resolve()

    def test_duplicate_parcel_ids_are_rejected_in_either_file(self):
        self.table.loc[1, "parcel_id"] = 1
        self.write_pair()
        with self.assertRaisesRegex(SystemExit, "duplicate parcel"):
            self.resolve(False)
        self.table.loc[1, "parcel_id"] = 2
        self.features[1]["properties"]["parcel_id"] = 1
        self.write_pair()
        with self.assertRaisesRegex(SystemExit, "duplicate parcel"):
            self.resolve()

    def test_noninteger_csv_and_geojson_ids_are_rejected(self):
        precise_fraction = "2.00000000000000000001"
        for invalid in (True, "NaN", "Infinity", "1.2", precise_fraction, 0):
            with self.subTest(value=invalid):
                self.table["parcel_id"] = [1, invalid]
                self.write_pair()
                with self.assertRaises(SystemExit) as rejected:
                    self.resolve(False)
                if invalid == precise_fraction:
                    self.assertIn("not a whole number", str(rejected.exception))
                self.table["parcel_id"] = [1, 2]
                self.features[1]["properties"]["parcel_id"] = invalid
                self.write_pair()
                with self.assertRaises(SystemExit) as rejected:
                    self.resolve()
                if invalid == precise_fraction:
                    self.assertIn("not a whole number", str(rejected.exception))
        self.features[1]["properties"]["parcel_id"] = 2

    def test_mismatched_parcel_sets_are_rejected(self):
        self.features[1]["properties"]["parcel_id"] = 3
        self.write_pair()
        with self.assertRaisesRegex(SystemExit, "disagree"):
            self.resolve()

    def test_optional_scene_id_must_agree_with_run(self):
        self.table["scene_id"] = [SCENE, "OTHER_SCENE"]
        self.write_pair()
        with self.assertRaisesRegex(SystemExit, "scene_id disagrees"):
            self.resolve(False)
        self.table["scene_id"] = SCENE
        self.features[1]["properties"]["scene_id"] = "OTHER_SCENE"
        self.write_pair()
        with self.assertRaisesRegex(SystemExit, "scene_id disagrees"):
            self.resolve()

    def test_wrong_scene_rejected_without_summary(self):
        self.assertFalse((self.directory / "parcel_summary.txt").exists())
        result, joined, bad = self.do_import([self.record(scene="OTHER_SCENE")])
        self.assertEqual((result, len(joined), len(bad)), (1, 0, 1))
        self.assertIn("disagrees", bad.iloc[0].reason)
        result, joined, bad = self.do_import([self.record()])
        self.assertEqual((result, len(joined), len(bad)), (0, 1, 0))

    def test_summary_cannot_override_authoritative_identity(self):
        (self.directory / "parcel_summary.txt").write_text("Scene OTHER_SCENE\n")
        result, joined, bad = self.do_import([self.record()])
        self.assertEqual((result, len(joined), len(bad)), (0, 1, 0))

    def test_missing_and_wrong_record_run_are_rejected(self):
        missing = self.record()
        del missing["run_id"]
        result, joined, bad = self.do_import([missing, self.record(run_id=OTHER)])
        self.assertEqual((result, len(joined), len(bad)), (1, 0, 2))

    def test_override_allows_coherent_other_run_only(self):
        record = self.record(scene="scene_B", run_id="scene_B.0123456789abcdef01234")
        self.assertEqual(self.do_import([record])[0], 1)
        result, joined, bad = self.do_import([record], ["--allow-scene-mismatch"])
        self.assertEqual((result, len(joined), len(bad)), (0, 1, 0))
        result, joined, bad = self.do_import(
            [self.record(scene="scene_B")], ["--allow-scene-mismatch"])
        self.assertEqual((result, len(joined), len(bad)), (1, 0, 1))

    def test_override_does_not_bypass_table_identity(self):
        self.table.loc[1, "run_id"] = None
        self.write_pair()
        with self.assertRaises(SystemExit):
            self.do_import([self.record()], ["--allow-scene-mismatch"])

    def test_malformed_field_ids_reject_without_crashing(self):
        invalid = [True, False, "NaN", "Infinity", 1.2, -1, 0, [], {}]
        result, joined, bad = self.do_import([self.record(field=value) for value in invalid])
        self.assertEqual((result, len(joined), len(bad)), (1, 0, len(invalid)))

    def test_repeated_records_are_deduplicated_after_validation(self):
        records = [self.record(scene="WRONG"), self.record(), self.record()]
        result, joined, bad = self.do_import(records)
        self.assertEqual((result, len(joined), len(bad)), (1, 1, 1))

    def test_output_cleanup_for_invalid_and_empty_imports(self):
        result, joined, bad = self.do_import([self.record()])
        self.assertEqual((result, len(joined), len(bad)), (0, 1, 0))
        result, joined, bad = self.do_import([self.record(field="NaN")])
        self.assertEqual((result, len(joined), len(bad)), (1, 0, 1))
        result, joined, bad = self.do_import([self.record()])
        self.assertEqual((result, len(joined), len(bad)), (0, 1, 0))
        result, joined, bad = self.do_import([])
        self.assertEqual((result, len(joined), len(bad)), (1, 0, 1))


if __name__ == "__main__":
    unittest.main()
