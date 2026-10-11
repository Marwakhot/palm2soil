"""Regression checks for normal and unavailable inspection-ranking states.

Run with: python -m unittest discover -s tests -p test_apps.py
Node.js is optional and enables isolated JavaScript rendering checks; no browser
or network is used. Builders run with Windows-compatible cp1252 console settings.
"""
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


class AppEligibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="palm2soil_apps_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        source = REPO / "examples" / "output"
        with (source / "parcel_table.csv").open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cls.columns = reader.fieldnames
            original = list(reader)
        cls.parcel_count = len(original)
        geojson = json.loads((source / "parcels.geojson").read_text(encoding="utf-8"))
        cls.pages = {}
        for case in ("normal", "all_low", "all_merged"):
            folder = cls.root / case
            folder.mkdir()
            rows = [dict(row) for row in original]
            shapes = json.loads(json.dumps(geojson))
            if case != "normal":
                key, value = (("confidence", "low") if case == "all_low"
                              else ("needs_split", True))
                for row in rows:
                    row[key] = value
                for feature in shapes["features"]:
                    feature["properties"][key] = value
            with (folder / "parcel_table.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=cls.columns)
                writer.writeheader()
                writer.writerows(rows)
            (folder / "parcels.geojson").write_text(json.dumps(shapes), encoding="utf-8")
            env = dict(os.environ, PYTHONUTF8="0", PYTHONIOENCODING="cp1252")
            for app in ("dashboard", "farmer_app"):
                output = folder / f"{app}.html"
                result = subprocess.run(
                    [sys.executable, "-X", "utf8=0", str(REPO / "src" / f"make_{app}.py"),
                     "--data-dir", str(folder), "-o", str(output)],
                    env=env, capture_output=True, timeout=60,
                )
                if result.returncode:
                    raise AssertionError(
                        f"{case}/{app} failed:\n{result.stderr.decode('cp1252', errors='replace')}")
                cls.pages[case, app] = output.read_text(encoding="utf-8")

    def fields(self, case):
        match = re.search(r"const FIELDS = (.*);", self.pages[case, "farmer_app"])
        return json.loads(match.group(1))

    def test_normal_builds_keep_ranking_and_dynamic_fields(self):
        html = self.pages["normal", "dashboard"]
        count = re.search(r"Usable for ranking</div>\s*<strong>(\d+)</strong>", html)
        self.assertGreater(int(count.group(1)), 0)
        self.assertIn("const RANKING_AVAILABLE = true;", html)
        self.assertNotIn("<b>No eligible parcels.</b>", html)
        fields = self.fields("normal")
        self.assertEqual([field["tier"] for field in fields],
                         ["Priority 1", "Priority 2", "Priority 3"])
        self.assertTrue(all(field["conf"] != "low" for field in fields))
        self.assertIn("One field flagged for a ground check · 3 shown",
                      self.pages["normal", "farmer_app"])

    def test_rejected_datasets_preserve_zero_eligible_and_no_featured_fields(self):
        for case in ("all_low", "all_merged"):
            with self.subTest(case=case):
                dashboard = self.pages[case, "dashboard"]
                self.assertRegex(dashboard, r"Usable for ranking</div>\s*<strong>0</strong>")
                self.assertIn("<b>No eligible parcels.</b>", dashboard)
                self.assertIn("const P20 = null, P40 = null;", dashboard)
                self.assertIn("const RANKING_AVAILABLE = false;", dashboard)
                self.assertNotIn("const P20 = nan", dashboard)
                self.assertEqual(self.fields(case), [])
                self.assertIn("No fields eligible for a ground check · 0 shown",
                              self.pages[case, "farmer_app"])

    def run_script(self, case, app, expression):
        html = self.pages[case, app]
        script = re.search(r"<script>([\s\S]*?)</script>", html).group(1)
        # Minimal isolated DOM stand-ins exercise the generated application code.
        harness = r"""
const vm = require('vm');
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const elements = {};
const document = {
  getElementById(id) { return elements[id] || (elements[id] = {
    innerHTML: '', textContent: '', style: {}, className: '',
    querySelectorAll() { return []; }, addEventListener() {}, setAttribute() {}
  }); },
  querySelectorAll() { return []; }
};
const saved = {};
const localStorage = { getItem(k) { return saved[k] || null; },
                       setItem(k, v) { saved[k] = v; } };
const context = vm.createContext({ document, localStorage, elements });
vm.runInContext(input.script, context);
process.stdout.write(JSON.stringify(vm.runInContext(input.expression, context)));
"""
        result = subprocess.run(
            [NODE, "-e", harness],
            input=json.dumps({"script": script, "expression": expression}),
            encoding="utf-8", capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    @unittest.skipUnless(NODE, "Node.js is optional for generated JavaScript checks")
    def test_no_eligible_dashboard_never_renders_priority_or_rank(self):
        for case in ("all_low", "all_merged"):
            with self.subTest(case=case):
                state = self.run_script(case, "dashboard", """
                  const initialMap = elements.map.innerHTML;
                  const initialRows = elements['d-rows'].innerHTML;
                  layer = LAYERS[1]; hideSplit = false; select(0);
                  ({initialMap, initialRows, rows:elements['d-rows'].innerHTML,
                    inspectedMeasurements:elements.map.innerHTML.includes('<path'),
                    band:band(F[0].p.stress_residual)[0]});
                """)
                self.assertIn("No parcels eligible for ranking", state["initialMap"])
                self.assertTrue(state["inspectedMeasurements"])
                self.assertEqual(state["band"], "Unavailable")
                for rows in (state["initialRows"], state["rows"]):
                    self.assertNotIn("Inspection priority", rows)
                    self.assertNotIn("<span>Rank</span>", rows)
                    self.assertNotRegex(rows, r"Priority [123]")

    @unittest.skipUnless(NODE, "Node.js is optional for generated JavaScript checks")
    def test_no_eligible_field_app_disables_capture_and_export(self):
        for case in ("all_low", "all_merged"):
            with self.subTest(case=case):
                state = self.run_script(case, "farmer_app", """
                  exportRecords();
                  ({fieldId:FIELD_ID, fields:elements.fieldlist.innerHTML,
                    tasks:elements.tasklist.innerHTML, label:elements.barlab.textContent,
                    exportDisplay:elements.exportrow.style.display, records});
                """)
                self.assertIsNone(state["fieldId"])
                self.assertIn("No eligible parcels", state["fields"])
                self.assertEqual(state["tasks"], "")
                self.assertEqual(state["label"], "Capture unavailable")
                self.assertEqual(state["exportDisplay"], "none")
                self.assertEqual(state["records"], {})

    @unittest.skipUnless(NODE, "Node.js is optional for generated JavaScript checks")
    def test_normal_javascript_renders_rank_and_keeps_correct_palette(self):
        dashboard = self.run_script("normal", "dashboard", """
          ({rows:elements['d-rows'].innerHTML, reverse:LAYERS[0].rev,
            low:ramp('diverging',0), high:ramp('diverging',1)});
        """)
        self.assertIn("<span>Rank</span>", dashboard["rows"])
        self.assertIn("Inspection priority", dashboard["rows"])
        self.assertFalse(dashboard["reverse"])
        self.assertEqual(dashboard["low"], "rgb(178,24,43)")
        self.assertEqual(dashboard["high"], "rgb(33,102,172)")
        capture = self.run_script("normal", "farmer_app", """
          ({fieldId:FIELD_ID, tasks:elements.tasklist.innerHTML,
            key:storeKey(FIELD_ID), run:RUN_ID});
        """)
        self.assertIsNotNone(capture["fieldId"])
        self.assertIn("Mark done", capture["tasks"])
        self.assertIn(capture["run"], capture["key"])


if __name__ == "__main__":
    unittest.main()
