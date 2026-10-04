"""Unit tests for src/validators.py and src/tune_validator.py (hand-made fixtures, no team data).

    python -m unittest discover -s tests -v
"""
import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tune_validator import main as tune_main  # noqa: E402
from src.validators import (DUPLICATION, FUTURE_AGGREGATION, LABEL_RELATION, NO_LEAKAGE,  # noqa: E402
                            TARGET_DERIVED, TEMPORAL, ImprovedValidator, ValidatorConfig,
                            column_identifiers, cosine_similarity, parse_timestamp)

CLEAN = {
    "case_id": "c0",
    "prediction_time": "2026-01-01T10:00:00Z",
    "feature_time": "2026-01-01T09:00:00Z",
    "aggregation_window_end": "2026-01-01T10:00:00Z",
    "source_max_event_time": "2026-01-01T09:30:00Z",
    "feature_expression": "SUM(t.amount) FILTER (WHERE t.event_time < :prediction_time)",
    "lineage_columns": '["transactions.amount", "transactions.event_time"]',
    "source_role": "feature",
    "source_table": "transactions",
    "exact_duplicate_hash": "aaa",
    "other_split_hash": "bbb",
    "feature_vector": "[1.0, 0.0, 2.0]",
    "other_split_vector": "[0.0, 1.0, 0.0]",
}


def row(**overrides):
    r = dict(CLEAN)
    r.update(overrides)
    return r


class GateTests(unittest.TestCase):
    def setUp(self):
        self.v = ImprovedValidator(ValidatorConfig())

    def test_clean_row_passes(self):
        out = self.v.validate(CLEAN)
        self.assertFalse(out["flagged"])
        self.assertEqual(out["root_cause"], NO_LEAKAGE)
        self.assertEqual(out["reasons"], [])

    def test_temporal(self):
        out = self.v.validate(row(feature_time="2026-01-01T12:00:00Z"))
        self.assertEqual(out["root_cause"], TEMPORAL)
        self.assertIn("feature_time 2026-01-01T12:00:00+00:00 exceeds prediction_time", out["reasons"][0])

    def test_temporal_normalises_timezones(self):
        # 10:00+07:00 is 03:00 UTC, before the 05:00 UTC prediction time
        out = self.v.validate(row(prediction_time="2026-01-01T05:00:00Z", feature_time="2026-01-01T10:00:00+07:00"))
        self.assertNotIn(TEMPORAL, out["detected_causes"])
        # 05:00+07:00 is 22:00 UTC the day before, so a 00:00 UTC feature is in the future
        out = self.v.validate(row(prediction_time="2026-01-01T05:00:00+07:00", feature_time="2026-01-01T00:00:00Z"))
        self.assertIn(TEMPORAL, out["detected_causes"])

    def test_future_aggregation_with_valid_feature_time(self):
        out = self.v.validate(row(aggregation_window_end="2026-01-01T12:00:00Z"))
        self.assertEqual(out["detected_causes"], [FUTURE_AGGREGATION])
        self.assertTrue(out["reasons"][0].startswith("aggregation_window_end 2026-01-01T12:00:00+00:00 exceeds"))
        out = self.v.validate(row(source_max_event_time="2026-01-02T00:00:00Z"))
        self.assertEqual(out["root_cause"], FUTURE_AGGREGATION)

    def test_window_end_equal_to_prediction_time_is_allowed(self):
        self.assertFalse(self.v.validate(row(aggregation_window_end="2026-01-01T10:00:00Z"))["flagged"])

    def test_target_lineage(self):
        out = self.v.validate(row(lineage_columns='["labels.churn_label", "transactions.amount"]'))
        self.assertEqual(out["root_cause"], TARGET_DERIVED)
        self.assertIn("target identifier 'churn_label' found in lineage_columns", out["reasons"])
        out = self.v.validate(row(feature_expression="AVG(l.churnLabel) OVER (PARTITION BY customer_id)"))
        self.assertIn("target identifier 'churn_label' found in feature_expression", out["reasons"])

    def test_target_lineage_avoids_broad_matches(self):
        for expr in ("COUNT(shipping_label)", "label_printer_id + 1", "status = 'label'", "x -- label here"):
            self.assertFalse(self.v.validate(row(feature_expression=expr))["flagged"], expr)
        # a label-table qualifier on a join key is not a target column
        self.assertFalse(self.v.validate(row(lineage_columns="labels.customer_id, transactions.amount"))["flagged"])

    def test_known_alias_failure_is_not_hardcoded(self):
        # semantic aliases without metadata are a documented limitation
        self.assertFalse(self.v.validate(row(lineage_columns='["tickets.resolution_code"]'))["flagged"])

    def test_label_relation(self):
        out = self.v.validate(row(source_role="Label"))
        self.assertEqual(out["root_cause"], LABEL_RELATION)
        self.assertIn("source_role='Label'", out["reasons"])
        out = self.v.validate(row(source_table="ml.training_labels"))
        self.assertIn("source_table 'ml.training_labels' is a governed label table", out["reasons"])
        for table in ("order_status", "customer_status_history", "shipping_labels_printed"):
            self.assertFalse(self.v.validate(row(source_table=table))["flagged"], table)

    def test_exact_duplicate(self):
        out = self.v.validate(row(exact_duplicate_hash="ABC", other_split_hash="abc "))
        self.assertEqual(out["root_cause"], DUPLICATION)
        self.assertEqual(out["scores"]["duplicate_kind"], "exact")

    def test_near_duplicate_uses_threshold(self):
        near = row(feature_vector="[1.0, 2.0, 3.0]", other_split_vector="[1.0, 2.0, 3.001]")
        out = self.v.validate(near)
        self.assertEqual(out["root_cause"], DUPLICATION)
        self.assertGreater(out["scores"]["cosine_similarity"], 0.9999)
        strict = ImprovedValidator(ValidatorConfig(near_duplicate_threshold=1.0))
        self.assertFalse(strict.validate(near)["flagged"])

    def test_multi_cause_priority_keeps_all_causes(self):
        out = self.v.validate(row(feature_time="2026-01-01T11:00:00Z", source_role="label",
                                  exact_duplicate_hash="x", other_split_hash="x"))
        self.assertEqual(out["root_cause"], TEMPORAL)
        self.assertEqual(out["detected_causes"], [TEMPORAL, LABEL_RELATION, DUPLICATION])
        self.assertEqual(len(out["reasons"]), 3)


class GroundTruthIsolationTests(unittest.TestCase):
    def test_ground_truth_fields_are_ignored(self):
        v = ImprovedValidator(ValidatorConfig())
        poisoned = row(ground_truth="1", is_leakage="true", leakage_type="temporal_leakage", severity="high")
        out = v.validate(poisoned)
        self.assertFalse(out["flagged"])
        self.assertEqual(sorted(out["debug_metadata"]["ignored_fields"]),
                         ["ground_truth", "is_leakage", "leakage_type", "severity"])
        base = v.validate(CLEAN)
        for key in ("flagged", "root_cause", "reasons", "detected_causes", "scores", "warnings"):
            self.assertEqual(out[key], base[key])

    def test_config_cannot_map_a_check_onto_ground_truth(self):
        cfg = ValidatorConfig()
        cfg.columns.source_role = "leakage_type"
        with self.assertRaises(ValueError):
            ImprovedValidator(cfg)


class RobustnessTests(unittest.TestCase):
    def test_bad_inputs_do_not_crash(self):
        v = ImprovedValidator(ValidatorConfig())
        nasty = [
            {},
            {"case_id": "x"},
            row(prediction_time=None, feature_time=float("nan")),
            row(feature_time="", aggregation_window_end="not a date"),
            row(feature_vector="[1, 'a']", other_split_vector="{bad json"),
            row(feature_vector="[1, 2]", other_split_vector="[1, 2, 3]"),
            row(feature_vector="[0, 0]", other_split_vector="[1, 2]"),
            row(feature_vector="[NaN, 1]", other_split_vector="[1, 1]"),
            row(lineage_columns="[broken", feature_expression=None, source_role=float("nan"), source_table=""),
        ]
        for r in nasty:
            out = v.validate(r)
            self.assertIn(out["root_cause"], (NO_LEAKAGE, TEMPORAL, FUTURE_AGGREGATION))
            self.assertIsInstance(out["warnings"], list)
        out = v.validate(row(feature_time="", prediction_time=None))
        self.assertTrue(any("temporal check skipped" in w for w in out["warnings"]))
        out = v.validate(row(feature_vector="[1, 2]", other_split_vector="[1, 2, 3]"))
        self.assertTrue(any("dims differ" in w for w in out["warnings"]))

    def test_out_of_range_timestamps_become_warnings(self):
        v = ImprovedValidator(ValidatorConfig())
        for bad in ("-86400000000000", 1e20, float("inf"), "0001-01-01T00:00:00+01:00", "9999-12-31T23:30:00-01:00"):
            out = v.validate(row(feature_time=bad, aggregation_window_end=bad))
            self.assertNotIn(TEMPORAL, out["detected_causes"], bad)
            self.assertTrue(any("out of range" in w for w in out["warnings"]), bad)
        # negative epochs are valid (pre-1970) and parse the same on every OS
        self.assertEqual(parse_timestamp("-86400"), parse_timestamp("1969-12-31T00:00:00Z"))

    def test_cosine_is_stable_for_extreme_magnitudes(self):
        self.assertAlmostEqual(cosine_similarity([1e308, 1e308], [1e308, 1e308]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1e-320, 0.0], [3e-320, 0.0]), 1.0)
        self.assertIsNone(cosine_similarity([0.0, 0.0], [1.0, 2.0]))
        out = ImprovedValidator(ValidatorConfig()).validate(
            row(feature_vector="[1e308, 1e308]", other_split_vector="[1e308, 1e308]"))
        self.assertEqual(out["root_cause"], DUPLICATION)

    def test_failing_check_is_isolated(self):
        v = ImprovedValidator(ValidatorConfig())

        def boom(_view):
            raise RuntimeError("bad field")
        v._checks = [("temporal", boom)] + [c for c in v._checks if c[0] != "temporal"]
        with self.assertLogs("src.validators", level="WARNING"):
            out = v.validate(row(source_role="label"))
        self.assertEqual(out["root_cause"], LABEL_RELATION)       # the other gates still ran
        self.assertEqual(out["debug_metadata"]["checks_failed"], ["temporal"])
        self.assertTrue(any("temporal check failed (RuntimeError" in w for w in out["warnings"]))

    def test_missing_duplication_metadata_warns(self):
        v = ImprovedValidator(ValidatorConfig())
        out = v.validate(row(exact_duplicate_hash=None, other_split_hash="", feature_vector=None))
        self.assertIn("duplication check skipped: no hash pair and no vector pair", out["warnings"])
        # hashes compared (and differ) but no vector: a normal non-duplicate, no warning
        out = v.validate(row(feature_vector=None, other_split_vector=None))
        self.assertFalse(any("duplication" in w for w in out["warnings"]))

    def test_warnings_are_deduplicated(self):
        out = ImprovedValidator(ValidatorConfig()).validate(row(prediction_time="garbage"))
        self.assertEqual(len(out["warnings"]), len(set(out["warnings"])))

    def test_rejects_non_mapping(self):
        with self.assertRaises(TypeError):
            ImprovedValidator(ValidatorConfig()).validate(["not", "a", "row"])

    def test_timestamp_formats(self):
        ref = parse_timestamp("2026-01-01T00:00:00Z")
        self.assertEqual(parse_timestamp("2026-01-01 00:00:00"), ref)
        self.assertEqual(parse_timestamp("2026-01-01T07:00:00+07:00"), ref)
        self.assertEqual(parse_timestamp(str(int(ref.timestamp()))), ref)
        self.assertEqual(parse_timestamp(ref.timestamp() * 1000), ref)
        self.assertIsNone(parse_timestamp("NaT"))
        self.assertTrue(math.isclose(parse_timestamp("2026/01/01 00:00").timestamp(), ref.timestamp()))

    def test_tokenizer(self):
        self.assertEqual(column_identifiers("SUM(l.churnLabel) + 'label' -- target"), ["sum", "churn_label"])


class TunerTests(unittest.TestCase):
    def _write(self, path, rows):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

    def test_refuses_heldout_path(self):
        with self.assertRaises(SystemExit):
            tune_main(["--dev", "data/heldout_cases.csv"])

    def test_tunes_threshold_and_freezes_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = [row(case_id=f"clean{i}", feature_vector=f"[1, {i}, 0]", other_split_vector=f"[1, {i + 3}, 5]",
                        is_leakage="0") for i in range(20)]
            rows += [row(case_id=f"dup{i}", feature_vector=f"[1, {i}, 2]", other_split_vector=f"[1, {i}, 2.0001]",
                         is_leakage="1") for i in range(5)]
            rows += [row(case_id="temp", feature_time="2026-01-02T00:00:00Z", is_leakage="1")]
            dev = tmp / "dev_cases.csv"
            self._write(dev, rows)
            code = tune_main(["--dev", str(dev), "--config-out", str(tmp / "cfg.json"),
                              "--report-out", str(tmp / "report.md")])
            self.assertEqual(code, 0)
            cfg = json.loads((tmp / "cfg.json").read_text(encoding="utf-8"))
            self.assertLessEqual(cfg["tuning"]["dev_metrics"]["fpr"], 0.05)
            self.assertEqual(cfg["tuning"]["dev_metrics"]["recall"], 1.0)
            self.assertFalse(cfg["tuning"]["heldout_read"])
            self.assertEqual(cfg["tuning"]["dev_missing_columns"], [])
            v = ImprovedValidator(tmp / "cfg.json")
            self.assertEqual(v.config.near_duplicate_threshold, cfg["near_duplicate_threshold"])
            self.assertIn("near_duplicate_threshold sweep", (tmp / "report.md").read_text(encoding="utf-8"))

    def test_schema_preflight_reports_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            renamed = {("feature_timestamp" if k == "feature_time" else k): v for k, v in CLEAN.items()}
            self._write(tmp / "dev_cases.csv", [dict(renamed, is_leakage="0"), dict(renamed, is_leakage="1")])
            tune_main(["--dev", str(tmp / "dev_cases.csv"), "--config-out", str(tmp / "cfg.json"),
                       "--report-out", str(tmp / "report.md")])
            cfg = json.loads((tmp / "cfg.json").read_text(encoding="utf-8"))
            self.assertEqual(cfg["tuning"]["dev_missing_columns"], ["feature_time"])
            self.assertIn("Schema warning", (tmp / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
