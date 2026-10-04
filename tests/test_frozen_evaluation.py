import json
from pathlib import Path

import pandas as pd
import pytest

from src import frozen_evaluation as evaluation
from src.baseline import baseline_validate
from src.generate_data import generate_cases, write_outputs
from src.validators import ImprovedValidator, ValidatorConfig, DEFAULT_CONFIG_PATH
from tests.test_validators import CLEAN, row


def fixture_data(directory):
    directory.mkdir()
    record = {"case_id": "D", "ground_truth": "clean", "is_leakage": "false", **CLEAN}
    record["case_id"] = "D"
    pd.DataFrame([record]).to_csv(directory / "dev_cases.csv", index=False)
    record["case_id"] = "H"
    pd.DataFrame([record]).to_csv(directory / "heldout_cases.csv", index=False)
    return directory


def test_baseline_misses_future_aggregate_improved_catches_it():
    example = row(aggregation_window_end="2026-01-01T12:00:00Z")
    assert not baseline_validate(example)["flagged"]
    assert ImprovedValidator(ValidatorConfig()).validate(example)["flagged"]


def test_current_shipping_label_policy_does_not_flag():
    example = row(feature_expression="shipping_label_length", lineage_columns='["shipping_label"]')
    assert not ImprovedValidator(DEFAULT_CONFIG_PATH).validate(example)["flagged"]


@pytest.mark.parametrize("value,expected", [("false", 0), ("true", 1), ("0", 0), ("1", 1), ("clean", 0), ("leakage", 1)])
def test_labels_are_parsed_explicitly(value, expected):
    assert evaluation.binary_label(value) == expected


@pytest.mark.parametrize("value", ["", "unknown", "2", None])
def test_invalid_labels_rejected(value):
    with pytest.raises(ValueError):
        evaluation.binary_label(value)


def test_confusion_counts_and_full_precision():
    result = evaluation.exact_metrics([1, 1, 1, 0, 0], [1, 0, 0, 1, 0])
    assert [result[k] for k in ("TP", "FP", "TN", "FN")] == [1, 1, 1, 2]
    assert result["recall"] == 1 / 3
    assert result["fpr"] == 1 / 2
    assert result["precision"] == 1 / 2
    assert result["f1"] == 2 / 5
    assert not result["fpr_constraint_pass"]


def test_fpr_constraint_uses_actual_fraction():
    labels = [0] * 40 + [1]
    assert evaluation.exact_metrics(labels, [1]*2 + [0]*38 + [1])["fpr_constraint_pass"]
    assert not evaluation.exact_metrics(labels, [1]*3 + [0]*37 + [1])["fpr_constraint_pass"]


def test_freeze_reused_and_input_change_refused(tmp_path):
    data = fixture_data(tmp_path / "data")
    results = tmp_path / "results"
    evaluation.freeze_validator(data, results, strict=False)
    freeze_path = results / "frozen_validator_config.json"
    original = freeze_path.read_bytes()
    evaluation.freeze_validator(data, results, strict=False)
    assert freeze_path.read_bytes() == original
    with (data / "heldout_cases.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Frozen code/config/data changed"):
        evaluation.freeze_validator(data, results, strict=False)
    assert freeze_path.read_bytes() == original


def test_effective_configuration_change_refused(tmp_path, monkeypatch):
    data = fixture_data(tmp_path / "data")
    results = tmp_path / "results"
    evaluation.freeze_validator(data, results, strict=False)
    original = evaluation.ValidatorConfig.from_dict
    def changed(config):
        updated = original(config)
        updated.near_duplicate_threshold = 0.5
        return updated
    monkeypatch.setattr(evaluation.ValidatorConfig, "from_dict", changed)
    with pytest.raises(ValueError, match="Frozen code/config/data changed"):
        evaluation.freeze_validator(data, results, strict=False)


def test_snapshot_written_before_heldout_parsing_and_prediction(tmp_path, monkeypatch):
    data = fixture_data(tmp_path / "data")
    results = tmp_path / "results"
    original_load = evaluation.load_cases
    seen = []
    def audited_load(path, split, strict):
        snapshot = json.loads((results / "frozen_validator_config.json").read_text())
        assert not snapshot["heldout_content_parsed_by_this_evaluator_before_freeze"]
        seen.append(split)
        return original_load(path, split, strict)
    monkeypatch.setattr(evaluation, "load_cases", audited_load)
    monkeypatch.setattr(evaluation, "export_results", lambda frame, directory: {"overall": pd.DataFrame(), "predictions": frame})
    output = evaluation.run_frozen_evaluation(data, results, strict=False)
    assert seen == ["dev", "heldout"]
    assert len(output["predictions"]) == 2


def test_case_and_truth_fields_never_reach_validator():
    record = {**CLEAN, "ground_truth": 1, "leakage_type": "target_derived", "severity": "high", "split": "heldout"}
    class AuditedValidator:
        def validate(self, features):
            assert evaluation.TRUTH_FIELDS.isdisjoint(features)
            return {"flagged": False, "root_cause": "none", "reasons": []}
    result = evaluation.predict(pd.DataFrame([record]), AuditedValidator())
    assert not result.improved_flagged.iloc[0]


def test_corrupted_labels_and_missing_family_rejected(tmp_path):
    write_outputs(generate_cases(), tmp_path)
    path = tmp_path / "heldout_cases.csv"
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    data.loc[0, "is_leakage"] = "true"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="inconsistent ground-truth"):
        evaluation.load_cases(path, "heldout")
    write_outputs(generate_cases(), tmp_path)
    data = pd.read_csv(path, dtype=str, keep_default_na=False).iloc[:-1]
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="wrong family counts"):
        evaluation.load_cases(path, "heldout")


def test_seed_reproducible_bytes_and_heldout_composition(tmp_path):
    write_outputs(generate_cases(), tmp_path / "first")
    write_outputs(generate_cases(), tmp_path / "second")
    for filename in ("dev_cases.csv", "heldout_cases.csv", "all_cases.csv"):
        assert (tmp_path / "first" / filename).read_bytes() == (tmp_path / "second" / filename).read_bytes()
    heldout = evaluation.load_cases(tmp_path / "first/heldout_cases.csv", "heldout")
    assert len(heldout) == 120
    assert heldout.ground_truth.sum() == 80
