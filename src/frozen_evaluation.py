"""Independent binary evaluation; never calibrate or tune on held-out data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from config.data_config import LEAKAGE_TYPES, REQUIRED_FIELDS
from src.baseline import baseline_ingestion_validate, baseline_validate
from src.validators import CHECK_NAMES, ROOT_CAUSE_PRIORITY, ImprovedValidator, ValidatorConfig

ROOT = Path(__file__).resolve().parents[1]
TRUTH_FIELDS = {"ground_truth", "is_leakage", "is_leaky", "label", "leakage_type", "severity", "split", "case_id", "entity_id"}
PROTECTED_FILES = (
    "src/validators.py", "src/baseline.py", "src/generate_data.py",
    "config/data_config.py", "config/validator_config.json", "config/validator_vocabulary.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    """Normalize CRLF only so a Git checkout on another OS preserves the freeze."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def binary_label(value) -> int:
    if value is None or pd.isna(value):
        raise ValueError(f"Invalid or missing ground truth: {value!r}")
    token = str(value).strip().lower()
    if token in {"1", "1.0", "true", "leakage"}:
        return 1
    if token in {"0", "0.0", "false", "clean", "none"}:
        return 0
    raise ValueError(f"Invalid or missing ground truth: {value!r}")


def load_cases(path: Path, split: str, strict: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    label_cols = [c for c in ("ground_truth", "is_leakage", "is_leaky") if c in frame]
    if not label_cols or "case_id" not in frame:
        raise ValueError(f"{path}: case_id and ground-truth columns are required")
    labels = frame[label_cols[0]].map(binary_label)
    for column in label_cols[1:]:
        if not frame[column].map(binary_label).equals(labels):
            raise ValueError(f"{path}: inconsistent ground-truth columns")
    if frame.empty or frame.case_id.eq("").any() or frame.case_id.duplicated().any():
        raise ValueError(f"{path}: empty dataset or missing/duplicate case_id")
    if strict:
        missing = set(REQUIRED_FIELDS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        expected = {"clean": 60 if split == "dev" else 40}
        expected.update({family: 24 if split == "dev" else 16 for family in LEAKAGE_TYPES})
        if Counter(frame.leakage_type) != Counter(expected):
            raise ValueError(f"{path}: wrong family counts {dict(Counter(frame.leakage_type))}")
        if not frame.split.eq(split).all():
            raise ValueError(f"{path}: incorrect split")
        if not (frame.leakage_type.ne("clean").astype(int) == labels).all():
            raise ValueError(f"{path}: family and labels disagree")
        if not frame.loc[labels == 0, "severity"].eq("none").all():
            raise ValueError(f"{path}: clean severity must be none")
        for family in LEAKAGE_TYPES:
            if set(frame.loc[frame.leakage_type == family, "severity"]) != {"low", "medium", "high"}:
                raise ValueError(f"{path}: missing severity in {family}")
    frame["ground_truth"] = labels
    frame["split"] = split
    return frame


def freeze_validator(data_dir: Path, results_dir: Path, strict: bool = True) -> dict:
    """Persist before held-out CSV parsing/prediction; existing snapshots are immutable."""
    snapshot_path = results_dir / "frozen_validator_config.json"
    current_hashes = {name: sha256(ROOT / name) for name in PROTECTED_FILES}
    data_hashes = {split: sha256(data_dir / f"{split}_cases.csv") for split in ("dev", "heldout")}
    normalized_files = {name: normalized_sha256(ROOT / name) for name in PROTECTED_FILES}
    normalized_data = {split: normalized_sha256(data_dir / f"{split}_cases.csv") for split in ("dev", "heldout")}
    raw_config = json.loads((ROOT / "config/validator_config.json").read_text(encoding="utf-8"))
    dev_lf = (data_dir / "dev_cases.csv").read_bytes().replace(b"\r\n", b"\n")
    dev_hash_variants = {data_hashes["dev"], normalized_data["dev"], hashlib.sha256(dev_lf.replace(b"\n", b"\r\n")).hexdigest()}
    if strict and raw_config.get("tuning", {}).get("dev_sha256") not in dev_hash_variants:
        raise ValueError("Agent #3 configuration was tuned on a different Dev file")
    effective = ValidatorConfig.from_dict(raw_config).to_dict()
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if (snapshot["file_normalized_sha256"] != normalized_files or snapshot["data_normalized_sha256"] != normalized_data
                or snapshot["effective_config"] != effective):
            raise ValueError("Frozen code/config/data changed. Refusing to overwrite the freeze or evaluate.")
        return snapshot
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, []
    snapshot = {
        "schema_version": 1, "frozen_at_utc": utc_now(), "git_commit": commit,
        "git_worktree_status_at_freeze": dirty, "file_sha256": current_hashes,
        "data_sha256": data_hashes, "effective_config": effective,
        "file_normalized_sha256": normalized_files, "data_normalized_sha256": normalized_data,
        "hash_policy": "Raw SHA-256 records original bytes; verification permits CRLF/LF conversion only using additional normalized hashes.",
        "root_cause_priority": list(ROOT_CAUSE_PRIORITY), "check_order": list(CHECK_NAMES),
        "agent3_tuning_provenance": raw_config.get("tuning", {}),
        "prediction_policy": "Use original binary flagged decisions; no additional score threshold or calibration.",
        "heldout_content_parsed_by_this_evaluator_before_freeze": False,
        "prior_heldout_artifacts_present": (results_dir / "metrics_overall.csv").exists(),
        "audit_limit": "This snapshot proves freeze ordering for this independent rerun only. Earlier held-out metrics already existed in the supplied repo; historical blindness cannot be certified.",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of the first snapshot.
    with snapshot_path.open("x", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return snapshot


def predict(frame: pd.DataFrame, validator: ImprovedValidator) -> pd.DataFrame:
    records = []
    functions = {"baseline": baseline_validate, "baseline_ingestion": baseline_ingestion_validate,
                 "improved": validator.validate}
    for row in frame.to_dict("records"):
        features = {k: v for k, v in row.items() if k not in TRUTH_FIELDS}
        output = dict(row)
        for name, function in functions.items():
            result = function(features)
            if result.get("debug_metadata", {}).get("checks_failed"):
                raise RuntimeError(f"{row['case_id']}: validator gate failed")
            output[f"{name}_flagged"] = bool(result["flagged"])
            output[f"{name}_root_cause"] = result["root_cause"]
            output[f"{name}_reasons"] = json.dumps(result["reasons"], ensure_ascii=False)
            if name == "improved":
                for key in ("detected_causes", "scores", "warnings"):
                    output[f"improved_{key}"] = json.dumps(result.get(key, [] if key != "scores" else {}), ensure_ascii=False)
        records.append(output)
    return pd.DataFrame(records)


def exact_metrics(truth, predictions) -> dict:
    """Full precision; independently counted, not the rounded legacy evaluate() metrics."""
    truth, predictions = np.asarray(truth, dtype=int), np.asarray(predictions, dtype=int)
    if truth.shape != predictions.shape or truth.ndim != 1:
        raise ValueError("Truth/prediction lengths must agree")
    if not np.isin(truth, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Only binary labels are supported")
    tp = int(((truth == 1) & (predictions == 1)).sum())
    fp = int(((truth == 0) & (predictions == 1)).sum())
    tn = int(((truth == 0) & (predictions == 0)).sum())
    fn = int(((truth == 1) & (predictions == 0)).sum())
    def divide(a, b):
        return a / b if b else 0.0
    recall, fpr = divide(tp, tp + fn), divide(fp, fp + tn)
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, recall=recall, fpr=fpr,
                precision=divide(tp, tp + fp), f1=divide(2 * tp, 2 * tp + fp + fn),
                fpr_constraint_pass=fpr <= 0.05)


def export_results(predictions: pd.DataFrame, results_dir: Path) -> dict:
    from src.evaluate import plot_confusion_matrix, plot_grouped_bar_chart
    names = ("baseline", "baseline_ingestion", "improved")
    splits = list(predictions.split.unique())
    overall_rows, type_rows, severity_rows = [], [], []
    for split in splits + (["all"] if len(splits) > 1 else []):
        subset = predictions if split == "all" else predictions[predictions.split == split]
        for name in names:
            overall_rows.append(dict(split=split, validator=name,
                                     **exact_metrics(subset.ground_truth, subset[f"{name}_flagged"])))
            leaky = subset[subset.ground_truth == 1]
            for column, destination in (("leakage_type", type_rows), ("severity", severity_rows)):
                if column not in leaky:
                    continue
                for group, rows in leaky.groupby(column, sort=True):
                    tp = int(rows[f"{name}_flagged"].sum())
                    destination.append(dict(split=split, validator=name, **{column: group},
                                            n=len(rows), tp=tp, fn=len(rows)-tp, recall=tp/len(rows)))
    overall = pd.DataFrame(overall_rows)
    by_type, by_severity = pd.DataFrame(type_rows), pd.DataFrame(severity_rows)
    for filename, frame in (("metrics_overall.csv", overall), ("metrics_by_type.csv", by_type),
                            ("metrics_by_severity.csv", by_severity), ("predictions.csv", predictions)):
        frame.to_csv(results_dir / filename, index=False, lineterminator="\n")
    gap_rows = []
    if {"dev", "heldout"} <= set(splits):
        for name in names:
            dev = overall[(overall.split == "dev") & (overall.validator == name)].iloc[0]
            held = overall[(overall.split == "heldout") & (overall.validator == name)].iloc[0]
            gap_rows.append(dict(validator=name, dev_recall=dev.recall, heldout_recall=held.recall,
                                 dev_fpr=dev.fpr, heldout_fpr=held.fpr,
                                 recall_gap=dev.recall-held.recall, fpr_gap=held.fpr-dev.fpr))
        pd.DataFrame(gap_rows).to_csv(results_dir / "generalization_gap.csv", index=False)
    overall[overall.split.isin(["dev", "heldout"])].to_csv(results_dir / "comparison_dev_vs_heldout.csv", index=False)
    if "heldout" in splits:
        held = predictions[predictions.split == "heldout"]
        held.to_csv(results_dir / "heldout_predictions.csv", index=False, lineterminator="\n")
        overall[overall.split == "heldout"].to_csv(results_dir / "heldout_metrics.csv", index=False)
        failures = []
        for name in names:
            wrong = held[held[f"{name}_flagged"].astype(int) != held.ground_truth]
            for row in wrong.to_dict("records"):
                failures.append(dict(validator=name, error="FN" if row["ground_truth"] else "FP", **row))
        # Keep a header even when no failures exist.
        pd.DataFrame(failures, columns=["validator", "error", *held.columns]).to_csv(results_dir / "heldout_failures.csv", index=False)
    chart_split = "heldout" if "heldout" in splits else "dev"
    for name in names:
        metric = overall[(overall.split == chart_split) & (overall.validator == name)].iloc[0]
        plot_confusion_matrix(int(metric.TP), int(metric.FP), int(metric.TN), int(metric.FN), name,
                              str(results_dir / f"confusion_{name}.png"), chart_split)
    for frame, column, filename, title in (
            (by_type, "leakage_type", "recall_by_type.png", "Recall by Leakage Type"),
            (by_severity, "severity", "recall_by_severity.png", "Recall by Severity")):
        if not frame.empty:
            plot_grouped_bar_chart(frame, column, "recall", title, str(results_dir / filename), chart_split)
    return dict(overall=overall, by_type=by_type, by_severity=by_severity, predictions=predictions)


def run_frozen_evaluation(data_dir=ROOT / "data", results_dir=ROOT / "results",
                          eval_split="all", strict=True) -> dict:
    data_dir, results_dir = Path(data_dir).resolve(), Path(results_dir).resolve()
    if eval_split not in {"all", "dev", "heldout"}:
        raise ValueError("split must be all, dev or heldout")
    started = utc_now()
    snapshot = freeze_validator(data_dir, results_dir, strict)
    # Reload the persisted snapshot, rather than using mutable active configuration.
    frozen = json.loads((results_dir / "frozen_validator_config.json").read_text(encoding="utf-8"))
    validator = ImprovedValidator(frozen["effective_config"])
    frames = []
    dev_ids = set()
    for split in ("dev", "heldout"):
        if eval_split != "all" and split != eval_split:
            continue
        cases = load_cases(data_dir / f"{split}_cases.csv", split, strict)
        if split == "dev":
            dev_ids = set(cases.case_id)
        elif dev_ids.intersection(cases.case_id):
            raise ValueError("Dev/Held-out case IDs overlap")
        print(f"[EVALUATE] {split}: {len(cases)} cases, frozen threshold {validator.config.near_duplicate_threshold}")
        frames.append(predict(cases, validator))
    # Abort if protected inputs changed during evaluation.
    freeze_validator(data_dir, results_dir, strict)
    outputs = export_results(pd.concat(frames, ignore_index=True), results_dir)
    write_json(results_dir / "evaluation_manifest.json", {
        "started_at_utc": started, "completed_at_utc": utc_now(), "frozen_at_utc": snapshot["frozen_at_utc"],
        "frozen_snapshot_sha256": sha256(results_dir / "frozen_validator_config.json"),
        "heldout_tuning": False, "python": platform.python_version(),
        "packages": {"pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "evaluator_sha256": sha256(Path(__file__)),
        "artifact_sha256": {p.name: sha256(p) for p in sorted(results_dir.iterdir()) if p.suffix in {".csv", ".png"}},
    })
    print(outputs["overall"].to_string(index=False))
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--split", choices=["all", "dev", "heldout"], default="all")
    args = parser.parse_args()
    run_frozen_evaluation(args.data_dir, args.results_dir, args.split)
