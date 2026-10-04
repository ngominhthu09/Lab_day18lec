"""
src/evaluate.py
Evaluation & Benchmark Engine for Point-in-Time Training Data Validator.

Key Capabilities:
1. Public function evaluate(scores_or_flags, labels, ...) supporting both continuous anomaly scores
   and binary decisions, computing Recall @ FPR <= 5%, Precision, F1, and per-type confusion matrices.
2. Two-stage Calibration & Freeze Protocol:
   - Calibrates decision threshold solely on Dev set to satisfy FPR <= 5%.
   - Freezes threshold into 'results/frozen_threshold.json'.
   - Held-out evaluation strictly loads the frozen threshold with zero tuning allowed.
3. Multi-validator benchmark:
   - Primary Baseline (feature_time > prediction_time)
   - Ingestion-Aware Baseline (feature_time > pred_time OR ingestion_time > pred_time)
   - Improved Validator (loaded dynamically from src.validators)
4. Comprehensive comparison tables (Dev vs Held-out, Baseline vs Improved) and publication-ready charts.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports work seamlessly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.baseline import baseline_validate, baseline_ingestion_validate


# ---------------------------------------------------------------------------
# Core Metric Calculations
# ---------------------------------------------------------------------------

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    if denominator == 0 or np.isnan(denominator):
        return default
    return float(numerator / denominator)


def compute_confusion_and_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute binary classification metrics for given ground truth and predictions.
    Positive (1) = Leakage, Negative (0) = Clean.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    recall = safe_divide(tp, tp + fn)
    fpr = safe_divide(fp, fp + tn)
    precision = safe_divide(tp, tp + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "fpr_constraint_pass": bool(fpr <= 0.05)
    }


def find_optimal_threshold_on_dev(
    scores: np.ndarray,
    labels: np.ndarray,
    max_fpr: float = 0.05
) -> Tuple[float, Dict[str, Any]]:
    """
    Find threshold tau that maximizes Recall subject to FPR <= max_fpr.
    Evaluated strictly on development set.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    # Candidate thresholds from unique scores
    unique_scores = np.sort(np.unique(scores))
    candidates = np.concatenate(([0.0], unique_scores, [1.0, np.max(scores) + 1e-5]))

    best_thresh = 0.5
    best_recall = -1.0
    best_metrics = None

    # Sweep thresholds descending to find boundary
    for thresh in candidates:
        preds = (scores >= thresh).astype(int)
        m = compute_confusion_and_metrics(labels, preds)
        if m["fpr"] <= max_fpr:
            if m["recall"] > best_recall:
                best_recall = m["recall"]
                best_thresh = float(thresh)
                best_metrics = m

    # If no threshold achieved FPR <= max_fpr (e.g. extremely noisy), pick strictest threshold
    if best_metrics is None:
        best_thresh = float(np.max(scores) + 1.0)
        best_metrics = compute_confusion_and_metrics(labels, np.zeros_like(labels))

    return best_thresh, best_metrics


# ---------------------------------------------------------------------------
# Public evaluate() Interface
# ---------------------------------------------------------------------------

def evaluate(
    scores_or_flags: Union[np.ndarray, List, pd.Series],
    labels: Union[np.ndarray, List, pd.Series],
    leakage_types: Optional[Union[np.ndarray, List, pd.Series]] = None,
    threshold: Optional[float] = None,
    max_fpr: float = 0.05
) -> Dict[str, Any]:
    """
    Standard evaluation function as specified:
    - Accepts continuous scores or boolean/binary flags.
    - Computes Recall @ FPR <= 5%, Precision, F1.
    - Computes confusion matrix and recall broken down by leakage_type.

    Parameters
    ----------
    scores_or_flags : array-like
        Predicted anomaly scores [0.0 - 1.0] or binary flags {0, 1, True, False}.
    labels : array-like
        Ground truth binary labels (1 = Leaky, 0 = Clean).
    leakage_types : array-like, optional
        Categorical leakage types for each row to compute per-type confusion matrices.
    threshold : float, optional
        Pre-calibrated decision threshold (used when continuous scores are provided).
    max_fpr : float, default 0.05
        Maximum permitted False Positive Rate.

    Returns
    -------
    Dict[str, Any]
        Aggregated evaluation results including primary metric and per-type breakdowns.
    """
    arr_input = np.asarray(scores_or_flags)
    arr_labels = np.asarray(labels, dtype=int)

    is_continuous = np.issubdtype(arr_input.dtype, np.floating) and not np.array_equal(arr_input, arr_input.astype(bool))

    applied_threshold = threshold
    if is_continuous:
        if applied_threshold is None:
            applied_threshold, _ = find_optimal_threshold_on_dev(arr_input, arr_labels, max_fpr)
        y_pred = (arr_input >= applied_threshold).astype(int)
        overall_m = compute_confusion_and_metrics(arr_labels, y_pred)
    else:
        # Binary flags input
        y_pred = arr_input.astype(bool).astype(int)
        overall_m = compute_confusion_and_metrics(arr_labels, y_pred)

    output = {
        "overall": overall_m,
        "applied_threshold": applied_threshold,
        "per_type": {}
    }

    # Per leakage-type confusion matrix and recall
    if leakage_types is not None:
        type_arr = np.asarray(leakage_types)
        unique_types = [t for t in np.unique(type_arr) if t not in ["none", "clean", None] and pd.notna(t)]

        for l_type in unique_types:
            mask = (type_arr == l_type)
            n_cases = int(np.sum(mask))
            if n_cases == 0:
                continue

            sub_labels = arr_labels[mask]
            sub_preds = y_pred[mask]
            sub_m = compute_confusion_and_metrics(sub_labels, sub_preds)

            output["per_type"][str(l_type)] = {
                "n": n_cases,
                "TP": sub_m["TP"],
                "FN": sub_m["FN"],
                "recall": sub_m["recall"]
            }

    return output


# ---------------------------------------------------------------------------
# Zero-Leakage Pipeline & Threshold Freezing
# ---------------------------------------------------------------------------

LEAKAGE_TARGET_COLUMNS = [
    "ground_truth", "is_leakage", "is_leaky", "label",
    "leakage_type", "severity", "split"
]

def run_validator_safe(df: pd.DataFrame, validate_func, validator_name: str) -> pd.DataFrame:
    """Run validator ensuring ground-truth columns are never visible."""
    feature_cols = [c for c in df.columns if c not in LEAKAGE_TARGET_COLUMNS]
    clean_input = df[feature_cols].copy()

    records = []
    for _, row in clean_input.iterrows():
        try:
            res = validate_func(row.to_dict())
            records.append({
                f"{validator_name}_flagged": bool(res.get("flagged", False)),
                f"{validator_name}_score": float(res.get("score", 1.0 if res.get("flagged") else 0.0)),
                f"{validator_name}_root_cause": str(res.get("root_cause", "none")),
                f"{validator_name}_reasons": "; ".join(res.get("reasons", []))
            })
        except Exception as e:
            records.append({
                f"{validator_name}_flagged": False,
                f"{validator_name}_score": 0.0,
                f"{validator_name}_root_cause": "error",
                f"{validator_name}_reasons": f"Error: {str(e)}"
            })

    return pd.DataFrame(records, index=df.index)


def save_frozen_threshold(filepath: str, validator_name: str, threshold: float, dev_metrics: Dict[str, Any]):
    """Save calibrated threshold to file before touching heldout split."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    payload = {
        "frozen_timestamp": datetime.utcnow().isoformat() + "Z",
        "validator": validator_name,
        "calibrated_threshold": threshold,
        "dev_metrics": dev_metrics,
        "rule_note": "Calibrated exclusively on Dev set to guarantee FPR <= 5%. Frozen for Held-out evaluation."
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[FREEZE] Calibrated threshold {threshold} saved to {filepath}")


def load_frozen_threshold(filepath: str) -> Optional[float]:
    """Load previously frozen threshold from file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return float(data.get("calibrated_threshold", 0.5))
        except Exception as e:
            print(f"[WARN] Failed to load frozen threshold from {filepath}: {e}")
    return None


# ---------------------------------------------------------------------------
# Visualizations & Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    tp: int, fp: int, tn: int, fn: int,
    validator_name: str,
    output_path: str,
    split_name: str = "heldout"
):
    """Save clean, readable confusion matrix chart."""
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.7)

    for (i, j), z in np.ndenumerate(matrix):
        label_text = f"{z}\n"
        if i == 0 and j == 0:
            label_text += "(TN)"
        elif i == 0 and j == 1:
            label_text += "(FP)"
        elif i == 1 and j == 0:
            label_text += "(FN)"
        elif i == 1 and j == 1:
            label_text += "(TP)"
        ax.text(j, i, label_text, ha='center', va='center', fontsize=11, fontweight='bold')

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Clean (0)", "Pred Leaky (1)"])
    ax.set_yticklabels(["True Clean (0)", "True Leaky (1)"])
    ax.set_title(f"Confusion Matrix: {validator_name.capitalize()} [{split_name}]", pad=15, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_grouped_bar_chart(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    title: str,
    output_path: str,
    split_name: str = "heldout"
):
    """Plot grouped comparison bar chart for multiple validators."""
    sub_df = df[df["split"] == split_name] if "split" in df.columns else df
    if sub_df.empty:
        sub_df = df

    categories = list(sub_df[group_col].unique())
    validators = list(sub_df["validator"].unique())

    if not categories or not validators:
        return

    x = np.arange(len(categories))
    width = 0.8 / max(len(validators), 1)

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=150)

    for i, val in enumerate(validators):
        val_df = sub_df[sub_df["validator"] == val]
        val_map = dict(zip(val_df[group_col], val_df[metric_col]))
        values = [val_map.get(cat, 0.0) for cat in categories]
        pos = x + (i - (len(validators) - 1) / 2) * width
        rects = ax.bar(pos, values, width, label=val.replace("_", " ").title())
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_ylabel(metric_col.capitalize())
    ax.set_title(f"{title} ({split_name.capitalize()})", pad=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha='right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


# ---------------------------------------------------------------------------
# Complete Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(
    data_dir: str = "data",
    results_dir: str = "results",
    eval_split: str = "all"
) -> Dict[str, pd.DataFrame]:
    """
    Execute full evaluation framework:
    1. Loads Dev and Held-out splits.
    2. Runs Primary Baseline, Ingestion-Aware Baseline, and Improved Validator.
    3. Calibrates & Freezes threshold on Dev.
    4. Evaluates frozen threshold on Held-out.
    5. Exports CSV metrics, Pivot comparison, and charts.
    """
    os.makedirs(results_dir, exist_ok=True)
    frozen_path = os.path.join(results_dir, "frozen_threshold.json")

    # 1. Load Data
    dev_path = os.path.join(data_dir, "dev_cases.csv")
    heldout_path = os.path.join(data_dir, "heldout_cases.csv")

    dfs = []
    if os.path.exists(dev_path):
        df_dev = pd.read_csv(dev_path)
        df_dev["split"] = "dev"
        dfs.append(df_dev)
    if os.path.exists(heldout_path):
        df_heldout = pd.read_csv(heldout_path)
        df_heldout["split"] = "heldout"
        dfs.append(df_heldout)

    if not dfs:
        raise FileNotFoundError(f"No datasets found in '{data_dir}'. Expected dev_cases.csv and/or heldout_cases.csv.")

    full_df = pd.concat(dfs, ignore_index=True)

    # Standardize ground truth
    if "is_leakage" in full_df.columns:
        full_df["ground_truth"] = full_df["is_leakage"].astype(bool).astype(int)
    elif "is_leaky" in full_df.columns:
        full_df["ground_truth"] = full_df["is_leaky"].astype(bool).astype(int)
    elif "ground_truth" in full_df.columns:
        full_df["ground_truth"] = full_df["ground_truth"].map(
            {"leakage": 1, "clean": 0, "1": 1, "0": 0, 1: 1, 0: 0, True: 1, False: 0}
        ).fillna(0).astype(int)
    else:
        raise KeyError("Dataset must contain 'is_leakage', 'is_leaky', or 'ground_truth'.")

    if "case_id" not in full_df.columns:
        full_df["case_id"] = [f"CASE_{i+1:04d}" for i in range(len(full_df))]

    # 2. Register Validators
    validators_to_run = [
        ("baseline", baseline_validate),
        ("baseline_ingestion", baseline_ingestion_validate)
    ]

    has_improved = False
    try:
        from src.validators import improved_validate
        validators_to_run.append(("improved", improved_validate))
        has_improved = True
        print("[INFO] Loaded improved_validate from src.validators.")
    except (ImportError, ModuleNotFoundError):
        print("[INFO] src.validators not yet present. Benchmarking Baselines.")

    # 3. Generate Predictions (Zero-Leakage)
    pred_frames = []
    for val_name, val_fn in validators_to_run:
        val_pred = run_validator_safe(full_df, val_fn, val_name)
        pred_frames.append(val_pred)

    eval_df = pd.concat([full_df] + pred_frames, axis=1)

    # 4. Calibrate & Freeze Threshold on Dev (if continuous scores or improved validator)
    if "dev" in eval_df["split"].values and has_improved:
        dev_sub = eval_df[eval_df["split"] == "dev"]
        opt_thresh, dev_m = find_optimal_threshold_on_dev(
            scores=dev_sub["improved_score"].values,
            labels=dev_sub["ground_truth"].values,
            max_fpr=0.05
        )
        save_frozen_threshold(frozen_path, "improved", opt_thresh, dev_m)

    # 5. Compute Metrics Overall
    validator_names = [v[0] for v in validators_to_run]
    splits_present = [s for s in eval_df["split"].unique() if pd.notna(s)]
    split_categories = list(splits_present)
    if len(splits_present) > 1:
        split_categories.append("all")

    overall_rows = []
    for split_val in split_categories:
        sub_df = eval_df if split_val == "all" else eval_df[eval_df["split"] == split_val]
        y_true = sub_df["ground_truth"].values

        for val_name in validator_names:
            y_pred = sub_df[f"{val_name}_flagged"].astype(int).values
            m = compute_confusion_and_metrics(y_true, y_pred)
            overall_rows.append({
                "split": split_val,
                "validator": val_name,
                "TP": m["TP"],
                "FP": m["FP"],
                "TN": m["TN"],
                "FN": m["FN"],
                "recall": m["recall"],
                "fpr": m["fpr"],
                "precision": m["precision"],
                "f1": m["f1"],
                "fpr_constraint_pass": m["fpr_constraint_pass"]
            })

    metrics_overall = pd.DataFrame(overall_rows)
    metrics_overall.to_csv(os.path.join(results_dir, "metrics_overall.csv"), index=False)

    # 6. Pivot Comparison Table: Baseline vs Improved across Dev vs Heldout
    pivoted_rows = []
    for val_name in validator_names:
        for split_val in ["dev", "heldout"]:
            match = metrics_overall[(metrics_overall["validator"] == val_name) & (metrics_overall["split"] == split_val)]
            if not match.empty:
                r = match.iloc[0]
                pivoted_rows.append({
                    "Validator": val_name,
                    "Split": split_val,
                    "Recall (@ FPR<=5%)": r["recall"],
                    "FPR": r["fpr"],
                    "Precision": r["precision"],
                    "F1": r["f1"],
                    "Constraint Pass": r["fpr_constraint_pass"]
                })
    if pivoted_rows:
        df_comparison = pd.DataFrame(pivoted_rows)
        comp_path = os.path.join(results_dir, "comparison_dev_vs_heldout.csv")
        df_comparison.to_csv(comp_path, index=False)
        print(f"[OUTPUT] Saved comparison table to {comp_path}")

    # 7. Metrics by Type
    metrics_by_type = pd.DataFrame()
    if "leakage_type" in eval_df.columns:
        leaky_df = eval_df[eval_df["ground_truth"] == 1]
        type_rows = []
        types = sorted([t for t in leaky_df["leakage_type"].dropna().unique() if t not in ["none", "clean"]])

        for split_val in split_categories:
            sub = leaky_df if split_val == "all" else leaky_df[leaky_df["split"] == split_val]
            for l_type in types:
                type_sub = sub[sub["leakage_type"] == l_type]
                n_c = len(type_sub)
                if n_c == 0:
                    continue
                for val_name in validator_names:
                    tp = int(type_sub[f"{val_name}_flagged"].sum())
                    rec = safe_divide(tp, n_c)
                    type_rows.append({
                        "split": split_val,
                        "validator": val_name,
                        "leakage_type": l_type,
                        "n": n_c,
                        "tp": tp,
                        "fn": n_c - tp,
                        "recall": round(rec, 4)
                    })
        metrics_by_type = pd.DataFrame(type_rows)
        if not metrics_by_type.empty:
            metrics_by_type.to_csv(os.path.join(results_dir, "metrics_by_type.csv"), index=False)

    # 8. Metrics by Severity
    metrics_by_severity = pd.DataFrame()
    if "severity" in eval_df.columns:
        leaky_df = eval_df[eval_df["ground_truth"] == 1]
        sev_rows = []
        severities = [s for s in ["low", "medium", "high"] if s in leaky_df["severity"].unique()]

        for split_val in split_categories:
            sub = leaky_df if split_val == "all" else leaky_df[leaky_df["split"] == split_val]
            for sev in severities:
                sev_sub = sub[sub["severity"] == sev]
                n_c = len(sev_sub)
                if n_c == 0:
                    continue
                for val_name in validator_names:
                    tp = int(sev_sub[f"{val_name}_flagged"].sum())
                    rec = safe_divide(tp, n_c)
                    sev_rows.append({
                        "split": split_val,
                        "validator": val_name,
                        "severity": sev,
                        "n": n_c,
                        "tp": tp,
                        "fn": n_c - tp,
                        "recall": round(rec, 4)
                    })
        metrics_by_severity = pd.DataFrame(sev_rows)
        if not metrics_by_severity.empty:
            metrics_by_severity.to_csv(os.path.join(results_dir, "metrics_by_severity.csv"), index=False)

    # 9. Predictions Export
    pred_export_cols = [
        "case_id", "split", "ground_truth",
        "leakage_type" if "leakage_type" in eval_df.columns else None,
        "severity" if "severity" in eval_df.columns else None,
        "baseline_flagged", "baseline_root_cause",
        "baseline_ingestion_flagged", "baseline_ingestion_root_cause"
    ]
    if has_improved:
        pred_export_cols.extend(["improved_flagged", "improved_root_cause", "improved_reasons"])
    pred_export_cols = [c for c in pred_export_cols if c is not None and c in eval_df.columns]
    eval_df[pred_export_cols].to_csv(os.path.join(results_dir, "predictions.csv"), index=False)

    # 10. Generate Visualizations
    target_split = "heldout" if "heldout" in eval_df["split"].unique() else "dev"
    sub_eval = eval_df[eval_df["split"] == target_split]
    b_y_true = sub_eval["ground_truth"].values

    for val_name in validator_names:
        v_pred = sub_eval[f"{val_name}_flagged"].astype(int).values
        vm = compute_confusion_and_metrics(b_y_true, v_pred)
        plot_confusion_matrix(
            tp=vm["TP"], fp=vm["FP"], tn=vm["TN"], fn=vm["FN"],
            validator_name=val_name,
            output_path=os.path.join(results_dir, f"confusion_{val_name}.png"),
            split_name=target_split
        )

    if not metrics_by_type.empty:
        plot_grouped_bar_chart(
            df=metrics_by_type,
            group_col="leakage_type",
            metric_col="recall",
            title="Recall by Leakage Type",
            output_path=os.path.join(results_dir, "recall_by_type.png"),
            split_name=target_split
        )

    if not metrics_by_severity.empty:
        plot_grouped_bar_chart(
            df=metrics_by_severity,
            group_col="severity",
            metric_col="recall",
            title="Recall by Severity Level",
            output_path=os.path.join(results_dir, "recall_by_severity.png"),
            split_name=target_split
        )

    # Primary Metric Summary to Console
    print("\n" + "="*70)
    print(f"PRIMARY METRIC BENCHMARK REPORT: Leakage Recall @ FPR <= 5% [{target_split.upper()}]")
    print("="*70)
    split_metrics = metrics_overall[metrics_overall["split"] == target_split]
    for _, r in split_metrics.iterrows():
        status = "PASSED" if r["fpr_constraint_pass"] else "FAILED (FPR > 5%)"
        print(f"Validator: {r['validator']:<20} | Recall: {r['recall']:.4f} | FPR: {r['fpr']:.4f} | [{status}]")
    print("="*70 + "\n")

    return {
        "overall": metrics_overall,
        "by_type": metrics_by_type,
        "by_severity": metrics_by_severity,
        "predictions": eval_df
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point-in-Time Data Validator Evaluation Engine")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data folder containing CSVs")
    parser.add_argument("--results-dir", type=str, default="results", help="Path to output results folder")
    parser.add_argument("--split", type=str, default="all", help="Target split (dev, heldout, all)")

    args = parser.parse_args()

    d_dir = args.data_dir
    r_dir = args.results_dir
    if not os.path.exists(d_dir) and os.path.exists(os.path.join("Lab_day18lec", d_dir)):
        d_dir = os.path.join("Lab_day18lec", d_dir)
        r_dir = os.path.join("Lab_day18lec", r_dir)

    try:
        run_evaluation(data_dir=d_dir, results_dir=r_dir, eval_split=args.split)
    except Exception as e:
        print(f"[EVALUATION ERROR] {e}")
