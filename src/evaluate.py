"""Public evaluation utilities and compatibility CLI.

Continuous-score calibration utilities are for Dev only. The submission runner
uses the supplied frozen binary validator, never calibrates on Held-out, and
routes orchestration to src.frozen_evaluation.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp/matplotlib"))
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
    if group_col == "severity":
        categories = [s for s in ["low", "medium", "high"] if s in categories]
    elif group_col == "leakage_type":
        from config.data_config import LEAKAGE_TYPES
        categories = [s for s in LEAKAGE_TYPES if s in categories] + [s for s in categories if s not in LEAKAGE_TYPES]
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
    ax.set_xticklabels([str(c).replace("_", " ") for c in categories], rotation=15, ha='right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=len(validators))
    ax.set_title(f"{title} ({split_name.capitalize()})", pad=55, fontweight='bold')
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
    """Compatibility entry point: frozen binary decisions, never retune.

    Official submissions use scripts/run_all.py or src.frozen_evaluation,
    with strict verification of the 300-case dataset contract enabled.
    This interface also accepts smaller datasets used by unit fixtures.
    """
    from src.frozen_evaluation import run_frozen_evaluation
    return run_frozen_evaluation(data_dir, results_dir, eval_split, strict=False)


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
        raise SystemExit(1) from e
