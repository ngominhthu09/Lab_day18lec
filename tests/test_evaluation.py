"""
tests/test_evaluation.py
Comprehensive test suite verifying:
1. Primary Baseline & Ingestion-aware Baseline logic.
2. Public evaluate() function with binary flags and continuous scores.
3. Threshold calibration and freezing mechanism.
4. End-to-end evaluation with comparison tables and chart generation.
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.baseline import baseline_validate, baseline_ingestion_validate
from src.evaluate import evaluate, run_evaluation, compute_confusion_and_metrics


def test_baseline_logic():
    # 1. Direct temporal leakage
    row1 = {"feature_time": "2026-03-01 12:00:00", "prediction_time": "2026-03-01 10:00:00"}
    res1 = baseline_validate(row1)
    assert res1["flagged"] is True
    assert res1["root_cause"] == "temporal_leakage"

    # 2. Ingestion-aware catches ingestion delay
    row_ingest = {
        "feature_time": "2026-03-01 09:00:00",
        "prediction_time": "2026-03-01 10:00:00",
        "ingestion_time": "2026-03-01 11:00:00"
    }
    # Naive baseline misses it
    assert baseline_validate(row_ingest)["flagged"] is False
    # Ingestion baseline catches it
    res_i = baseline_ingestion_validate(row_ingest)
    assert res_i["flagged"] is True
    assert res_i["root_cause"] == "ingestion_delay_leakage"


def test_evaluate_public_interface():
    # Test with continuous scores and calibration
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3])
    labels = np.array([0, 0, 1, 1, 0])
    l_types = np.array(["clean", "clean", "temporal_leakage", "future_aggregation_leakage", "clean"])

    res = evaluate(scores_or_flags=scores, labels=labels, leakage_types=l_types, max_fpr=0.05)
    assert "overall" in res
    assert "per_type" in res
    assert res["overall"]["fpr_constraint_pass"] is True
    assert res["overall"]["recall"] == 1.0


def test_end_to_end_mock_evaluation():
    mock_data_dir = os.path.join(PROJECT_ROOT, "mock_data")
    mock_results_dir = os.path.join(PROJECT_ROOT, "mock_results")
    os.makedirs(mock_data_dir, exist_ok=True)

    sample_cases = [
        {"case_id": "C01", "feature_time": "2026-01-01 09:00:00", "ingestion_time": "2026-01-01 09:10:00", "prediction_time": "2026-01-01 10:00:00", "ground_truth": 0, "leakage_type": "none", "severity": "none"},
        {"case_id": "C02", "feature_time": "2026-01-01 08:00:00", "ingestion_time": "2026-01-01 08:10:00", "prediction_time": "2026-01-01 10:00:00", "ground_truth": 0, "leakage_type": "none", "severity": "none"},
        {"case_id": "C03", "feature_time": "2026-01-01 11:00:00", "ingestion_time": "2026-01-01 11:10:00", "prediction_time": "2026-01-01 10:00:00", "ground_truth": 1, "leakage_type": "temporal_leakage", "severity": "high"},
        {"case_id": "C04", "feature_time": "2026-01-01 09:30:00", "ingestion_time": "2026-01-01 10:30:00", "prediction_time": "2026-01-01 10:00:00", "ground_truth": 1, "leakage_type": "ingestion_delay_leakage", "severity": "medium"},
    ]

    df_dev = pd.DataFrame(sample_cases)
    df_heldout = pd.DataFrame(sample_cases)
    df_dev.to_csv(os.path.join(mock_data_dir, "dev_cases.csv"), index=False)
    df_heldout.to_csv(os.path.join(mock_data_dir, "heldout_cases.csv"), index=False)

    run_evaluation(data_dir=mock_data_dir, results_dir=mock_results_dir)

    assert os.path.exists(os.path.join(mock_results_dir, "comparison_dev_vs_heldout.csv"))
    assert os.path.exists(os.path.join(mock_results_dir, "confusion_baseline.png"))
    assert os.path.exists(os.path.join(mock_results_dir, "confusion_baseline_ingestion.png"))

    shutil.rmtree(mock_data_dir, ignore_errors=True)
    shutil.rmtree(mock_results_dir, ignore_errors=True)


if __name__ == "__main__":
    test_baseline_logic()
    test_evaluate_public_interface()
    test_end_to_end_mock_evaluation()
    print("All enhanced evaluation and baseline tests passed!")
