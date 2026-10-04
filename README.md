# 1. Problem

Training features can contain information unavailable at prediction time. A timestamp-only check misses target-derived, aggregation and split-contamination leakage. Leakage includes more than time violations ([Kapoor & Narayanan, 2023](https://doi.org/10.1016/j.patter.2023.100804)).

# 2. Hypothesis

Combining point-in-time timestamps, aggregation boundaries, lineage, source roles and cross-split duplication improves leakage recall over the baseline while maintaining **FPR <= 5%**.

# 3. Setup

Python 3.11; `SEED = 20261004`. Synthetic **300 cases: 100 clean + 200 leakage** across five families. Stratified Dev/Held-out = **180/120 (60/40)**; Held-out contains 40 clean + 16 cases per family. Shared templates limit external generalization.

Run from this repository root (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_all.py
pytest -q
```

If activation is disabled, use `.\.venv\Scripts\python.exe scripts/run_all.py` and `.\.venv\Scripts\python.exe -m pytest -q` after installing with that interpreter.
The runner regenerates data in staging, verifies existing CSV content (CRLF/LF conversion allowed), checks schema/labels/counts, freezes Agent #3's supplied configuration before Held-out prediction, runs both validators and exports metrics/predictions/charts/docs. It never tunes. Existing snapshots are verified and preserved; changed protected inputs stop the run. See [submission guide](docs/submission.md) for the file tree and audit details.

# 4. Baseline

Primary baseline: flag only when `feature_time > prediction_time`. The ingestion-aware baseline is a secondary comparison.

# 5. Method

Improved Multi-Signal Validator: **Temporal / Future Aggregation / Target Lineage / Label Relation / Duplication**. Label relation includes an extra label-timing gate (six checks for five families). UTC timestamps; normalized full identifiers; governed roles/tables; exact hash or cosine >= **0.998**. Root-cause priority and all detection evidence are frozen. Historical retrieval must reconstruct past feature state ([Feast documentation](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)).

# 6. Primary Metric

**Leakage Recall @ FPR <= 5%**. Recall = TP/(TP+FN); FPR = FP/(FP+TN). Precision = TP/(TP+FP); F1 = 2TP/(2TP+FP+FN). Report recall at the preselected operating point only if its observed FPR meets the constraint. No Held-out calibration. Preprocessing statistics must be learned on training data ([scikit-learn guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)).

# 7. Final Result

Measured on the unchanged 120-case Held-out; generated from [heldout metrics](results/heldout_metrics.csv):

| Validator | Recall | FPR | Precision | F1 | FPR <= 5% |
| --- | --- | --- | --- | --- | --- |
| baseline | 20.00% | 0.00% | 100.00% | 33.33% | PASS |
| improved | 100.00% | 0.00% | 100.00% | 100.00% | PASS |

| Validator | TP | FP | TN | FN |
| --- | --- | --- | --- | --- |
| baseline | 16 | 0 | 40 | 64 |
| improved | 80 | 0 | 40 | 0 |

Observed constraint: **PASS** for Improved. This is a synthetic benchmark result, not a production guarantee. Outputs: `results/frozen_validator_config.json`, `heldout_predictions.csv`, `heldout_metrics.csv`, `generalization_gap.csv`, per-family/severity CSVs, confusion and recall PNGs. Earlier Held-out outputs existed; the freeze certifies this rerun's ordering only. [Research references](docs/references.md).

# 8. Failure Case

Improved Held-out FN/FP = **0/0**. `shipping_label` is handled by full-identifier matching. Separate measured probes expose an unknown semantic/label alias, a subtle near duplicate below threshold, and a benign HTTP `status_code` false positive; these are excluded from benchmark metrics. See [failure analysis](failure_analysis.md), [probe outputs](results/failure_probes.csv) and [four-slide outline](slides_outline.md). Production needs governed lineage, target IDs, availability timestamps and actual duplicate indexes.
