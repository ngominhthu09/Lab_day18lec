# Submission and reproduction guide

From the repo root: `python scripts/run_all.py`, then `pytest -q`, in the environment installed from `requirements.txt`.
The exact dependency versions used by this run are recorded in `results/environment_lock.txt`; Python/package versions also appear in the evaluation manifest.

`run_all` creates the seed dataset in temporary staging and validates it before comparing every existing CSV byte-for-byte after CRLF/LF normalization only.
It preserves existing data and fails on disagreement. It runs the primary baseline, optional ingestion baseline and Improved validator on Dev then Held-out.
`results/frozen_validator_config.json` is created exclusively before Held-out CSV parsing/prediction and reused unchanged on reruns.
Raw SHA-256 records original bytes; additional normalized hashes allow Git's CRLF/LF conversion across operating systems. Checks protect validator/baseline code, generator/config/vocabulary, Dev and Held-out CSVs. Agent #3's recorded Dev hash must agree with raw, LF or CRLF bytes.
Protected-input changes fail instead of silently creating a new freeze. A method revision requires a new explicitly versioned experiment and new sealed Held-out evidence.

`src/evaluate.py` retains the public metric utilities and routes orchestration through the frozen evaluator.
The evaluator uses binary `flagged` decisions directly; no additional score threshold is needed.
The legacy `results/frozen_threshold.json` is preserved for audit but is not used. Its binary-score threshold 1.0 differs from the cosine threshold 0.998.
Per-family/severity charts show Held-out, with the ingestion baseline as a secondary comparison.
Every case is exported; every incorrect prediction is included in `heldout_failures.csv`, with warnings/scores retained for independent review.
`failure_probes.csv` is a separate diagnostic artifact; it contributes zero cases to the 300-case benchmark.
README, slide outline and failure analysis are regenerated from actual CSV outputs to prevent stale numbers.

The repo originally contained Held-out metrics. The snapshot establishes ordering for this rerun only, not historical secrecy.
No optional downstream model metric is reported. The slide deliverable is a four-slide Markdown outline, not a PowerPoint deck.

## Project file inventory (generated)

```text
.gitignore
README.md
config/__init__.py
config/data_config.py
config/validator_config.json
config/validator_vocabulary.json
data/all_cases.csv
data/dev_cases.csv
data/heldout_cases.csv
docs/references.md
docs/submission.md
docs/validation.md
failure_analysis.md
lakehouse-open-research-challenges-student-brief.pdf
pytest.ini
reports/validator_dev_tuning.md
requirements.txt
results/comparison_dev_vs_heldout.csv
results/confusion_baseline.png
results/confusion_baseline_ingestion.png
results/confusion_improved.png
results/environment_lock.txt
results/evaluation_manifest.json
results/failure_probes.csv
results/frozen_threshold.json
results/frozen_validator_config.json
results/generalization_gap.csv
results/heldout_failures.csv
results/heldout_metrics.csv
results/heldout_predictions.csv
results/metrics_by_severity.csv
results/metrics_by_type.csv
results/metrics_overall.csv
results/predictions.csv
results/pytest_output.txt
results/recall_by_severity.png
results/recall_by_type.png
scripts/package_submission.py
scripts/run_all.py
slides_outline.md
src/__init__.py
src/baseline.py
src/evaluate.py
src/frozen_evaluation.py
src/generate_data.py
src/submission_reports.py
src/tune_validator.py
src/validators.py
tests/__init__.py
tests/test_evaluation.py
tests/test_frozen_evaluation.py
tests/test_validators.py
```
