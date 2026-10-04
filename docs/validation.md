# Validation record

Validated on 2026-10-04 using a newly created Python 3.11.9 virtual environment installed from `requirements.txt`.

- `python scripts/run_all.py`: exit 0; regenerated/validated 300 cases, preserved existing CSVs, evaluated Dev then Held-out, exported metrics, predictions, charts and reports.
- `pytest -q`: **54 passed**, 14 dependency deprecation warnings. Full console output: [pytest_output.txt](../results/pytest_output.txt). Warnings originate from Matplotlib 3.9.2 calling deprecated pyparsing APIs; there are no failing tests.
- Repeated runner invocation preserved SHA-256 of all 16 CSV/PNG/freeze artifacts present during the reproducibility check. The original freeze timestamp remained unchanged.
- An independent standard-library CSV recount verified every Held-out TP/FP/TN/FN and Recall/FPR against [heldout_metrics.csv](../results/heldout_metrics.csv).
- Baseline: TP 16, FP 0, TN 40, FN 64; Recall 20%, FPR 0%, Precision 100%, F1 33.33%.
- Improved: TP 80, FP 0, TN 40, FN 0; Recall 100%, FPR 0%, Precision 100%, F1 100%. The observed FPR <= 5% constraint passes.
- Held-out retained all 120 cases (40 clean + 16 per leakage family); Dev/Held-out Recall and FPR gaps are zero.
- Protected source, configuration, vocabulary and all three input CSVs were compared with the supplied Git revision and found unchanged, allowing Git's CRLF/LF conversion only.
- README contains exactly eight main sections; the slide outline contains exactly four slides. Both are generated from actual CSV results.
- Recall and confusion charts were rendered and visually inspected; legends were moved above the plots to keep values readable.

The 100% synthetic Held-out result does not establish production accuracy. Independent diagnostic probes reveal three missed leakage scenarios and one benign `status_code` false positive. These probes are excluded from benchmark metrics. Earlier Held-out artifacts existed before this work; the new snapshot establishes ordering for this rerun only.

Packaging: `python scripts/package_submission.py` builds a ZIP beside the project, excluding `.git`, `.venv`, caches and temporary files. Keep the supplied frozen configuration with the archive; run the README commands in a fresh environment to reproduce it.
