"""Generate submission prose from measured CSVs, never from expected benchmark numbers."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.frozen_evaluation import write_json, sha256
from src.tune_validator import PROBE_BASE, PROBES
from src.validators import ImprovedValidator


def table(headers, rows) -> str:
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |",
                       *("| " + " | ".join(escape(v) for v in row) + " |" for row in rows)])


def percent(value) -> str:
    return f"{value:.2%}"


def run_probes(validator: ImprovedValidator) -> pd.DataFrame:
    probes = list(PROBES) + [
        ("benign shipping_label_length", False,
         {"feature_expression": "shipping_label_length", "lineage_columns": '["shipping_label", "package_weight"]',
          "source_table": "shipping_events"}),
    ]
    records = []
    for description, leaky, overrides in probes:
        result = validator.validate({**PROBE_BASE, **overrides})
        kind = ("TP" if result["flagged"] else "FN") if leaky else ("FP" if result["flagged"] else "TN")
        records.append(dict(probe=description, ground_truth=int(leaky), flagged=result["flagged"],
                            outcome=kind, root_cause=result["root_cause"],
                            cosine_similarity=result["scores"].get("cosine_similarity"),
                            reasons=json.dumps(result["reasons"]), warnings=json.dumps(result["warnings"])))
    return pd.DataFrame(records)


def build_reports(root: Path) -> None:
    results = root / "results"
    metrics = pd.read_csv(results / "heldout_metrics.csv")
    predictions = pd.read_csv(results / "heldout_predictions.csv", keep_default_na=False)
    gap = pd.read_csv(results / "generalization_gap.csv")
    snapshot = json.loads((results / "frozen_validator_config.json").read_text(encoding="utf-8"))
    cfg = snapshot["effective_config"]
    probes = run_probes(ImprovedValidator(cfg))
    probes.to_csv(results / "failure_probes.csv", index=False)
    primary = metrics[metrics.validator.isin(["baseline", "improved"])]
    scores = table(["Validator", "Recall", "FPR", "Precision", "F1", "FPR <= 5%"], [
        [r.validator, percent(r.recall), percent(r.fpr), percent(r.precision), percent(r.f1),
         "PASS" if r.fpr_constraint_pass else "FAIL"] for r in primary.itertuples()])
    confusion = table(["Validator", "TP", "FP", "TN", "FN"], [
        [r.validator, r.TP, r.FP, r.TN, r.FN] for r in primary.itertuples()])
    get_metric = lambda name: metrics[metrics.validator == name].iloc[0]
    baseline, improved = get_metric("baseline"), get_metric("improved")
    patterns = {
        "direct_temporal": "feature_time exceeds prediction_time",
        "future_aggregation": "valid feature_time hides a future aggregation boundary/source event",
        "target_derived": "baseline does not inspect expression or target lineage",
        "label_source": "baseline does not inspect source role or label availability",
        "duplication": "baseline has no counterpart hash/vector comparison",
    }
    failures = []
    for name in ("baseline", "improved"):
        wrong = predictions[predictions[f"{name}_flagged"].astype(int) != predictions.ground_truth.astype(int)]
        for row in wrong.to_dict("records"):
            error = "FN" if row["ground_truth"] else "FP"
            reason = patterns.get(row["leakage_type"], "metadata/lexical ambiguity; review supplied evidence")
            if error == "FP":
                reason = "Review emitted cause/reasons; do not infer a specific ambiguity without evidence"
            failures.append([name, error, row["case_id"], row["leakage_type"], row["feature_expression"],
                             row["improved_root_cause"], reason])
    family_counts = Counter(row[3] for row in failures if row[0] == "baseline" and row[1] == "FN")
    failure_table = table(["Validator", "Error", "Case", "Family", "Expression", "Improved cause", "Explanation"], failures)
    probe_table = table(["Separate probe", "True leakage", "Flagged", "Outcome", "Cause", "Cosine"], [
        [r.probe, bool(r.ground_truth), r.flagged, r.outcome, r.root_cause,
         f"{r.cosine_similarity:.6f}" if pd.notna(r.cosine_similarity) else "n/a"] for r in probes.itertuples()])
    shipping = predictions[(predictions.ground_truth.astype(int) == 0)
                           & predictions.feature_expression.str.contains("shipping_label", regex=False)]
    duplicate_rows = predictions[predictions.leakage_type == "duplication"]
    cosines = [json.loads(value).get("cosine_similarity") for value in duplicate_rows.improved_scores]
    cosines = [value for value in cosines if value is not None]
    cosine_text = f"Minimum evaluated near-duplicate cosine: {min(cosines):.6f}." if cosines else "No near-duplicate cosines reported."
    generalization = table(["Validator", "Dev Recall", "Held-out Recall", "Dev FPR", "Held-out FPR", "Recall gap (pp)", "FPR gap (pp)"], [
        [r.validator, percent(r.dev_recall), percent(r.heldout_recall), percent(r.dev_fpr), percent(r.heldout_fpr),
         f"{100*r.recall_gap:.2f}", f"{100*r.fpr_gap:.2f}"] for r in gap.itertuples()])
    warnings = Counter(w for value in predictions.improved_warnings for w in json.loads(value))
    warning_table = table(["Warning", "Held-out rows"], sorted(warnings.items()))
    fn_text = ("There are no Improved false negatives on these 120 cases; no Held-out FN examples can be claimed."
               if improved.FN == 0 else f"All {int(improved.FN)} Improved FNs are included in the complete case ledger below.")
    fp_text = ("There are no Improved false positives on these 120 cases; no Held-out FP examples can be claimed."
               if improved.FP == 0 else f"All {int(improved.FP)} Improved FPs are included in the complete case ledger below.")
    analysis = f"""# Failure Analysis

## 1. Final Held-out Result

Independent rerun of the supplied, unchanged benchmark: 120 cases = 40 clean + 16 per leakage family.
Positive means leakage. Predictions use the original binary decisions; no second threshold is optimized.
Source: `results/heldout_metrics.csv` and `results/heldout_predictions.csv`.

{scores}

Freeze: `{snapshot['frozen_at_utc']}` (UTC); near-duplicate threshold **{cfg['near_duplicate_threshold']}**, comparison `cosine >= threshold`.
Target identifiers: {', '.join('`'+token+'`' for token in cfg['target_tokens'])}.
Label tables: {', '.join('`'+token+'`' for token in cfg['label_tables'])}; roles: {', '.join(cfg['label_roles'])}.
Timestamp tolerance: {cfg['temporal_tolerance_seconds']} seconds; naive timestamp UTC offset: {cfg['naive_utc_offset_hours']} hours.
All six gates and root-cause precedence are captured in `results/frozen_validator_config.json`, with SHA-256 of protected code/config/data and the supplied Git commit.

Audit limitation: Held-out results already existed before this work. This snapshot establishes freeze ordering for this rerun; it cannot retroactively prove that previous authors never viewed Held-out.
The independent evaluator performs no tuning and leaves `src/validators.py`, vocabulary, thresholds and ground truth unchanged.
The earlier `results/frozen_threshold.json` value 1.0 is a threshold on a binary decision score, **not** the cosine threshold; it is retained as a legacy artifact and is not used by this runner.

## 2. Confusion Matrix

{confusion}

Plots use rows = true class, columns = predicted class, ordered clean/leakage: `[[TN, FP], [FN, TP]]`.
See `results/confusion_baseline.png` and `results/confusion_improved.png`.

## 3. False Negatives

Baseline misses {int(baseline.FN)} cases: {', '.join(f'{family}: {count}' for family, count in sorted(family_counts.items()))}.
Each missed case has a valid feature timestamp but requires a signal the baseline does not inspect.
{fn_text}
`resolution_code` and `status_code` are already in Agent #3's Dev-reviewed vocabulary; they are not new Held-out failures.
{cosine_text} These measured pairs do not establish detection of duplicates below the frozen threshold.

## 4. False Positives

Baseline FP = {int(baseline.FP)}; Improved FP = {int(improved.FP)}.
{fp_text}
The {len(shipping)} clean Held-out `shipping_label` cases produce {int(shipping.improved_flagged.sum())} Improved flags.
Policy: normalized full identifiers are matched, so `shipping_label` and `shipping_label_length` do not match the identifier `label`.
For observed FPs, categories to inspect are lexical ambiguity, metadata ambiguity, threshold error, parser error or other; there is no observed Held-out FP to assign here.

## 5. Failure Pattern

The following are **hand-made diagnostic probes**, inherited from the Dev tuning report plus a benign shipping control.
They are outside Dev/Held-out, have explicit scenario assumptions and do not contribute to any benchmark metric.
Their outputs are measured with the frozen config in `results/failure_probes.csv`; they are never used to tune it.

{probe_table}

Three important mechanisms: (1) unknown target/label aliases and missing source metadata can yield false negatives;
(2) broad, context-free alias vocabulary can turn a benign HTTP `status_code` into a lexical-ambiguity false positive;
(3) real near duplicates below cosine {cfg['near_duplicate_threshold']} are missed by the current threshold.
The label-timing rule also assumes that label metadata refers to the prediction target; prior labels for another task could be legitimate features and require target-specific governance.

## 6. What the Validator Cannot Infer

**Column name alone != semantics.** A terminal case status and an HTTP response status can share a name.
Missing lineage/source role/availability metadata prevents reliable semantic inference. Unknown or malformed fields produce warnings and may leave a case unflagged; an unflagged row is not a proof of correctness.

{warning_table}

Synthetic cases share generator templates and alias names across Dev/Held-out; the split is by index within each family, not an independent external distribution.
Counterpart vectors/hashes are supplied inside each case. The benchmark does not retrieve actual train/test neighbors or parse production SQL/query plans.
An empty `other_split_vector` serializes as `[]` and generates a parser warning; this is recorded, not silently removed from the dataset.
The apparent 100% result demonstrates coverage of these known templates, not universal leakage detection.
With only 40 clean Held-out rows, each FP changes FPR by 2.5 percentage points; two FPs meet 5%, three exceed it.
For zero observed FPs, the one-sided exact 95% binomial upper bound is `1 - 0.05**(1/40)` = {percent(1 - 0.05**(1/40))}; 0% observed FPR does not establish a population FPR <= 5%.

## 7. Production Recommendation

- Govern feature lineage with target IDs and source roles scoped to the current prediction task; use a data catalog rather than names alone.
- Record event, ingestion/availability and correction timestamps; reconstruct historical feature states using as-of joins and aggregation cutoffs.
- Generate canonical hashes and an index over actual splits; use blocking/ANN to retrieve near-duplicate candidates at scale, then validate similarity.
- Route missing/invalid metadata warnings to review and measure coverage, rather than treating every unflagged row as safe.
- Evaluate unseen aliases, new leakage templates and independent time windows in a new sealed benchmark. Record any future method revision as a new version.

## 8. Dev vs Held-out Generalization

Recall gap = Dev Recall - Held-out Recall. FPR gap = Held-out FPR - Dev FPR.

{generalization}

Source: `results/generalization_gap.csv`. This is a validator generalization/validation optimism comparison, not evidence of model overfitting.
Zero gap on shared templates does not rule out a gap on independent production data.
The optional downstream model experiment was not run; no model AUC/accuracy or offline-vs-clean optimism numbers are claimed.

## 9. Complete Held-out Error Ledger

Every observed error of the primary baseline and Improved validator is listed below; no difficult case is removed.
Machine-readable evidence for all three validators, including the optional ingestion baseline: `results/heldout_failures.csv`.

{failure_table}
"""
    (root / "failure_analysis.md").write_text(analysis, encoding="utf-8")
    readme = f"""# 1. Problem

Training features can contain information unavailable at prediction time. A timestamp-only check misses target-derived, aggregation and split-contamination leakage. Leakage includes more than time violations ([Kapoor & Narayanan, 2023](https://doi.org/10.1016/j.patter.2023.100804)).

# 2. Hypothesis

Combining point-in-time timestamps, aggregation boundaries, lineage, source roles and cross-split duplication improves leakage recall over the baseline while maintaining **FPR <= 5%**.

# 3. Setup

Python 3.11; `SEED = 20261004`. Synthetic **300 cases: 100 clean + 200 leakage** across five families. Stratified Dev/Held-out = **180/120 (60/40)**; Held-out contains 40 clean + 16 cases per family. Shared templates limit external generalization.

Run from this repository root (PowerShell):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python scripts/run_all.py
pytest -q
```

If activation is disabled, use `.\\.venv\\Scripts\\python.exe scripts/run_all.py` and `.\\.venv\\Scripts\\python.exe -m pytest -q` after installing with that interpreter.
The runner regenerates data in staging, verifies existing CSV content (CRLF/LF conversion allowed), checks schema/labels/counts, freezes Agent #3's supplied configuration before Held-out prediction, runs both validators and exports metrics/predictions/charts/docs. It never tunes. Existing snapshots are verified and preserved; changed protected inputs stop the run. See [submission guide](docs/submission.md) for the file tree and audit details.

# 4. Baseline

Primary baseline: flag only when `feature_time > prediction_time`. The ingestion-aware baseline is a secondary comparison.

# 5. Method

Improved Multi-Signal Validator: **Temporal / Future Aggregation / Target Lineage / Label Relation / Duplication**. Label relation includes an extra label-timing gate (six checks for five families). UTC timestamps; normalized full identifiers; governed roles/tables; exact hash or cosine >= **{cfg['near_duplicate_threshold']}**. Root-cause priority and all detection evidence are frozen. Historical retrieval must reconstruct past feature state ([Feast documentation](https://docs.feast.dev/getting-started/concepts/point-in-time-joins)).

# 6. Primary Metric

**Leakage Recall @ FPR <= 5%**. Recall = TP/(TP+FN); FPR = FP/(FP+TN). Precision = TP/(TP+FP); F1 = 2TP/(2TP+FP+FN). Report recall at the preselected operating point only if its observed FPR meets the constraint. No Held-out calibration. Preprocessing statistics must be learned on training data ([scikit-learn guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)).

# 7. Final Result

Measured on the unchanged 120-case Held-out; generated from [heldout metrics](results/heldout_metrics.csv):

{scores}

{confusion}

Observed constraint: **{'PASS' if improved.fpr_constraint_pass else 'FAIL'}** for Improved. This is a synthetic benchmark result, not a production guarantee. Outputs: `results/frozen_validator_config.json`, `heldout_predictions.csv`, `heldout_metrics.csv`, `generalization_gap.csv`, per-family/severity CSVs, confusion and recall PNGs. Earlier Held-out outputs existed; the freeze certifies this rerun's ordering only. [Research references](docs/references.md).

# 8. Failure Case

Improved Held-out FN/FP = **{int(improved.FN)}/{int(improved.FP)}**. `shipping_label` is handled by full-identifier matching. Separate measured probes expose an unknown semantic/label alias, a subtle near duplicate below threshold, and a benign HTTP `status_code` false positive; these are excluded from benchmark metrics. See [failure analysis](failure_analysis.md), [probe outputs](results/failure_probes.csv) and [four-slide outline](slides_outline.md). Production needs governed lineage, target IDs, availability timestamps and actual duplicate indexes.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    slides = f"""# Slide 1 - Pain Point & Hypothesis

- Training data can see the future.
- Baseline: `feature_time > prediction_time`.
- Hidden leakage survives valid timestamps.
- Hypothesis: multi-signal checks increase Recall at FPR <= 5%.

# Slide 2 - Dataset & Approach

```text
300 cases -> 100 Clean + 200 Leakage -> 5 Families
         -> Dev 180 / Held-out 120 -> Frozen Validator
```

Time | Aggregation | Lineage | Source + label timing | Duplicate

Seed 20261004; cosine threshold {cfg['near_duplicate_threshold']}; no Held-out tuning.

# Slide 3 - Evidence & Result

| Held-out (120 cases) | Baseline | Improved |
| --- | --- | --- |
| Recall | {percent(baseline.recall)} | {percent(improved.recall)} |
| FPR | {percent(baseline.fpr)} | {percent(improved.fpr)} |
| FP / FN | {int(baseline.FP)} / {int(baseline.FN)} | {int(improved.FP)} / {int(improved.FN)} |

Visual: `results/recall_by_type.png`; evidence: `results/heldout_metrics.csv`.
Improved observed FPR constraint: **{'PASS' if improved.fpr_constraint_pass else 'FAIL'}**. Shared synthetic templates limit this result.

# Slide 4 - Decision & Remaining Risk

- Decision: {'multi-signal improves recall under the observed FPR constraint' if improved.fpr_constraint_pass and improved.recall > baseline.recall else 'review the measured recall/FPR tradeoff before adoption'}.
- Probe risks: unseen aliases, benign status_code, below-threshold duplicates.
- Production: target-specific lineage + availability timestamps + hash index / ANN.
- Next evidence: sealed unseen templates and independent time windows.

Sources and audit caveats: `docs/references.md`, `failure_analysis.md`.
"""
    (root / "slides_outline.md").write_text(slides, encoding="utf-8")
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    files = sorted({"docs/submission.md"} | {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                   and not any(part in {".git", ".venv", "__pycache__", ".pytest_cache", "tmp"} for part in p.relative_to(root).parts)})
    guide = """# Submission and reproduction guide

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
""" + "\n".join(files) + "\n```\n"
    (docs / "submission.md").write_text(guide, encoding="utf-8")
    # Refresh the manifest after probe/report generation.
    manifest_path = results / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["report_generator_sha256"] = sha256(Path(__file__))
    manifest["documentation_sha256"] = {name: sha256(root / name) for name in
                                        ("README.md", "slides_outline.md", "failure_analysis.md", "docs/submission.md", "docs/references.md")}
    manifest["artifact_sha256"]["failure_probes.csv"] = sha256(results / "failure_probes.csv")
    write_json(manifest_path, manifest)
