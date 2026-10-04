"""Tune ImprovedValidator on the dev split only and freeze the result.

    python -m src.tune_validator                      # data/dev_cases.csv -> config/validator_config.json
    python -m src.tune_validator --dev path/to/dev.csv

Only near_duplicate_threshold is searched numerically. TARGET_TOKENS / LABEL_TABLES stay a
controlled vocabulary edited by hand with a semantic reason; the report shows which gates
produce false positives on dev so that edits are evidence-based.

Ground-truth columns are read here, only to score the validator on dev. Every row is stripped
of them before it reaches the validator. Any path that looks like a held-out split is refused.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from src.validators import (CHECK_NAMES, GROUND_TRUTH_FIELDS, ROOT_CAUSE_PRIORITY, ImprovedValidator,
                                ValidatorConfig, is_missing)
except ImportError:                        # executed as a script from inside src/
    from validators import (CHECK_NAMES, GROUND_TRUTH_FIELDS, ROOT_CAUSE_PRIORITY, ImprovedValidator,
                            ValidatorConfig, is_missing)

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_GRID = [0.90, 0.95, 0.97, 0.98, 0.99, 0.993, 0.995, 0.997, 0.998, 0.999, 0.9995, 0.9999, 1.0]
FPR_LIMIT = 0.05
HELDOUT_MARKERS = ("heldout", "held_out", "held-out", "holdout")
TRUE_VALUES = {"1", "1.0", "true", "t", "yes", "y", "leak", "leaky", "leakage", "positive"}
FALSE_VALUES = {"0", "0.0", "false", "f", "no", "n", "clean", "none", "no_leakage", "negative", "ok"}


def assert_dev_only(path: Path) -> None:
    if any(m in str(path).lower() for m in HELDOUT_MARKERS):
        raise SystemExit(f"Refusing to read {path}: tuning must use the dev split only.")


def truth_of(row: dict[str, str]) -> bool | None:
    """Evaluation label: is_leakage if present, else ground_truth (a type name counts as leaky)."""
    for col in ("is_leakage", "ground_truth"):
        if col not in row or is_missing(row[col]):
            continue
        v = str(row[col]).strip().lower()
        if v in TRUE_VALUES:
            return True
        if v in FALSE_VALUES:
            return False
        if col == "ground_truth":
            return True
    return None


def strip_truth(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k.strip().lower() not in GROUND_TRUTH_FIELDS}


def score(preds: list[dict[str, Any]], truth: list[bool]) -> dict[str, Any]:
    tp = sum(p["flagged"] and t for p, t in zip(preds, truth))
    fp = sum(p["flagged"] and not t for p, t in zip(preds, truth))
    fn = sum(t and not p["flagged"] for p, t in zip(preds, truth))
    tn = sum(not t and not p["flagged"] for p, t in zip(preds, truth))
    rec = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=rec, fpr=fpr, precision=prec, f1=f1)


def run(cfg: ValidatorConfig, rows: list[dict[str, Any]], truth: list[bool]):
    preds = ImprovedValidator(cfg).validate_many(rows)
    return preds, score(preds, truth)


def f3(x: float) -> str:
    return "n/a" if x != x else f"{x:.3f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tune ImprovedValidator on dev only")
    ap.add_argument("--dev", default=str(ROOT / "data" / "dev_cases.csv"))
    ap.add_argument("--config-out", default=str(ROOT / "config" / "validator_config.json"))
    ap.add_argument("--report-out", default=str(ROOT / "reports" / "validator_dev_tuning.md"))
    ap.add_argument("--grid", type=float, nargs="+", default=THRESHOLD_GRID)
    args = ap.parse_args(argv)

    dev_path = Path(args.dev)
    assert_dev_only(dev_path)
    if not dev_path.exists():
        raise SystemExit(f"Dev file not found: {dev_path}")
    with open(dev_path, newline="", encoding="utf-8-sig") as fh:
        raw = list(csv.DictReader(fh))
    digest = hashlib.sha256(dev_path.read_bytes()).hexdigest()

    # Schema preflight: a column the validator expects but dev does not have silently disables a
    # gate (every row warns "skipped"), which would look like a recall problem, not a schema one.
    base = ValidatorConfig()
    header = set(raw[0]) if raw else set()
    missing_cols = sorted(col for col in dataclasses.asdict(base.columns).values() if col not in header)
    if missing_cols:
        print(f"WARNING: dev has no column(s) {missing_cols}; the gates reading them will be skipped. "
              f"Map the dev names via ValidatorConfig.columns if they are only spelled differently.")

    labelled = [(r, truth_of(r)) for r in raw]
    skipped = sum(t is None for _, t in labelled)
    labelled = [(r, t) for r, t in labelled if t is not None]
    if not labelled:
        raise SystemExit("No dev row has a readable is_leakage / ground_truth value.")
    truth = [t for _, t in labelled]
    types = [r.get("leakage_type") or ("leak" if t else "clean") for r, t in labelled]
    ids = [r.get("case_id") or str(i) for i, (r, _) in enumerate(labelled)]
    rows = [strip_truth(r) for r, _ in labelled]          # the validator only ever sees these

    sweep = []
    for t in sorted(set(args.grid)):
        _, m = run(dataclasses.replace(base, near_duplicate_threshold=t), rows, truth)
        sweep.append((t, m))
    feasible = [(t, m) for t, m in sweep if m["fpr"] <= FPR_LIMIT]
    if feasible:
        best_t, _ = max(feasible, key=lambda x: (x[1]["recall"], -x[1]["fpr"], x[0]))
    else:
        best_t, _ = min(sweep, key=lambda x: (x[1]["fpr"], -x[1]["recall"], -x[0]))
    final_cfg = dataclasses.replace(base, near_duplicate_threshold=best_t)
    preds, final = run(final_cfg, rows, truth)

    ablations = [("temporal gate only (brief baseline rule)", ["temporal"])]
    ablations += [(f"without {name}", [n for n in CHECK_NAMES if n != name]) for name in CHECK_NAMES]
    ablation_rows = [(label, run(dataclasses.replace(final_cfg, enabled_checks=checks), rows, truth)[1])
                     for label, checks in ablations]

    gate_leaky, gate_clean, gate_sole_fp = Counter(), Counter(), Counter()
    for p, t in zip(preds, truth):
        for cause in p["detected_causes"]:
            (gate_leaky if t else gate_clean)[cause] += 1
        if not t and len(p["detected_causes"]) == 1:
            gate_sole_fp[p["detected_causes"][0]] += 1

    by_type = defaultdict(list)
    for p, t, ty in zip(preds, truth, types):
        by_type[ty].append((p, t))

    cfg_out = Path(args.config_out)
    cfg_out.parent.mkdir(parents=True, exist_ok=True)
    frozen = final_cfg.to_dict()
    frozen["tuning"] = dict(
        tuned_on=str(dev_path), dev_sha256=digest, rows=len(rows), rows_skipped_no_truth=skipped,
        dev_missing_columns=missing_cols,
        tuned_at=time.strftime("%Y-%m-%d %H:%M:%S"), searched="near_duplicate_threshold", grid=sorted(set(args.grid)),
        rule=f"max dev recall subject to dev FPR <= {FPR_LIMIT}; ties -> lower FPR -> higher threshold",
        fpr_target_met=bool(feasible),
        dev_metrics={k: (round(v, 4) if isinstance(v, float) else v) for k, v in final.items()},
        heldout_read=False)
    with open(cfg_out, "w", encoding="utf-8") as fh:
        json.dump(frozen, fh, indent=2)

    L: list[str] = []
    w = L.append
    w("# ImprovedValidator - dev tuning report\n")
    w(f"Dev file `{dev_path.name}` (sha256 `{digest[:16]}...`): {len(rows)} rows, {sum(truth)} leaky, "
      f"{len(truth) - sum(truth)} clean, {skipped} skipped without ground truth. Held-out was not read.\n")
    if missing_cols:
        w(f"**Schema warning:** dev has no column(s) {', '.join(f'`{c}`' for c in missing_cols)}; "
          "the gates reading them were skipped on every row.\n")
    w("## near_duplicate_threshold sweep\n")
    w("| Threshold | Recall | FPR | Precision | F1 | TP/FN/FP/TN | FPR <= 5% |")
    w("|---|---|---|---|---|---|---|")
    for t, m in sweep:
        mark = " **(selected)**" if t == best_t else ""
        w(f"| {t}{mark} | {f3(m['recall'])} | {f3(m['fpr'])} | {f3(m['precision'])} | {f3(m['f1'])} | "
          f"{m['tp']}/{m['fn']}/{m['fp']}/{m['tn']} | {'yes' if m['fpr'] <= FPR_LIMIT else 'no'} |")
    w(f"\nSelection rule: max recall subject to FPR <= {FPR_LIMIT:.0%}; ties -> lower FPR -> higher threshold."
      + ("" if feasible else " **No threshold met the FPR target; the lowest-FPR threshold was kept.**") + "\n")
    w("## Final dev metrics\n")
    w(f"near_duplicate_threshold = **{best_t}**  |  Recall **{f3(final['recall'])}**  |  FPR **{f3(final['fpr'])}**  |  "
      f"Precision {f3(final['precision'])}  |  F1 {f3(final['f1'])}  |  TP/FN/FP/TN {final['tp']}/{final['fn']}/{final['fp']}/{final['tn']}\n")
    w("## Gate firing on dev\n")
    w("| Cause | Fires on leaky rows | Fires on clean rows | Clean rows where it is the only cause (pure FP) |")
    w("|---|---|---|---|")
    for cause in ROOT_CAUSE_PRIORITY:
        w(f"| {cause} | {gate_leaky[cause]} | {gate_clean[cause]} | {gate_sole_fp[cause]} |")
    w("\n## Ablation on dev\n")
    w("| Configuration | Recall | FPR |")
    w("|---|---|---|")
    w(f"| all five gates | {f3(final['recall'])} | {f3(final['fpr'])} |")
    for label, m in ablation_rows:
        w(f"| {label} | {f3(m['recall'])} | {f3(m['fpr'])} |")
    w("\n## Per leakage_type (dev; leakage_type read for evaluation only)\n")
    w("| leakage_type | Rows | Flagged | Root causes assigned |")
    w("|---|---|---|---|")
    for ty in sorted(by_type):
        items = by_type[ty]
        roots = Counter(p["root_cause"] for p, _ in items)
        w(f"| {ty} | {len(items)} | {sum(p['flagged'] for p, _ in items)} | "
          + ", ".join(f"{k}: {v}" for k, v in roots.most_common()) + " |")
    missed = [(i, ty, p) for i, ty, p, t in zip(ids, types, preds, truth) if t and not p["flagged"]]
    fps = [(i, ty, p) for i, ty, p, t in zip(ids, types, preds, truth) if not t and p["flagged"]]
    w(f"\n## Missed leaky dev rows ({len(missed)}; first 40)\n")
    w("| case_id | leakage_type | Cosine | Warnings |")
    w("|---|---|---|---|")
    for i, ty, p in missed[:40]:
        cos = p["scores"].get("cosine_similarity")
        w(f"| {i} | {ty} | {'' if cos is None else f'{cos:.5f}'} | {'; '.join(p['warnings']) or '-'} |")
    w(f"\n## False positives on dev ({len(fps)}; first 40)\n")
    w("| case_id | Root cause | Reasons |")
    w("|---|---|---|")
    for i, ty, p in fps[:40]:
        w(f"| {i} | {p['root_cause']} | {'; '.join(p['reasons'])} |")
    w(f"\nFrozen config: `{cfg_out}`.")

    rep_out = Path(args.report_out)
    rep_out.parent.mkdir(parents=True, exist_ok=True)
    rep_out.write_text("\n".join(L) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")
    print(f"selected near_duplicate_threshold={best_t}  dev recall={f3(final['recall'])}  dev FPR={f3(final['fpr'])}"
          f"  (TP/FN/FP/TN {final['tp']}/{final['fn']}/{final['fp']}/{final['tn']})")
    print(f"config -> {cfg_out}\nreport -> {rep_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
