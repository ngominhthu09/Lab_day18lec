# Slide 1 - Pain Point & Hypothesis

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

Seed 20261004; cosine threshold 0.998; no Held-out tuning.

# Slide 3 - Evidence & Result

| Held-out (120 cases) | Baseline | Improved |
| --- | --- | --- |
| Recall | 20.00% | 100.00% |
| FPR | 0.00% | 0.00% |
| FP / FN | 0 / 64 | 0 / 0 |

Visual: `results/recall_by_type.png`; evidence: `results/heldout_metrics.csv`.
Improved observed FPR constraint: **PASS**. Shared synthetic templates limit this result.

# Slide 4 - Decision & Remaining Risk

- Decision: multi-signal improves recall under the observed FPR constraint.
- Probe risks: unseen aliases, benign status_code, below-threshold duplicates.
- Production: target-specific lineage + availability timestamps + hash index / ANN.
- Next evidence: sealed unseen templates and independent time windows.

Sources and audit caveats: `docs/references.md`, `failure_analysis.md`.
