# ImprovedValidator - dev tuning report

Dev file `dev_cases.csv` (sha256 `46d7db984b4ce253...`): 180 rows, 120 leaky, 60 clean, 0 skipped without ground truth. Held-out was not read.

## Dev-reviewed vocabulary

Merged from `config/validator_vocabulary.json` on top of the built-in defaults. Each entry was added by hand after reading missed dev rows; the reason is the semantic justification.

| List | Entry | Reason |
|---|---|---|
| target_tokens | `outcome_code` | Coded case outcome, i.e. the value being predicted. Dev: TARGET-DERIVED latest(outcome_code); LABEL-SOURCE status_transition_rank(outcome_code). |
| target_tokens | `outcome_status` | Label column of the governed `labels` table. Dev: LABEL-SOURCE latest(outcome_status). |
| target_tokens | `resolution_code` | How a ticket / claim was resolved; written only once the outcome is known. Dev: TARGET-DERIVED rolling_mean(resolution_code) and resolution_velocity(ticket_history) lineage. |
| target_tokens | `status_code` | Terminal case status, which encodes the outcome. Dev: TARGET-DERIVED status_transition_rank(status_code); LABEL-SOURCE latest(status_code). Generic name: a benign HTTP status_code feature becomes a false positive (see probes). |

## near_duplicate_threshold sweep

| Threshold | Recall | FPR | Precision | F1 | TP/FN/FP/TN | FPR <= 5% |
|---|---|---|---|---|---|---|
| 0.9 | 1.000 | 0.000 | 1.000 | 1.000 | 120/0/0/60 | yes |
| 0.95 | 1.000 | 0.000 | 1.000 | 1.000 | 120/0/0/60 | yes |
| 0.97 | 1.000 | 0.000 | 1.000 | 1.000 | 120/0/0/60 | yes |
| 0.98 | 1.000 | 0.000 | 1.000 | 1.000 | 120/0/0/60 | yes |
| 0.99 | 1.000 | 0.000 | 1.000 | 1.000 | 120/0/0/60 | yes |
| 0.993 | 1.000 | 0.000 | 1.000 | 1.000 | 120/0/0/60 | yes |
| 0.995 | 1.000 | 0.000 | 1.000 | 1.000 | 120/0/0/60 | yes |
| 0.997 | 1.000 | 0.000 | 1.000 | 1.000 | 120/0/0/60 | yes |
| 0.998 **(selected)** | 1.000 | 0.000 | 1.000 | 1.000 | 120/0/0/60 | yes |
| 0.999 | 0.992 | 0.000 | 1.000 | 0.996 | 119/1/0/60 | yes |
| 0.9995 | 0.967 | 0.000 | 1.000 | 0.983 | 116/4/0/60 | yes |
| 0.9999 | 0.933 | 0.000 | 1.000 | 0.966 | 112/8/0/60 | yes |
| 1.0 | 0.867 | 0.000 | 1.000 | 0.929 | 104/16/0/60 | yes |

Selection rule: max recall subject to FPR <= 5%; ties -> lower FPR -> higher threshold.

## Final dev metrics

near_duplicate_threshold = **0.998**  |  Recall **1.000**  |  FPR **0.000**  |  Precision 1.000  |  F1 1.000  |  TP/FN/FP/TN 120/0/0/60

## Gate firing on dev

| Cause | Fires on leaky rows | Fires on clean rows | Clean rows where it is the only cause (pure FP) |
|---|---|---|---|
| temporal_leakage | 24 | 0 | 0 |
| future_aggregation_leakage | 48 | 0 | 0 |
| label_relation_leakage | 24 | 0 | 0 |
| target_derived_leakage | 48 | 0 | 0 |
| duplication_leakage | 24 | 0 | 0 |

## Ablation on dev

| Configuration | Recall | FPR | Root cause correct (flagged leaky rows) |
|---|---|---|---|
| all gates | 1.000 | 0.000 | 120/120 |
| temporal gate only (brief baseline rule) | 0.200 | 0.000 | 24/24 |
| without temporal | 1.000 | 0.000 | 96/120 |
| without future_aggregation | 0.800 | 0.000 | 96/96 |
| without target_lineage | 0.800 | 0.000 | 96/96 |
| without label_relation | 1.000 | 0.000 | 120/120 |
| without label_timing | 1.000 | 0.000 | 102/120 |
| without duplication | 0.800 | 0.000 | 96/96 |
| without dev-reviewed vocabulary (built-in only) | 0.800 | 0.000 | 96/96 |

## Per leakage_type (dev; leakage_type read for evaluation only)

| leakage_type | Rows | Flagged | Root causes assigned |
|---|---|---|---|
| clean | 60 | 0 | none: 60 |
| direct_temporal | 24 | 24 | temporal_leakage: 24 |
| duplication | 24 | 24 | duplication_leakage: 24 |
| future_aggregation | 24 | 24 | future_aggregation_leakage: 24 |
| label_source | 24 | 24 | label_relation_leakage: 24 |
| target_derived | 24 | 24 | target_derived_leakage: 24 |

Root-cause localization on flagged leaky dev rows: **120/120** (100.0%) report the cause expected for their leakage_type (direct_temporal -> temporal_leakage, future_aggregation -> future_aggregation_leakage, target_derived -> target_derived_leakage, label_source -> label_relation_leakage, duplication -> duplication_leakage).

## Missed leaky dev rows (0; first 40)

| case_id | leakage_type | Cosine | Warnings |
|---|---|---|---|

## False positives on dev (0; first 40)

| case_id | Root cause | Reasons |
|---|---|---|

## Known blind spots (hand-made probe rows, not dev or held-out)

The held-out split reuses the generator's leakage templates and alias names, so held-out recall measures coverage of known patterns. These probes run the frozen config on patterns it was not built for.

| Probe | Truly leaky | Flagged | Outcome | Root cause |
|---|---|---|---|---|
| target alias outside the reviewed vocabulary | yes | no | FN (missed) | none |
| label copied into a feature table, label time not recorded | yes | no | FN (missed) | none |
| benign HTTP status_code feature (generic alias name) | no | yes | FP | target_derived_leakage |
| cross-split near duplicate below the cosine threshold | yes | no | FN (missed) | none |

Frozen config: `config/validator_config.json`.
