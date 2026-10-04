# ImprovedValidator - dev tuning report

Dev file `dev_cases.csv` (sha256 `46d7db984b4ce253...`): 180 rows, 120 leaky, 60 clean, 0 skipped without ground truth. Held-out was not read.

## near_duplicate_threshold sweep

| Threshold | Recall | FPR | Precision | F1 | TP/FN/FP/TN | FPR <= 5% |
|---|---|---|---|---|---|---|
| 0.9 | 0.650 | 0.000 | 1.000 | 0.788 | 78/42/0/60 | yes |
| 0.95 | 0.650 | 0.000 | 1.000 | 0.788 | 78/42/0/60 | yes |
| 0.97 | 0.650 | 0.000 | 1.000 | 0.788 | 78/42/0/60 | yes |
| 0.98 | 0.650 | 0.000 | 1.000 | 0.788 | 78/42/0/60 | yes |
| 0.99 | 0.650 | 0.000 | 1.000 | 0.788 | 78/42/0/60 | yes |
| 0.993 | 0.650 | 0.000 | 1.000 | 0.788 | 78/42/0/60 | yes |
| 0.995 | 0.650 | 0.000 | 1.000 | 0.788 | 78/42/0/60 | yes |
| 0.997 | 0.650 | 0.000 | 1.000 | 0.788 | 78/42/0/60 | yes |
| 0.998 **(selected)** | 0.650 | 0.000 | 1.000 | 0.788 | 78/42/0/60 | yes |
| 0.999 | 0.642 | 0.000 | 1.000 | 0.782 | 77/43/0/60 | yes |
| 0.9995 | 0.617 | 0.000 | 1.000 | 0.763 | 74/46/0/60 | yes |
| 0.9999 | 0.583 | 0.000 | 1.000 | 0.737 | 70/50/0/60 | yes |
| 1.0 | 0.517 | 0.000 | 1.000 | 0.681 | 62/58/0/60 | yes |

Selection rule: max recall subject to FPR <= 5%; ties -> lower FPR -> higher threshold.

## Final dev metrics

near_duplicate_threshold = **0.998**  |  Recall **0.650**  |  FPR **0.000**  |  Precision 1.000  |  F1 0.788  |  TP/FN/FP/TN 78/42/0/60

## Gate firing on dev

| Cause | Fires on leaky rows | Fires on clean rows | Clean rows where it is the only cause (pure FP) |
|---|---|---|---|
| temporal_leakage | 24 | 0 | 0 |
| future_aggregation_leakage | 48 | 0 | 0 |
| target_derived_leakage | 0 | 0 | 0 |
| label_relation_leakage | 6 | 0 | 0 |
| duplication_leakage | 24 | 0 | 0 |

## Ablation on dev

| Configuration | Recall | FPR |
|---|---|---|
| all five gates | 0.650 | 0.000 |
| temporal gate only (brief baseline rule) | 0.200 | 0.000 |
| without temporal | 0.650 | 0.000 |
| without future_aggregation | 0.450 | 0.000 |
| without target_lineage | 0.650 | 0.000 |
| without label_relation | 0.600 | 0.000 |
| without duplication | 0.450 | 0.000 |

## Per leakage_type (dev; leakage_type read for evaluation only)

| leakage_type | Rows | Flagged | Root causes assigned |
|---|---|---|---|
| clean | 60 | 0 | none: 60 |
| direct_temporal | 24 | 24 | temporal_leakage: 24 |
| duplication | 24 | 24 | duplication_leakage: 24 |
| future_aggregation | 24 | 24 | future_aggregation_leakage: 24 |
| label_source | 24 | 6 | none: 18, label_relation_leakage: 6 |
| target_derived | 24 | 0 | none: 24 |

## Missed leaky dev rows (42; first 40)

| case_id | leakage_type | Cosine | Warnings |
|---|---|---|---|
| TARGET-DERIVED-000 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-001 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-002 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-003 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-004 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-005 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-006 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-007 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-008 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-009 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-010 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-011 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-012 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-013 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-014 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-015 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-016 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-017 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-018 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-019 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-020 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-021 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-022 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| TARGET-DERIVED-023 | target_derived |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-001 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-002 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-003 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-005 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-006 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-007 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-009 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-010 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-011 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-013 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-014 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-015 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-017 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-018 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-019 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |
| LABEL-SOURCE-021 | label_source |  | other_split_vector: vector is not a non-empty list; near-duplicate check skipped |

## False positives on dev (0; first 40)

| case_id | Root cause | Reasons |
|---|---|---|

Frozen config: `C:\Bo\VinAI\LEC\261004_LEC17\Lab_day18lec\config\validator_config.json`.
