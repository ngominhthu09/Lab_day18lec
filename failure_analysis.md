# Failure Analysis

## 1. Final Held-out Result

Independent rerun of the supplied, unchanged benchmark: 120 cases = 40 clean + 16 per leakage family.
Positive means leakage. Predictions use the original binary decisions; no second threshold is optimized.
Source: `results/heldout_metrics.csv` and `results/heldout_predictions.csv`.

| Validator | Recall | FPR | Precision | F1 | FPR <= 5% |
| --- | --- | --- | --- | --- | --- |
| baseline | 20.00% | 0.00% | 100.00% | 33.33% | PASS |
| improved | 100.00% | 0.00% | 100.00% | 100.00% | PASS |

Freeze: `2026-10-04T10:43:39.759907+00:00` (UTC); near-duplicate threshold **0.998**, comparison `cosine >= threshold`.
Target identifiers: `target`, `label`, `outcome`, `churn_label`, `default_label`, `y_true`, `ground_truth`, `outcome_code`, `outcome_status`, `resolution_code`, `status_code`.
Label tables: `labels`, `training_labels`, `outcomes`, `target_events`; roles: label, target.
Timestamp tolerance: 0.0 seconds; naive timestamp UTC offset: 0.0 hours.
All six gates and root-cause precedence are captured in `results/frozen_validator_config.json`, with SHA-256 of protected code/config/data and the supplied Git commit.

Audit limitation: Held-out results already existed before this work. This snapshot establishes freeze ordering for this rerun; it cannot retroactively prove that previous authors never viewed Held-out.
The independent evaluator performs no tuning and leaves `src/validators.py`, vocabulary, thresholds and ground truth unchanged.
The earlier `results/frozen_threshold.json` value 1.0 is a threshold on a binary decision score, **not** the cosine threshold; it is retained as a legacy artifact and is not used by this runner.

## 2. Confusion Matrix

| Validator | TP | FP | TN | FN |
| --- | --- | --- | --- | --- |
| baseline | 16 | 0 | 40 | 64 |
| improved | 80 | 0 | 40 | 0 |

Plots use rows = true class, columns = predicted class, ordered clean/leakage: `[[TN, FP], [FN, TP]]`.
See `results/confusion_baseline.png` and `results/confusion_improved.png`.

## 3. False Negatives

Baseline misses 64 cases: duplication: 16, future_aggregation: 16, label_source: 16, target_derived: 16.
Each missed case has a valid feature timestamp but requires a signal the baseline does not inspect.
There are no Improved false negatives on these 120 cases; no Held-out FN examples can be claimed.
`resolution_code` and `status_code` are already in Agent #3's Dev-reviewed vocabulary; they are not new Held-out failures.
Minimum evaluated near-duplicate cosine: 0.999449. These measured pairs do not establish detection of duplicates below the frozen threshold.

## 4. False Positives

Baseline FP = 0; Improved FP = 0.
There are no Improved false positives on these 120 cases; no Held-out FP examples can be claimed.
The 10 clean Held-out `shipping_label` cases produce 0 Improved flags.
Policy: normalized full identifiers are matched, so `shipping_label` and `shipping_label_length` do not match the identifier `label`.
For observed FPs, categories to inspect are lexical ambiguity, metadata ambiguity, threshold error, parser error or other; there is no observed Held-out FP to assign here.

## 5. Failure Pattern

The following are **hand-made diagnostic probes**, inherited from the Dev tuning report plus a benign shipping control.
They are outside Dev/Held-out, have explicit scenario assumptions and do not contribute to any benchmark metric.
Their outputs are measured with the frozen config in `results/failure_probes.csv`; they are never used to tune it.

| Separate probe | True leakage | Flagged | Outcome | Cause | Cosine |
| --- | --- | --- | --- | --- | --- |
| target alias outside the reviewed vocabulary | True | False | FN | none | n/a |
| label copied into a feature table, label time not recorded | True | False | FN | none | n/a |
| benign HTTP status_code feature (generic alias name) | False | True | FP | target_derived_leakage | n/a |
| cross-split near duplicate below the cosine threshold | True | False | FN | none | 0.991655 |
| benign shipping_label_length | False | False | TN | none | n/a |

Three important mechanisms: (1) unknown target/label aliases and missing source metadata can yield false negatives;
(2) broad, context-free alias vocabulary can turn a benign HTTP `status_code` into a lexical-ambiguity false positive;
(3) real near duplicates below cosine 0.998 are missed by the current threshold.
The label-timing rule also assumes that label metadata refers to the prediction target; prior labels for another task could be legitimate features and require target-specific governance.

## 6. What the Validator Cannot Infer

**Column name alone != semantics.** A terminal case status and an HTTP response status can share a name.
Missing lineage/source role/availability metadata prevents reliable semantic inference. Unknown or malformed fields produce warnings and may leave a case unflagged; an unflagged row is not a proof of correctness.

| Warning | Held-out rows |
| --- | --- |
| other_split_vector: vector is not a non-empty list; near-duplicate check skipped | 104 |

Synthetic cases share generator templates and alias names across Dev/Held-out; the split is by index within each family, not an independent external distribution.
Counterpart vectors/hashes are supplied inside each case. The benchmark does not retrieve actual train/test neighbors or parse production SQL/query plans.
An empty `other_split_vector` serializes as `[]` and generates a parser warning; this is recorded, not silently removed from the dataset.
The apparent 100% result demonstrates coverage of these known templates, not universal leakage detection.
With only 40 clean Held-out rows, each FP changes FPR by 2.5 percentage points; two FPs meet 5%, three exceed it.
For zero observed FPs, the one-sided exact 95% binomial upper bound is `1 - 0.05**(1/40)` = 7.22%; 0% observed FPR does not establish a population FPR <= 5%.

## 7. Production Recommendation

- Govern feature lineage with target IDs and source roles scoped to the current prediction task; use a data catalog rather than names alone.
- Record event, ingestion/availability and correction timestamps; reconstruct historical feature states using as-of joins and aggregation cutoffs.
- Generate canonical hashes and an index over actual splits; use blocking/ANN to retrieve near-duplicate candidates at scale, then validate similarity.
- Route missing/invalid metadata warnings to review and measure coverage, rather than treating every unflagged row as safe.
- Evaluate unseen aliases, new leakage templates and independent time windows in a new sealed benchmark. Record any future method revision as a new version.

## 8. Dev vs Held-out Generalization

Recall gap = Dev Recall - Held-out Recall. FPR gap = Held-out FPR - Dev FPR.

| Validator | Dev Recall | Held-out Recall | Dev FPR | Held-out FPR | Recall gap (pp) | FPR gap (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 20.00% | 20.00% | 0.00% | 0.00% | 0.00 | 0.00 |
| baseline_ingestion | 20.00% | 20.00% | 0.00% | 0.00% | 0.00 | 0.00 |
| improved | 100.00% | 100.00% | 0.00% | 0.00% | 0.00 | 0.00 |

Source: `results/generalization_gap.csv`. This is a validator generalization/validation optimism comparison, not evidence of model overfitting.
Zero gap on shared templates does not rule out a gap on independent production data.
The optional downstream model experiment was not run; no model AUC/accuracy or offline-vs-clean optimism numbers are claimed.

## 9. Complete Held-out Error Ledger

Every observed error of the primary baseline and Improved validator is listed below; no difficult case is removed.
Machine-readable evidence for all three validators, including the optional ingestion baseline: `results/heldout_failures.csv`.

| Validator | Error | Case | Family | Expression | Improved cause | Explanation |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | FN | AGGREGATION-024 | future_aggregation | count(events, window='7d') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-025 | future_aggregation | rolling_mean(transaction_amount, window='24h') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-026 | future_aggregation | count(events, window='7d') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-027 | future_aggregation | rolling_mean(transaction_amount, window='24h') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-028 | future_aggregation | count(events, window='7d') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-029 | future_aggregation | rolling_mean(transaction_amount, window='24h') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-030 | future_aggregation | count(events, window='7d') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-031 | future_aggregation | rolling_mean(transaction_amount, window='24h') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-032 | future_aggregation | count(events, window='7d') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-033 | future_aggregation | rolling_mean(transaction_amount, window='24h') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-034 | future_aggregation | count(events, window='7d') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-035 | future_aggregation | rolling_mean(transaction_amount, window='24h') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-036 | future_aggregation | count(events, window='7d') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-037 | future_aggregation | rolling_mean(transaction_amount, window='24h') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-038 | future_aggregation | count(events, window='7d') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | AGGREGATION-039 | future_aggregation | rolling_mean(transaction_amount, window='24h') | future_aggregation_leakage | valid feature_time hides a future aggregation boundary/source event |
| baseline | FN | TARGET-DERIVED-024 | target_derived | rolling_mean(resolution_code, window='30d') | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-025 | target_derived | latest(outcome_code) | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-026 | target_derived | status_transition_rank(status_code) | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-027 | target_derived | resolution_velocity(ticket_history) | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-028 | target_derived | rolling_mean(resolution_code, window='30d') | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-029 | target_derived | latest(outcome_code) | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-030 | target_derived | status_transition_rank(status_code) | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-031 | target_derived | resolution_velocity(ticket_history) | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-032 | target_derived | rolling_mean(resolution_code, window='30d') | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-033 | target_derived | latest(outcome_code) | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-034 | target_derived | status_transition_rank(status_code) | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-035 | target_derived | resolution_velocity(ticket_history) | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-036 | target_derived | rolling_mean(resolution_code, window='30d') | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-037 | target_derived | latest(outcome_code) | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-038 | target_derived | status_transition_rank(status_code) | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | TARGET-DERIVED-039 | target_derived | resolution_velocity(ticket_history) | target_derived_leakage | baseline does not inspect expression or target lineage |
| baseline | FN | LABEL-SOURCE-024 | label_source | latest(outcome_status) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-025 | label_source | latest(status_code) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-026 | label_source | status_transition_rank(outcome_code) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-027 | label_source | latest(status_code) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-028 | label_source | latest(outcome_status) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-029 | label_source | latest(status_code) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-030 | label_source | status_transition_rank(outcome_code) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-031 | label_source | latest(status_code) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-032 | label_source | latest(outcome_status) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-033 | label_source | latest(status_code) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-034 | label_source | status_transition_rank(outcome_code) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-035 | label_source | latest(status_code) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-036 | label_source | latest(outcome_status) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-037 | label_source | latest(status_code) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-038 | label_source | status_transition_rank(outcome_code) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | LABEL-SOURCE-039 | label_source | latest(status_code) | label_relation_leakage | baseline does not inspect source role or label availability |
| baseline | FN | DUPLICATION-024 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-025 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-026 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-027 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-028 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-029 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-030 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-031 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-032 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-033 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-034 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-035 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-036 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-037 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-038 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
| baseline | FN | DUPLICATION-039 | duplication | customer_activity_profile(events) | duplication_leakage | baseline has no counterpart hash/vector comparison |
