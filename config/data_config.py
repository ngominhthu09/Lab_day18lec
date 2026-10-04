"""Stable configuration for the synthetic point-in-time dataset."""

from __future__ import annotations

SEED = 20261004
TOTAL_CASES = 300
DEV_CASES = 180
HELDOUT_CASES = 120
DEV_RATIO = 0.60

CLEAN_COUNT = 100
LEAKAGE_COUNT_PER_TYPE = 40

LEAKAGE_TYPES = (
    "direct_temporal",
    "future_aggregation",
    "target_derived",
    "label_source",
    "duplication",
)

SEVERITIES = ("low", "medium", "high")

# Keep this list as the contract consumed by downstream benchmark code.
REQUIRED_FIELDS = (
    "case_id",
    "split",
    "ground_truth",
    "is_leakage",
    "leakage_type",
    "severity",
    "entity_id",
    "prediction_time",
    "feature_time",
    "feature_expression",
    "lineage_columns",
    "source_table",
    "source_role",
    "aggregation_window_end",
    "source_max_event_time",
    "label_available_time",
    "feature_vector",
    "other_split_vector",
    "exact_duplicate_hash",
    "other_split_hash",
)
