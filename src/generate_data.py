"""Generate deterministic synthetic point-in-time training cases.

The module deliberately creates data only.  It does not contain a baseline or
improved validator.  Ground-truth fields are metadata for benchmark scoring;
the feature metadata is varied enough to include realistic ambiguous cases and
to avoid making every family identifiable through one trivial string rule.

Run from the repository root with::

    python src/generate_data.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.data_config import (  # noqa: E402
    CLEAN_COUNT,
    DEV_CASES,
    DEV_RATIO,
    HELDOUT_CASES,
    LEAKAGE_COUNT_PER_TYPE,
    LEAKAGE_TYPES,
    REQUIRED_FIELDS,
    SEED,
    SEVERITIES,
    TOTAL_CASES,
)


UTC = timezone.utc
BASE_TIME = datetime(2026, 1, 5, 8, 0, tzinfo=UTC)
CSV_FIELDS = list(REQUIRED_FIELDS)


@dataclass
class Case:
    """In-memory representation of one benchmark record."""

    case_id: str
    split: str
    ground_truth: str
    is_leakage: bool
    leakage_type: str
    severity: str
    entity_id: str
    prediction_time: str
    feature_time: str
    feature_expression: str
    lineage_columns: list[str]
    source_table: str
    source_role: str
    aggregation_window_end: str
    source_max_event_time: str
    label_available_time: str
    feature_vector: list[float]
    other_split_vector: list[float]
    exact_duplicate_hash: str
    other_split_hash: str

    def as_csv_row(self) -> dict[str, str]:
        """Serialize nested values consistently for CSV output."""

        row = {
            field: getattr(self, field)
            for field in CSV_FIELDS
        }
        row["is_leakage"] = str(self.is_leakage).lower()
        for field in ("lineage_columns", "feature_vector", "other_split_vector"):
            row[field] = json.dumps(row[field], ensure_ascii=False, separators=(",", ":"))
        return {field: str(row[field]) for field in CSV_FIELDS}


def iso(value: datetime) -> str:
    """Return a stable UTC timestamp representation."""

    return value.astimezone(UTC).isoformat(timespec="minutes").replace("+00:00", "Z")


def vector_hash(vector: Sequence[float]) -> str:
    """Hash a vector using a canonical JSON representation."""

    payload = json.dumps(list(vector), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rounded_vector(values: Iterable[float]) -> list[float]:
    """Keep CSV values compact while retaining near-duplicate differences."""

    return [round(value, 6) for value in values]


def make_vector(rng: random.Random, size: int = 8) -> list[float]:
    """Create a varied, deterministic feature vector."""

    return rounded_vector(rng.uniform(-2.5, 2.5) for _ in range(size))


def severity_for(index: int) -> str:
    """Cycle severities so each split contains all severity levels."""

    return SEVERITIES[index % len(SEVERITIES)]


def time_context(index: int) -> tuple[datetime, datetime]:
    """Create deterministic, distinct prediction and baseline feature times."""

    prediction = BASE_TIME + timedelta(hours=(index * 7) % (24 * 28), minutes=(index * 11) % 60)
    available_feature = prediction - timedelta(minutes=15 + (index % 6) * 10)
    return prediction, available_feature


def base_kwargs(
    *,
    case_id: str,
    split: str,
    leakage_type: str,
    severity: str,
    entity_id: str,
    prediction: datetime,
    feature_time: datetime,
    feature_expression: str,
    lineage_columns: list[str],
    source_table: str,
    source_role: str,
    source_max_event_time: datetime,
    label_available_time: datetime,
    feature_vector: list[float],
    other_split_vector: list[float] | None = None,
    exact_duplicate_hash: str = "",
    other_split_hash: str = "",
) -> Case:
    """Build a case and fill common point-in-time fields."""

    return Case(
        case_id=case_id,
        split=split,
        ground_truth="clean" if leakage_type == "clean" else "leakage",
        is_leakage=leakage_type != "clean",
        leakage_type=leakage_type,
        severity=severity,
        entity_id=entity_id,
        prediction_time=iso(prediction),
        feature_time=iso(feature_time),
        feature_expression=feature_expression,
        lineage_columns=lineage_columns,
        source_table=source_table,
        source_role=source_role,
        aggregation_window_end=iso(feature_time),
        source_max_event_time=iso(source_max_event_time),
        label_available_time=iso(label_available_time),
        feature_vector=feature_vector,
        other_split_vector=other_split_vector or [],
        exact_duplicate_hash=exact_duplicate_hash,
        other_split_hash=other_split_hash,
    )


def make_clean_case(index: int, split: str, rng: random.Random) -> Case:
    """Create clean cases, including lexical traps for naive rules."""

    case_id = f"CLEAN-{index:03d}"
    prediction, feature_time = time_context(index)
    expression_templates = (
        (
            "shipping_label_length",
            ["shipping_label", "package_weight"],
        ),
        (
            "target_temperature_delta",
            ["target_temperature", "ambient_temperature"],
        ),
        (
            "package_label_print_delay",
            ["label_print_date", "scan_time"],
        ),
        (
            "customer_resolution_time_estimate",
            ["ticket_open_time", "agent_queue"],
        ),
    )
    expression, lineage = expression_templates[index % len(expression_templates)]
    return base_kwargs(
        case_id=case_id,
        split=split,
        leakage_type="clean",
        severity="none",
        entity_id=f"entity-{index:04d}",
        prediction=prediction,
        feature_time=feature_time,
        feature_expression=expression,
        lineage_columns=lineage,
        source_table=("shipping_events" if index % 2 == 0 else "customer_facts"),
        source_role="feature",
        source_max_event_time=feature_time,
        label_available_time=prediction + timedelta(hours=6),
        feature_vector=make_vector(rng),
    )


def make_temporal_case(index: int, split: str, severity: str, rng: random.Random) -> Case:
    """Create direct temporal leakage with a severity-dependent time gap."""

    case_id = f"TEMPORAL-{index:03d}"
    prediction, available_feature = time_context(100 + index)
    offset = {"low": 5, "medium": 60, "high": 1440}[severity]
    future_feature = prediction + timedelta(minutes=offset)
    return base_kwargs(
        case_id=case_id,
        split=split,
        leakage_type="direct_temporal",
        severity=severity,
        entity_id=f"entity-{1000 + index:04d}",
        prediction=prediction,
        feature_time=future_feature,
        feature_expression=("latest(account_balance)" if index % 2 else "event_delta(login_count)"),
        lineage_columns=["account_balance", "event_time"] if index % 2 else ["login_count", "event_time"],
        source_table="account_events",
        source_role="feature",
        source_max_event_time=future_feature,
        label_available_time=prediction + timedelta(hours=6),
        feature_vector=make_vector(rng),
    )


def make_aggregation_case(index: int, split: str, severity: str, rng: random.Random) -> Case:
    """Create future-window aggregation where the point feature itself is old enough."""

    case_id = f"AGGREGATION-{index:03d}"
    prediction, feature_time = time_context(140 + index)
    horizon = {"low": 30, "medium": 180, "high": 1440}[severity]
    window_end = prediction + timedelta(minutes=horizon)
    source_max = prediction + timedelta(minutes=max(10, horizon - 10))
    case = base_kwargs(
        case_id=case_id,
        split=split,
        leakage_type="future_aggregation",
        severity=severity,
        entity_id=f"entity-{2000 + index:04d}",
        prediction=prediction,
        feature_time=feature_time,
        feature_expression=(
            "rolling_mean(transaction_amount, window='24h')"
            if index % 2
            else "count(events, window='7d')"
        ),
        lineage_columns=["transaction_amount", "event_time"] if index % 2 else ["event_id", "event_time"],
        source_table="transactions_v2",
        source_role="feature",
        source_max_event_time=source_max,
        label_available_time=prediction + timedelta(hours=6),
        feature_vector=make_vector(rng),
    )
    case.aggregation_window_end = iso(window_end)
    return case


def make_target_case(index: int, split: str, severity: str, rng: random.Random) -> Case:
    """Create target-derived leakage with aliases that avoid obvious target words."""

    case_id = f"TARGET-DERIVED-{index:03d}"
    prediction, feature_time = time_context(180 + index)
    variants = (
        ("rolling_mean(resolution_code, window='30d')", ["resolution_code", "account_age_days"]),
        ("latest(outcome_code)", ["outcome_code", "case_open_time"]),
        ("status_transition_rank(status_code)", ["status_code", "event_time"]),
        ("resolution_velocity(ticket_history)", ["ticket_history", "resolution_code"]),
    )
    expression, lineage = variants[index % len(variants)]
    return base_kwargs(
        case_id=case_id,
        split=split,
        leakage_type="target_derived",
        severity=severity,
        entity_id=f"entity-{3000 + index:04d}",
        prediction=prediction,
        feature_time=feature_time,
        feature_expression=expression,
        lineage_columns=lineage,
        source_table=("support_tickets" if index % 2 else "claims_v2"),
        source_role="feature",
        source_max_event_time=feature_time,
        label_available_time=prediction + timedelta(hours=6),
        feature_vector=make_vector(rng),
    )


def make_label_source_case(index: int, split: str, severity: str, rng: random.Random) -> Case:
    """Create label-source leakage, mixing obvious and deliberately subtle forms."""

    case_id = f"LABEL-SOURCE-{index:03d}"
    prediction, feature_time = time_context(220 + index)
    obvious = index % 4 == 0
    if obvious:
        source_table, source_role, expression, lineage = (
            "labels",
            "label",
            "latest(outcome_status)",
            ["outcome_status", "entity_id"],
        )
    else:
        source_table, source_role, expression, lineage = (
            "facts_v2",
            "feature",
            ("latest(status_code)" if index % 2 else "status_transition_rank(outcome_code)"),
            (["status_code", "event_time"] if index % 2 else ["outcome_code", "event_time"]),
        )
    return base_kwargs(
        case_id=case_id,
        split=split,
        leakage_type="label_source",
        severity=severity,
        entity_id=f"entity-{4000 + index:04d}",
        prediction=prediction,
        feature_time=feature_time,
        feature_expression=expression,
        lineage_columns=lineage,
        source_table=source_table,
        source_role=source_role,
        source_max_event_time=feature_time,
        label_available_time=prediction - timedelta(minutes=5),
        feature_vector=make_vector(rng),
    )


def make_duplication_case(index: int, split: str, severity: str, rng: random.Random) -> Case:
    """Create cross-split exact or near-duplicate vectors."""

    case_id = f"DUPLICATION-{index:03d}"
    prediction, feature_time = time_context(260 + index)
    feature_vector = make_vector(rng)
    if severity == "high":
        other_vector = list(feature_vector)
    elif severity == "medium":
        other_vector = rounded_vector(value + rng.uniform(-0.0008, 0.0008) for value in feature_vector)
    else:
        other_vector = rounded_vector(value + rng.uniform(-0.08, 0.08) for value in feature_vector)
    other_hash = vector_hash(other_vector)
    return base_kwargs(
        case_id=case_id,
        split=split,
        leakage_type="duplication",
        severity=severity,
        entity_id=f"entity-{5000 + index:04d}",
        prediction=prediction,
        feature_time=feature_time,
        feature_expression="customer_activity_profile(events)",
        lineage_columns=["event_count", "session_duration", "entity_id"],
        source_table="feature_snapshots",
        source_role="feature",
        source_max_event_time=feature_time,
        label_available_time=prediction + timedelta(hours=6),
        feature_vector=feature_vector,
        other_split_vector=other_vector,
        exact_duplicate_hash=(vector_hash(feature_vector) if severity == "high" else ""),
        other_split_hash=other_hash,
    )


def generate_cases(seed: int = SEED) -> list[Case]:
    """Generate all 300 cases in a deterministic, stratified order."""

    rng = random.Random(seed)
    cases: list[Case] = []

    # Clean is the only class whose size is not 40. It is still split by class.
    clean_cases = [make_clean_case(i, "dev" if i < 60 else "heldout", rng) for i in range(CLEAN_COUNT)]
    cases.extend(clean_cases)

    builders = {
        "direct_temporal": make_temporal_case,
        "future_aggregation": make_aggregation_case,
        "target_derived": make_target_case,
        "label_source": make_label_source_case,
        "duplication": make_duplication_case,
    }
    for leakage_type in LEAKAGE_TYPES:
        builder = builders[leakage_type]
        for index in range(LEAKAGE_COUNT_PER_TYPE):
            split = "dev" if index < int(LEAKAGE_COUNT_PER_TYPE * DEV_RATIO) else "heldout"
            case = builder(index, split, severity_for(index), rng)
            cases.append(case)

    validate_cases(cases)
    return cases


def validate_cases(cases: Sequence[Case]) -> None:
    """Assert dataset integrity before any CSV is written."""

    assert len(cases) == TOTAL_CASES, len(cases)
    assert sum(not case.is_leakage for case in cases) == CLEAN_COUNT
    assert sum(case.is_leakage for case in cases) == TOTAL_CASES - CLEAN_COUNT
    assert len({case.case_id for case in cases}) == TOTAL_CASES
    assert all(case.split in {"dev", "heldout"} for case in cases)
    assert sum(case.split == "dev" for case in cases) == DEV_CASES
    assert sum(case.split == "heldout" for case in cases) == HELDOUT_CASES

    for case in cases:
        row = case.as_csv_row()
        # Empty strings are intentional for non-duplication hash fields; the
        # contract is that every required column exists in every record.
        assert all(field in row for field in REQUIRED_FIELDS), f"missing field in {case.case_id}"
        assert all(row[field] is not None for field in REQUIRED_FIELDS), f"null field in {case.case_id}"
        assert case.is_leakage == (case.ground_truth == "leakage")

    for leakage_type in LEAKAGE_TYPES:
        typed = [case for case in cases if case.leakage_type == leakage_type]
        assert len(typed) == LEAKAGE_COUNT_PER_TYPE
        assert {case.severity for case in typed} == set(SEVERITIES)
        assert sum(case.split == "dev" for case in typed) == 24
        assert sum(case.split == "heldout" for case in typed) == 16
        assert {case.severity for case in typed if case.split == "dev"} == set(SEVERITIES)
        assert {case.severity for case in typed if case.split == "heldout"} == set(SEVERITIES)

    assert sum(case.split == "dev" and not case.is_leakage for case in cases) == 60
    assert sum(case.split == "heldout" and not case.is_leakage for case in cases) == 40

    duplication_cases = [case for case in cases if case.leakage_type == "duplication"]
    assert all(case.other_split_vector for case in duplication_cases)
    assert all(case.other_split_hash == vector_hash(case.other_split_vector) for case in duplication_cases)
    assert all(
        case.exact_duplicate_hash == vector_hash(case.feature_vector)
        for case in duplication_cases
        if case.severity == "high"
    )
    assert all(case.exact_duplicate_hash == "" for case in duplication_cases if case.severity != "high")


def write_csv(path: Path, cases: Sequence[Case]) -> None:
    """Write cases with stable column order and UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(case.as_csv_row() for case in cases)


def write_outputs(cases: Sequence[Case], output_dir: Path = ROOT / "data") -> None:
    """Write all cases and the fixed Dev/Held-out partitions."""

    write_csv(output_dir / "all_cases.csv", cases)
    write_csv(output_dir / "dev_cases.csv", [case for case in cases if case.split == "dev"])
    write_csv(output_dir / "heldout_cases.csv", [case for case in cases if case.split == "heldout"])


def main() -> None:
    """Generate and validate the benchmark dataset."""

    cases = generate_cases()
    write_outputs(cases)
    print(
        f"Generated {len(cases)} cases with seed {SEED}: "
        f"{sum(case.split == 'dev' for case in cases)} dev, "
        f"{sum(case.split == 'heldout' for case in cases)} heldout."
    )


if __name__ == "__main__":
    main()
