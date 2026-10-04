"""Improved multi-signal point-in-time leakage validator.

Five independent quality gates, each cheap per row:

  1. temporal            feature_time > prediction_time                                  O(1)
  2. future aggregation  aggregation_window_end or source_max_event_time > prediction_time O(1)
  3. target lineage      controlled target vocabulary in feature_expression / lineage     O(L)
  4. label relation      source_role is a label role, or source_table is a governed table O(1)
  5. duplication         exact hash match with the other split, else cosine >= threshold  O(d)

All timestamps are normalised to UTC before comparison (naive values are interpreted with
`naive_utc_offset_hours`). The validator never reads ground-truth columns: they are dropped
from the row before any check runs.

Root-cause priority (first detected cause wins, all causes are kept in `detected_causes`):
  temporal > future aggregation > target-derived > label relation > duplication
Rationale: row-level evidence of future data is the most direct point-in-time violation;
an aggregate over future events is the same violation hidden inside a window; target-derived
and label-table features leak the label regardless of timing; duplication is split
contamination rather than a temporal defect, and its fix (dedup across splits) is different.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

TEMPORAL = "temporal_leakage"
FUTURE_AGGREGATION = "future_aggregation_leakage"
TARGET_DERIVED = "target_derived_leakage"
LABEL_RELATION = "label_relation_leakage"
DUPLICATION = "duplication_leakage"
NO_LEAKAGE = "none"

ROOT_CAUSE_PRIORITY = (TEMPORAL, FUTURE_AGGREGATION, TARGET_DERIVED, LABEL_RELATION, DUPLICATION)
CHECK_NAMES = ("temporal", "future_aggregation", "target_lineage", "label_relation", "duplication")

# Evaluation-only columns. They are removed from every row before the checks see it.
GROUND_TRUTH_FIELDS = frozenset({"ground_truth", "is_leakage", "leakage_type", "severity"})
MISSING_STRINGS = frozenset({"", "nan", "none", "null", "nat", "na", "n/a"})

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "validator_config.json"


@dataclass
class ColumnMap:
    """Input column names, configurable in case the dataset schema differs."""
    prediction_time: str = "prediction_time"
    feature_time: str = "feature_time"
    aggregation_window_end: str = "aggregation_window_end"
    source_max_event_time: str = "source_max_event_time"
    feature_expression: str = "feature_expression"
    lineage_columns: str = "lineage_columns"
    source_role: str = "source_role"
    source_table: str = "source_table"
    exact_duplicate_hash: str = "exact_duplicate_hash"
    other_split_hash: str = "other_split_hash"
    feature_vector: str = "feature_vector"
    other_split_vector: str = "other_split_vector"


@dataclass
class ValidatorConfig:
    # Tuned on dev only by src/tune_validator.py; 0.995 is the untuned starting value.
    near_duplicate_threshold: float = 0.995
    # Exact identifiers only (after normalisation): "shipping_label" does not match "label".
    target_tokens: list[str] = field(default_factory=lambda: [
        "target", "label", "outcome", "churn_label", "default_label", "y_true", "ground_truth"])
    # Governed label tables; matched on the full name or the last dotted component.
    label_tables: list[str] = field(default_factory=lambda: [
        "labels", "training_labels", "outcomes", "target_events"])
    label_roles: list[str] = field(default_factory=lambda: ["label", "target"])
    temporal_tolerance_seconds: float = 0.0
    naive_utc_offset_hours: float = 0.0
    enabled_checks: list[str] = field(default_factory=lambda: list(CHECK_NAMES))
    columns: ColumnMap = field(default_factory=ColumnMap)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidatorConfig":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "columns"}
        cfg = cls(**known)
        if isinstance(data.get("columns"), Mapping):
            cfg.columns = ColumnMap(**{k: v for k, v in data["columns"].items()
                                       if k in ColumnMap.__dataclass_fields__})
        return cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "ValidatorConfig":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CheckResult:
    cause: str | None = None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- parsing helpers

def is_missing(value: Any) -> bool:
    """None, NaN, NaT, empty string and common null spellings."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_STRINGS
    try:
        return bool(value != value)        # NaN / NaT
    except (TypeError, ValueError):
        return False


_NUMERIC = re.compile(r"[+-]?\d+(\.\d+)?")
_FALLBACK_FORMATS = ("%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d",
                     "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y")


def parse_timestamp(value: Any, naive_tz: timezone = timezone.utc) -> datetime | None:
    """Parse ISO-8601 strings, datetimes, pandas Timestamps or epoch seconds/milliseconds into an
    aware UTC datetime. Returns None for missing values; raises ValueError if unparseable or out of
    the representable range (e.g. inf, year 1 with a positive offset, epochs the platform rejects)."""
    if is_missing(value):
        return None
    try:
        return _parse_timestamp(value, naive_tz)
    except (OverflowError, OSError):
        raise ValueError(f"timestamp {value!r} out of range") from None


def _parse_timestamp(value: Any, naive_tz: timezone) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        dt = _from_epoch(float(value))
    else:
        text = str(value).strip()
        if _NUMERIC.fullmatch(text):
            dt = _from_epoch(float(text))
        else:
            if text[-1:] in ("Z", "z"):
                text = text[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                for fmt in _FALLBACK_FORMATS:
                    try:
                        dt = datetime.strptime(text, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"unparseable timestamp {value!r}") from None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=naive_tz)
    return dt.astimezone(timezone.utc)


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _from_epoch(x: float) -> datetime:
    if abs(x) > 1e11:                      # milliseconds
        x /= 1000.0
    # EPOCH + timedelta instead of fromtimestamp: same result on every OS (Windows rejects
    # negative epochs in fromtimestamp); out-of-range values raise OverflowError
    return _EPOCH + timedelta(seconds=x)


def parse_vector(value: Any) -> list[float] | None:
    """Parse a JSON list (or a bracketed / comma / space separated list) of finite numbers.
    Returns None for missing values; raises ValueError if malformed."""
    if isinstance(value, (list, tuple)):
        items = list(value)
    elif hasattr(value, "tolist"):
        items = value.tolist()
    elif is_missing(value):
        return None
    else:
        text = str(value).strip()
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            items = [t for t in re.split(r"[\s,;]+", text.strip("[]() ")) if t]
    if not isinstance(items, list) or not items:
        raise ValueError("vector is not a non-empty list")
    try:
        out = [float(x) for x in items]
    except (TypeError, ValueError):
        raise ValueError("vector contains non-numeric entries") from None
    if not all(math.isfinite(x) for x in out):
        raise ValueError("vector contains NaN/inf")
    return out


def cosine_similarity(a: list[float], b: list[float]) -> float | None:
    """None when either vector has zero norm. Caller guarantees equal length.
    Each vector is first divided by its largest magnitude (cosine is scale-invariant), so values
    near 1e308 or subnormals cannot overflow / underflow the norms into inf or 0."""
    sa = max((abs(x) for x in a), default=0.0)
    sb = max((abs(x) for x in b), default=0.0)
    if sa == 0.0 or sb == 0.0:
        return None
    a = [x / sa for x in a]
    b = [x / sb for x in b]
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (na * nb)))


def normalize_identifier(name: str) -> str:
    """'churnLabel' / 'CHURN_LABEL' / '`churn-label`' -> 'churn_label'."""
    name = name.strip().strip('`"[]')
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")


_SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)
_STRING_LITERAL = re.compile(r"'(?:[^'\\]|\\.|'')*'")
_DOTTED_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\s*\.\s*[A-Za-z_][A-Za-z0-9_]*)*")


def column_identifiers(text: str) -> list[str]:
    """Column-level identifiers in an expression. String literals and SQL comments are removed,
    and for qualified names ('labels.customer_id') only the column part is kept, so a table
    qualifier never counts as a column."""
    text = _STRING_LITERAL.sub(" ", _SQL_COMMENT.sub(" ", text))
    return [normalize_identifier(m.group(0).split(".")[-1]) for m in _DOTTED_IDENT.finditer(text)]


def parse_lineage(value: Any) -> list[str]:
    """Lineage as a JSON list, a Python list, or a ',', ';', '|' or newline separated string."""
    if isinstance(value, (list, tuple)):
        items = list(value)
    elif is_missing(value):
        return []
    else:
        text = str(value).strip()
        items = None
        if text.startswith("["):
            try:
                items = json.loads(text)
            except json.JSONDecodeError:
                items = None
        if not isinstance(items, list):
            items = re.split(r"[,;|\n]+", text.strip("[]"))
    out = []
    for item in items:
        if isinstance(item, Mapping):          # e.g. {"table": "...", "column": "..."}
            item = item.get("column") or item.get("name") or ""
        item = str(item).strip().strip("'\"")
        if item:
            out.append(item)
    return out


def _table_names(raw: str) -> set[str]:
    name = raw.strip().strip('`"[]').lower()
    return {name, name.split(".")[-1].strip('`"[]')}


def _fmt_ts(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _fmt_lag(delta: timedelta) -> str:
    s = delta.total_seconds()
    return f"{s / 3600:.2f} h" if abs(s) >= 3600 else f"{s:.0f} s"


# --------------------------------------------------------------------------- validator

class ImprovedValidator:
    """Multi-signal point-in-time leakage validator.

    validate(row) accepts a dict, a pandas Series or a namedtuple and returns
        {"flagged", "root_cause", "reasons", "detected_causes", "scores", "warnings", "debug_metadata"}.
    """

    def __init__(self, config: ValidatorConfig | Mapping[str, Any] | str | Path | None = None):
        if config is None:
            if DEFAULT_CONFIG_PATH.exists():
                config, source = ValidatorConfig.from_json(DEFAULT_CONFIG_PATH), str(DEFAULT_CONFIG_PATH)
            else:
                config, source = ValidatorConfig(), "built-in defaults"
        elif isinstance(config, (str, Path)):
            config, source = ValidatorConfig.from_json(config), str(config)
        elif isinstance(config, Mapping):
            config, source = ValidatorConfig.from_dict(config), "dict"
        else:
            source = "ValidatorConfig"
        self.config: ValidatorConfig = config
        self.config_source = source
        self._check_config()
        self._cols = config.columns
        self._target_tokens = frozenset(normalize_identifier(t) for t in config.target_tokens)
        self._label_tables = frozenset(t.strip().lower() for t in config.label_tables)
        self._label_roles = frozenset(r.strip().lower() for r in config.label_roles)
        self._naive_tz = timezone(timedelta(hours=config.naive_utc_offset_hours))
        self._tolerance = timedelta(seconds=config.temporal_tolerance_seconds)
        registry: dict[str, Callable[[Mapping[str, Any]], CheckResult]] = {
            "temporal": self._check_temporal,
            "future_aggregation": self._check_future_aggregation,
            "target_lineage": self._check_target_lineage,
            "label_relation": self._check_label_relation,
            "duplication": self._check_duplication,
        }
        self._checks = [(name, registry[name]) for name in CHECK_NAMES if name in config.enabled_checks]

    def _check_config(self) -> None:
        cfg = self.config
        if not -1.0 <= cfg.near_duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be in [-1, 1]")
        unknown = set(cfg.enabled_checks) - set(CHECK_NAMES)
        if unknown:
            raise ValueError(f"unknown checks in enabled_checks: {sorted(unknown)}")
        leaked = [c for c in asdict(cfg.columns).values() if c.lower() in GROUND_TRUTH_FIELDS]
        if leaked:
            raise ValueError(f"column map points at ground-truth fields: {leaked}")

    # ---- public API

    def validate(self, row: Any) -> dict[str, Any]:
        view, ignored = self._public_view(row)
        detected: list[str] = []
        reasons_by_cause: dict[str, list[str]] = {}
        scores: dict[str, Any] = {}
        warnings: list[str] = []
        failed: list[str] = []
        for name, check in self._checks:
            try:
                res = check(view)
            except Exception as exc:           # one malformed field must not stop a whole batch
                logger.warning("%s: %s check failed", view.get("case_id", "?"), name, exc_info=True)
                failed.append(name)
                warnings.append(f"{name} check failed ({type(exc).__name__}: {exc}); treated as not flagged")
                continue
            scores.update(res.scores)
            warnings.extend(res.warnings)
            if res.cause:
                detected.append(res.cause)
                reasons_by_cause.setdefault(res.cause, []).extend(res.reasons)
        detected = [c for c in ROOT_CAUSE_PRIORITY if c in detected]
        warnings = list(dict.fromkeys(warnings))   # gates 1 and 2 both parse prediction_time
        for w in warnings:
            logger.debug("%s: %s", view.get("case_id", "?"), w)
        return {
            "flagged": bool(detected),
            "root_cause": detected[0] if detected else NO_LEAKAGE,
            "reasons": [r for c in detected for r in reasons_by_cause[c]],
            "detected_causes": detected,
            "scores": scores,
            "warnings": warnings,
            "debug_metadata": {"ignored_fields": ignored, "checks_run": [n for n, _ in self._checks],
                               "checks_failed": failed, "config_source": self.config_source},
        }

    def validate_many(self, rows: Iterable[Any]) -> list[dict[str, Any]]:
        return [self.validate(r) for r in rows]

    @staticmethod
    def _public_view(row: Any) -> tuple[dict[str, Any], list[str]]:
        """Copy of the row without ground-truth fields. The checks only ever see this copy."""
        if hasattr(row, "_asdict"):
            row = row._asdict()
        if not hasattr(row, "items"):
            raise TypeError(f"row must be a mapping-like object, got {type(row).__name__}")
        view, ignored = {}, []
        for k, v in row.items():
            if str(k).strip().lower() in GROUND_TRUTH_FIELDS:
                ignored.append(str(k))
            else:
                view[k] = v
        return view, ignored

    # ---- helpers

    def _timestamp(self, view: Mapping[str, Any], column: str, res: CheckResult) -> datetime | None:
        try:
            return parse_timestamp(view.get(column), self._naive_tz)
        except ValueError as exc:
            res.warnings.append(f"{column}: {exc}; treated as missing")
            return None

    # ---- gate 1

    def _check_temporal(self, view: Mapping[str, Any]) -> CheckResult:
        res = CheckResult()
        c = self._cols
        pred = self._timestamp(view, c.prediction_time, res)
        feat = self._timestamp(view, c.feature_time, res)
        if pred is None or feat is None:
            missing = [n for n, v in ((c.prediction_time, pred), (c.feature_time, feat)) if v is None]
            res.warnings.append(f"temporal check skipped: {', '.join(missing)} missing")
            return res
        lag = feat - pred
        res.scores["temporal_lag_seconds"] = lag.total_seconds()
        if lag > self._tolerance:
            res.cause = TEMPORAL
            res.reasons.append(f"feature_time {_fmt_ts(feat)} exceeds prediction_time {_fmt_ts(pred)} by {_fmt_lag(lag)}")
        return res

    # ---- gate 2

    def _check_future_aggregation(self, view: Mapping[str, Any]) -> CheckResult:
        res = CheckResult()
        c = self._cols
        pred = self._timestamp(view, c.prediction_time, res)
        bounds = [(c.aggregation_window_end, self._timestamp(view, c.aggregation_window_end, res)),
                  (c.source_max_event_time, self._timestamp(view, c.source_max_event_time, res))]
        if all(ts is None for _, ts in bounds):
            return res                         # not an aggregate feature, nothing to check
        if pred is None:
            res.warnings.append("future-aggregation check skipped: prediction_time missing")
            return res
        for column, ts in bounds:
            if ts is None:
                continue
            lag = ts - pred
            res.scores[f"{column}_lag_seconds"] = lag.total_seconds()
            if lag > self._tolerance:
                res.cause = FUTURE_AGGREGATION
                res.reasons.append(f"{column} {_fmt_ts(ts)} exceeds prediction_time {_fmt_ts(pred)} by {_fmt_lag(lag)}")
        return res

    # ---- gate 3

    def _check_target_lineage(self, view: Mapping[str, Any]) -> CheckResult:
        res = CheckResult()
        c = self._cols
        sources: list[tuple[str, list[str]]] = []
        expr = view.get(c.feature_expression)
        if not is_missing(expr):
            sources.append((c.feature_expression, column_identifiers(str(expr))))
        lineage = parse_lineage(view.get(c.lineage_columns))
        if lineage:
            sources.append((c.lineage_columns, [tok for item in lineage for tok in column_identifiers(item)]))
        if not sources:
            res.warnings.append("target-lineage check skipped: no feature_expression or lineage_columns")
            return res
        found: list[str] = []
        for where, tokens in sources:
            for tok in tokens:
                if tok in self._target_tokens:
                    msg = f"target identifier '{tok}' found in {where}"
                    if msg not in res.reasons:
                        res.reasons.append(msg)
                        found.append(tok)
        res.scores["target_tokens_found"] = sorted(set(found))
        if found:
            res.cause = TARGET_DERIVED
        return res

    # ---- gate 4

    def _check_label_relation(self, view: Mapping[str, Any]) -> CheckResult:
        res = CheckResult()
        c = self._cols
        role, table = view.get(c.source_role), view.get(c.source_table)
        if is_missing(role) and is_missing(table):
            res.warnings.append("label-relation check skipped: no source_role or source_table")
            return res
        if not is_missing(role) and str(role).strip().lower() in self._label_roles:
            res.reasons.append(f"source_role='{str(role).strip()}'")
        if not is_missing(table) and _table_names(str(table)) & self._label_tables:
            res.reasons.append(f"source_table '{str(table).strip()}' is a governed label table")
        if res.reasons:
            res.cause = LABEL_RELATION
        return res

    # ---- gate 5

    def _check_duplication(self, view: Mapping[str, Any]) -> CheckResult:
        res = CheckResult()
        c = self._cols
        h1, h2 = view.get(c.exact_duplicate_hash), view.get(c.other_split_hash)
        hashes_present = not is_missing(h1) and not is_missing(h2)
        if hashes_present and str(h1).strip().lower() == str(h2).strip().lower():
            res.cause = DUPLICATION
            res.scores["duplicate_kind"] = "exact"
            res.reasons.append(f"exact duplicate: exact_duplicate_hash equals other_split_hash ({str(h1).strip()[:12]})")
            return res
        vecs = []
        for column in (c.feature_vector, c.other_split_vector):
            try:
                vecs.append(parse_vector(view.get(column)))
            except ValueError as exc:
                res.warnings.append(f"{column}: {exc}; near-duplicate check skipped")
                return res
        a, b = vecs
        if a is None or b is None:
            if not hashes_present:             # neither signal available: say so instead of passing silently
                res.warnings.append("duplication check skipped: no hash pair and no vector pair")
            return res                         # hashes compared and differ; no counterpart vector
        if len(a) != len(b):
            res.warnings.append(f"near-duplicate check skipped: vector dims differ ({len(a)} vs {len(b)})")
            return res
        sim = cosine_similarity(a, b)
        if sim is None:
            res.warnings.append("near-duplicate check skipped: zero-norm vector")
            return res
        res.scores["cosine_similarity"] = sim
        if sim >= self.config.near_duplicate_threshold:
            res.cause = DUPLICATION
            res.scores["duplicate_kind"] = "near"
            res.reasons.append(f"near duplicate: cosine similarity {sim:.5f} with the other split "
                               f">= threshold {self.config.near_duplicate_threshold}")
        return res
