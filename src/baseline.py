"""
src/baseline.py
Point-in-Time Training Data Validator - Baseline Implementations.

1. Naive Baseline (Primary):
   - Strictly applies: feature_time > prediction_time
   - Flags only direct timestamp violations.

2. Ingestion-Aware Baseline (Secondary / Optional comparison):
   - Checks: (feature_time > prediction_time) OR (ingestion_time > prediction_time)
   - Demonstrates that even when pipeline delay / ingestion lag is tracked,
     it still misses complex patterns (aggregation, SCD1 overwrites, target derivation).
"""

from typing import Any, Dict, List, Mapping, Optional
import pandas as pd
from datetime import datetime


def _parse_timestamp(val: Any) -> Optional[pd.Timestamp]:
    """Safely parse input value to a timezone-naive UTC pd.Timestamp."""
    if val is None or pd.isna(val):
        return None
    try:
        ts = pd.to_datetime(val)
        if ts is pd.NaT:
            return None
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts
    except Exception:
        return None


def baseline_validate(row: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Primary Baseline: Strictly checks feature_time > prediction_time.

    Parameters
    ----------
    row : Mapping[str, Any]
        Dictionary or Series containing feature_time/feature_timestamp and
        prediction_time/prediction_timestamp.

    Returns
    -------
    Dict[str, Any]:
        {
            "flagged": bool,
            "root_cause": str,
            "reasons": list[str]
        }
    """
    f_val = row.get("feature_time", row.get("feature_timestamp"))
    p_val = row.get("prediction_time", row.get("prediction_timestamp"))

    f_ts = _parse_timestamp(f_val)
    p_ts = _parse_timestamp(p_val)

    if f_ts is not None and p_ts is not None and f_ts > p_ts:
        return {
            "flagged": True,
            "root_cause": "temporal_leakage",
            "reasons": [
                f"Direct temporal violation: feature_time ({f_ts}) > prediction_time ({p_ts})"
            ]
        }

    return {
        "flagged": False,
        "root_cause": "none",
        "reasons": []
    }


def baseline_ingestion_validate(row: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Secondary Baseline: Checks feature_time > prediction_time OR ingestion_time > prediction_time.
    Catches simple pipeline ingestion delays, but still fails on current-state aggregates.
    """
    f_val = row.get("feature_time", row.get("feature_timestamp"))
    p_val = row.get("prediction_time", row.get("prediction_timestamp"))
    i_val = row.get("ingestion_time", row.get("processing_time", row.get("commit_time")))

    f_ts = _parse_timestamp(f_val)
    p_ts = _parse_timestamp(p_val)
    i_ts = _parse_timestamp(i_val)

    reasons = []
    if f_ts is not None and p_ts is not None and f_ts > p_ts:
        reasons.append(f"feature_time ({f_ts}) > prediction_time ({p_ts})")
    if i_ts is not None and p_ts is not None and i_ts > p_ts:
        reasons.append(f"ingestion_time ({i_ts}) > prediction_time ({p_ts})")

    if reasons:
        return {
            "flagged": True,
            "root_cause": "temporal_leakage" if (f_ts and p_ts and f_ts > p_ts) else "ingestion_delay_leakage",
            "reasons": reasons
        }

    return {
        "flagged": False,
        "root_cause": "none",
        "reasons": []
    }


def baseline_validate_dataframe(df: pd.DataFrame, use_ingestion: bool = False) -> pd.DataFrame:
    """
    Apply baseline validation across an entire DataFrame.
    """
    val_fn = baseline_ingestion_validate if use_ingestion else baseline_validate
    records = [val_fn(row) for _, row in df.iterrows()]
    return pd.DataFrame(records, index=df.index)
