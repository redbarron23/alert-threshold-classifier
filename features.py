"""Input contract shared by training and shadow scoring."""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "p99_latency_ms", "error_rate_pct", "pod_restarts", "cpu_throttle_pct",
    "mem_working_set_pct", "ready_ratio", "p99_latency_ms_15m",
    "error_rate_pct_15m", "pod_restarts_15m",
]
STEP_MINUTES = 5


def validate_data(df: pd.DataFrame, *, labeled: bool = False) -> pd.DataFrame:
    """Normalize timestamps to UTC; reject incomplete or irregular telemetry.

    Extra columns are ignored by the model, never automatically made features.
    Naive timestamps are interpreted as UTC. Rolling features must be causal.
    """
    required = ["ts", *FEATURE_COLUMNS, *(["incident"] if labeled else [])]
    missing = sorted(set(required) - set(df.columns))
    if missing or not df.columns.is_unique:
        raise ValueError(f"missing columns {missing} or duplicate column names")
    if df.empty:
        raise ValueError("telemetry must contain at least one window")
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in FEATURE_COLUMNS):
        raise ValueError("features must be numeric")
    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("features must be finite, with no missing values")
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="raise")
    if df["ts"].isna().any() or not df["ts"].is_monotonic_increasing or df["ts"].duplicated().any():
        raise ValueError("timestamps must be present, unique and increasing")
    if not df["ts"].diff().iloc[1:].eq(pd.Timedelta(minutes=STEP_MINUTES)).all():
        raise ValueError("telemetry must have contiguous 5-minute windows")
    if labeled and not df["incident"].isin([0, 1]).all():
        raise ValueError("incident labels must be 0 or 1, with no missing values")
    return df
