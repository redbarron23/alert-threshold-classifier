"""Score trusted local model artifacts in shadow mode; never send notifications."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
from pathlib import Path
import pickle
import platform

import pandas as pd

from evaluate import PENDING_WINDOWS, TEST_DATE, VAL_DATE, Results, alert_policy
from features import FEATURE_COLUMNS, STEP_MINUTES, validate_data


def runtime_versions() -> dict:
    return {"python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")}}


def save_model(r: Results, seed: int, target_precision: float, directory: Path) -> Path:
    """Save a uniquely named bundle, preserving earlier runs for rollback."""
    directory.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc)
    source_hash = hashlib.sha256()
    for name in ("simulate.py", "features.py", "evaluate.py", "main.py", "score.py"):
        source_hash.update(name.encode() + Path(__file__).with_name(name).read_bytes())
    artifact = {
        "model": r.model, "threshold": r.threshold, "features": FEATURE_COLUMNS.copy(),
        "pending_windows": PENDING_WINDOWS, "step_minutes": STEP_MINUTES,
        "metadata": {"created_utc": created.isoformat(), "seed": seed,
                     "target_precision": target_precision, "validation_start": VAL_DATE,
                     "test_start": TEST_DATE, "versions": runtime_versions(),
                     "data_sha256": r.data_sha256, "source_sha256": source_hash.hexdigest()},
    }
    path = directory / f"model-{created.strftime('%Y%m%dT%H%M%S%fZ')}-seed{seed}.pkl"
    with path.open("xb") as f:
        pickle.dump(artifact, f)
    return path


def load_model(path: Path) -> dict:
    """Only load artifacts you trust: pickle files can execute Python code."""
    with path.open("rb") as f:
        artifact = pickle.load(f)
    if artifact["metadata"]["versions"] != runtime_versions():
        raise ValueError("model runtime versions differ; use the training environment")
    if (artifact["features"] != FEATURE_COLUMNS
            or artifact["pending_windows"] != PENDING_WINDOWS
            or artifact["step_minutes"] != STEP_MINUTES):
        raise ValueError("model feature or alert-policy contract differs from this scorer")
    return artifact


def shadow_score(artifact: dict, df: pd.DataFrame, as_of: str | None = None) -> pd.DataFrame:
    """Reject stale/future telemetry. Each batch starts with no pending alert state."""
    df = validate_data(df)
    now = pd.to_datetime(as_of, utc=True) if as_of is not None else pd.Timestamp.now(tz="UTC")
    age = now - df["ts"].iloc[-1]
    if pd.isna(now) or age < pd.Timedelta(0) or age > pd.Timedelta(minutes=STEP_MINUTES):
        raise ValueError("latest telemetry must be no more than 5 minutes old and not in the future")
    scores = artifact["model"].decision_function(df[artifact["features"]])
    return pd.DataFrame({"ts": df["ts"], "decision_score": scores,
                         "alert": alert_policy(scores >= artifact["threshold"])})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path, help="trusted .pkl artifact produced by main.py")
    ap.add_argument("input", type=Path, help="CSV with ts and all nine prepared features")
    ap.add_argument("output", type=Path, help="new CSV path for shadow predictions")
    ap.add_argument("--as-of", help="UTC reference time for historical replay; defaults to now")
    args = ap.parse_args()
    try:
        result = shadow_score(load_model(args.model), pd.read_csv(args.input), args.as_of)
        result.to_csv(args.output, index=False, mode="x")
    except (ValueError, OSError, KeyError, EOFError, pickle.UnpicklingError) as e:
        raise SystemExit(f"error: {e}")
    print(f"Wrote {len(result)} shadow predictions to {args.output}; no notifications sent")


if __name__ == "__main__":
    main()
