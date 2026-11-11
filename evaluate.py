"""Train the alert classifier and score it per window and per incident."""
from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS, STEP_MINUTES, validate_data

# Time-based split: train on days 1-14, pick the threshold on days 15-20,
# report results on days 21-30. The test set never influences any choice.
VAL_DATE = "2026-10-15"
TEST_DATE = "2026-10-21"
PENDING_WINDOWS = 2


@dataclass
class Results:
    y: np.ndarray            # true labels for test windows
    scores: np.ndarray       # classifier decision scores for test windows
    rule: np.ndarray         # baseline rule predictions for test windows
    precisions: np.ndarray
    recalls: np.ndarray
    thresholds: np.ndarray   # test-set curve, for the figures
    threshold: float         # chosen on the validation set
    model: Pipeline
    tuned_rule: np.ndarray | None
    tuned_rule_threshold: float | None
    data_sha256: str


def train_and_score(df: pd.DataFrame, target_precision: float = 0.90) -> Results:
    df = validate_data(df, labeled=True)
    train = df[df["ts"] < VAL_DATE]
    val = df[(df["ts"] >= VAL_DATE) & (df["ts"] < TEST_DATE)]
    test = df[df["ts"] >= TEST_DATE]
    if any(part.empty for part in (train, val, test)):
        raise ValueError("train, validation and test splits must all be nonempty")
    if train["incident"].nunique() != 2:
        raise ValueError("training data must contain both incident classes")

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(train[FEATURE_COLUMNS], train["incident"])

    # Pick the threshold on validation data, never on the test set
    val_scores = clf.decision_function(val[FEATURE_COLUMNS])
    val_y = val["incident"].to_numpy()
    threshold = choose_threshold(val_y, val_scores, target_precision)

    scores = clf.decision_function(test[FEATURE_COLUMNS])
    y = test["incident"].to_numpy()
    precisions, recalls, thresholds = policy_curve(y, scores)

    # Baseline: the classic single-metric rule "5xx rate (15m) > 1%"
    rule = alert_policy(test["error_rate_pct_15m"].to_numpy() > 1.0)
    try:
        rule_threshold = choose_threshold(
            val_y, val["error_rate_pct_15m"].to_numpy(), target_precision)
    except ValueError:
        rule_threshold = None  # An infeasible baseline is reported, never silently relaxed.
    tuned_rule = (alert_policy(test["error_rate_pct_15m"].to_numpy() >= rule_threshold)
                  if rule_threshold is not None else None)
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(
        df[["ts", *FEATURE_COLUMNS, "incident"]], index=False).to_numpy().tobytes()).hexdigest()
    return Results(y, scores, rule, precisions, recalls, thresholds, threshold,
                   clf, tuned_rule, rule_threshold, data_hash)


def alert_policy(above: np.ndarray) -> np.ndarray:
    """Fire on the second consecutive above-threshold sample; clear immediately.

    Every evaluated batch starts inactive. Two 5-minute samples imply a
    5-minute wait after the first above-threshold sample, not a 10-minute wait.
    """
    above = np.asarray(above, dtype=bool)
    fired = np.zeros(len(above), dtype=bool)
    fired[1:] = above[1:] & above[:-1]
    return fired


def choose_threshold(y: np.ndarray, scores: np.ndarray, target_precision: float) -> float:
    """Meet window AND episode precision after pending, then maximize incident recall.

    Break ties by fewer false episodes/day, shorter mean detection delay,
    less false alert time, then the lower threshold. Window precision guards
    against an always-active alert winning on incident overlap alone.
    """
    if not 0 < target_precision <= 1:
        raise ValueError("precision target must be in (0, 1]")
    best = None
    for threshold in np.unique(scores):
        pred = alert_policy(scores >= threshold)
        tp = int((pred & (y == 1)).sum())
        if not pred.any() or tp / pred.sum() < target_precision:
            continue
        metrics = incident_metrics(y, pred)
        if metrics["precision"] < target_precision:
            continue
        rank = (-metrics["recall"], metrics["false_alert_episodes_per_day"],
                metrics["mean_detection_delay_minutes"], metrics["false_alert_minutes"], threshold)
        if best is None or rank < best[0]:
            best = (rank, float(threshold))
    if best is None:
        raise ValueError(f"no alert threshold reaches {target_precision:.0%} "
                         "window and episode precision after pending")
    return best[1]


def policy_curve(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window precision/recall after applying the same pending policy at each threshold."""
    thresholds, precisions, recalls = [], [], []
    positives = int((y == 1).sum())
    for threshold in np.unique(scores):
        pred = alert_policy(scores >= threshold)
        if not pred.any():
            continue
        thresholds.append(threshold)
        tp = int((pred & (y == 1)).sum())
        precisions.append(tp / pred.sum() if pred.any() else 0.0)
        recalls.append(tp / positives if positives else 0.0)
    return np.array([*precisions, 1.0]), np.array([*recalls, 0.0]), np.array(thresholds)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True as (start, stop) index pairs."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def window_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Every 5-minute window counts on its own."""
    pred = pred.astype(bool)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "false_alarm_windows": int((pred & (y == 0)).sum()),
        "missed_windows": int((~pred & (y == 1)).sum()),
    }


def incident_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Incident overlap, detection delay and alert duration for 5-minute windows.

    Recall: share of incidents with at least one alerted window.
    Precision: share of alert episodes (runs of consecutive alerted windows)
    that overlap a real incident. Overlap alone rewards alerts that never clear.
    Mean detection delay is measured from incident onset for caught incidents
    only (None if none are caught); an already-active alert has zero delay.
    """
    pred = pred.astype(bool)
    incidents, alerts = _runs(y == 1), _runs(pred)
    delays = [int(np.flatnonzero(pred[a:b])[0]) * STEP_MINUTES
              for a, b in incidents if pred[a:b].any()]
    caught = len(delays)
    real_alerts = sum((y[a:b] == 1).any() for a, b in alerts)
    return {
        "precision": real_alerts / len(alerts) if alerts else 0.0,
        "recall": caught / len(incidents) if incidents else 0.0,
        "incidents": len(incidents),
        "caught": int(caught),
        "alert_episodes": len(alerts),
        "false_alert_episodes": len(alerts) - int(real_alerts),
        "mean_detection_delay_minutes": float(np.mean(delays)) if delays else None,
        "alert_active_minutes": int(pred.sum()) * STEP_MINUTES,
        "false_alert_minutes": int((pred & (y == 0)).sum()) * STEP_MINUTES,
        "false_alert_episodes_per_day": ((len(alerts) - int(real_alerts)) * 1440
                                         / (len(y) * STEP_MINUTES) if len(y) else 0.0),
    }
