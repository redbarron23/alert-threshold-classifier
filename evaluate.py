"""Train the alert classifier and score it per window and per incident."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from simulate import feature_columns

# Time-based split: train on days 1-14, pick the threshold on days 15-20,
# report results on days 21-30. The test set never influences any choice.
VAL_DATE = "2026-10-15"
TEST_DATE = "2026-10-21"


@dataclass
class Results:
    y: np.ndarray            # true labels for test windows
    scores: np.ndarray       # classifier decision scores for test windows
    rule: np.ndarray         # baseline rule predictions for test windows
    precisions: np.ndarray
    recalls: np.ndarray
    thresholds: np.ndarray   # test-set curve, for the figures
    threshold: float         # chosen on the validation set


def train_and_score(df: pd.DataFrame, target_precision: float = 0.90) -> Results:
    features = feature_columns(df)
    train = df[df["ts"] < VAL_DATE]
    val = df[(df["ts"] >= VAL_DATE) & (df["ts"] < TEST_DATE)]
    test = df[df["ts"] >= TEST_DATE]

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(train[features], train["incident"])

    # Pick the threshold on validation data, never on the test set
    val_scores = clf.decision_function(val[features])
    threshold = choose_threshold(val["incident"].to_numpy(), val_scores, target_precision)

    scores = clf.decision_function(test[features])  # like Géron's sgd_clf.decision_function
    y = test["incident"].to_numpy()
    precisions, recalls, thresholds = precision_recall_curve(y, scores)

    # Baseline: the classic single-metric rule "5xx rate (15m) > 1%"
    rule = test["error_rate_pct_15m"].to_numpy() > 1.0
    return Results(y, scores, rule, precisions, recalls, thresholds, float(threshold))


def choose_threshold(y: np.ndarray, scores: np.ndarray, target_precision: float) -> float:
    """Lowest threshold reaching the precision target, as in Géron ch. 3.

    Raises ValueError if no threshold reaches it, instead of silently
    falling back to the lowest threshold (which alerts on almost everything).
    """
    precisions, _, thresholds = precision_recall_curve(y, scores)
    reached = precisions[:-1] >= target_precision  # last point has no threshold
    if not reached.any():
        raise ValueError(f"no threshold reaches {target_precision:.0%} precision "
                         f"(best is {precisions[:-1].max():.0%})")
    return float(thresholds[np.argmax(reached)])


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
    """What on-call cares about.

    Recall: share of incidents with at least one alerted window.
    Precision: share of alert episodes (runs of consecutive alerted windows)
    that overlap a real incident.
    """
    pred = pred.astype(bool)
    incidents, alerts = _runs(y == 1), _runs(pred)
    caught = sum(pred[a:b].any() for a, b in incidents)
    real_alerts = sum((y[a:b] == 1).any() for a, b in alerts)
    return {
        "precision": real_alerts / len(alerts) if alerts else 0.0,
        "recall": caught / len(incidents) if incidents else 0.0,
        "incidents": len(incidents),
        "caught": int(caught),
        "alert_episodes": len(alerts),
        "false_alert_episodes": len(alerts) - int(real_alerts),
    }
