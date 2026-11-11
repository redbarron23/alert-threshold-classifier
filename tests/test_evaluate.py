import numpy as np
import pytest

from evaluate import _runs, incident_metrics, window_metrics


def test_runs_finds_contiguous_blocks():
    assert _runs(np.array([0, 1, 1, 0, 1, 0, 0, 1])) == [(1, 3), (4, 5), (7, 8)]
    assert _runs(np.zeros(4)) == []


def test_window_metrics():
    y = np.array([0, 1, 1, 0, 0, 1])
    pred = np.array([0, 1, 0, 1, 0, 1])
    m = window_metrics(y, pred)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["false_alarm_windows"] == 1
    assert m["missed_windows"] == 1


def test_incident_metrics_counts_episodes_not_windows():
    # Two incidents; the first is caught by one window, the second is missed.
    # Two alert episodes: one overlaps the first incident, one is a false page.
    y = np.array([1, 1, 1, 0, 0, 0, 1, 1])
    pred = np.array([0, 1, 0, 0, 1, 1, 0, 0])
    m = incident_metrics(y, pred)
    assert (m["incidents"], m["caught"]) == (2, 1)
    assert (m["alert_episodes"], m["false_alert_episodes"]) == (2, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["mean_detection_delay_minutes"] == 5
    assert m["alert_active_minutes"] == 15
    assert m["false_alert_minutes"] == 10


def test_always_active_alert_has_perfect_overlap_but_exposes_false_alert_time():
    y = np.array([0, 0, 1, 0, 0, 1, 0])
    m = incident_metrics(y, np.ones(7))
    assert m["precision"] == 1 and m["recall"] == 1
    assert m["alert_episodes"] == 1 and m["false_alert_episodes"] == 0
    assert m["mean_detection_delay_minutes"] == 0
    assert m["alert_active_minutes"] == 35
    assert m["false_alert_minutes"] == 25


def test_detection_delay_includes_alerts_in_last_incident_window():
    m = incident_metrics(np.array([0, 1, 1, 1, 0]), np.array([0, 0, 0, 1, 0]))
    assert m["recall"] == 1
    assert m["mean_detection_delay_minutes"] == 10
    assert m["alert_active_minutes"] == 5
    assert m["false_alert_minutes"] == 0


@pytest.mark.parametrize("y, pred, active_minutes", [
    ([1, 1, 0], [0, 0, 0], 0),
    ([0, 0, 0], [1, 1, 1], 15),
    ([], [], 0),
])
def test_no_caught_incidents_have_no_detection_delay(y, pred, active_minutes):
    m = incident_metrics(np.array(y), np.array(pred))
    assert m["mean_detection_delay_minutes"] is None
    assert m["caught"] == 0
    assert m["alert_active_minutes"] == active_minutes
    assert m["false_alert_minutes"] == active_minutes
