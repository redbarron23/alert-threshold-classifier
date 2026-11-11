import numpy as np
import pandas as pd
import pickle
import pytest

from benchmark import run_benchmark
from evaluate import (alert_policy, choose_threshold, incident_metrics,
                      policy_curve, train_and_score, window_metrics)
from features import FEATURE_COLUMNS, validate_data
from score import load_model, save_model, shadow_score
from simulate import simulate


@pytest.fixture(scope="module")
def trained():
    return train_and_score(simulate())


def test_pending_policy_waits_and_clears():
    assert alert_policy(np.array([1, 0, 1, 1, 1, 0, 1, 1])).tolist() == [
        False, False, False, True, True, False, False, True]
    assert alert_policy(np.array([])).tolist() == []


def test_operational_selection_checks_both_precision_units():
    y = np.array([0, 0, 0, 1, 1, 0, 0, 0])
    scores = np.array([0, 0, 0, 2, 2, 0, 0, 0])
    threshold = choose_threshold(y, scores, .9)
    assert threshold == 2
    pred = alert_policy(scores >= threshold)
    assert window_metrics(y, pred)["precision"] == 1
    assert incident_metrics(y, pred)["precision"] == 1
    with pytest.raises(ValueError, match="no alert threshold"):
        choose_threshold(y, np.ones(len(y)), .9)


def test_selection_prefers_catching_more_incidents():
    y = np.array([0, 1, 1, 0, 1, 1, 0])
    scores = np.array([0, 2, 2, 0, 1, 1, 0])
    assert choose_threshold(y, scores, .9) == 1


def test_selection_prefers_fewer_false_episodes_then_faster_detection():
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    scores = np.array([1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0])
    assert choose_threshold(y, scores, .5) == 2
    scores = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0])
    assert choose_threshold(y, scores, .9) == 1


def test_policy_curve_matches_predictions():
    y = np.array([0, 1, 1, 0, 0])
    scores = np.array([0., 2., 2., 1., 0.])
    precisions, recalls, thresholds = policy_curve(y, scores)
    for j, threshold in enumerate(thresholds):
        m = window_metrics(y, alert_policy(scores >= threshold))
        assert precisions[j] == m["precision"] and recalls[j] == m["recall"]
    assert precisions[-1] == 1 and recalls[-1] == 0


@pytest.mark.parametrize("precision", [0, -1, 1.1, float("nan")])
def test_invalid_operational_precision_is_rejected(precision):
    with pytest.raises(ValueError, match="precision target"):
        choose_threshold(np.array([0, 1]), np.array([0., 1.]), precision)


def test_extra_metadata_cannot_become_a_feature(trained):
    df = simulate()
    df["postmortem_incident_id"] = df.incident * 1000
    other = train_and_score(df)
    assert other.model.feature_names_in_.tolist() == FEATURE_COLUMNS
    assert other.threshold == trained.threshold
    np.testing.assert_array_equal(other.scores, trained.scores)


@pytest.mark.parametrize("problem", ["missing", "nan", "infinity", "nonnumeric", "unsorted",
                                     "gap", "duplicate", "missing_time", "bad_label"])
def test_input_contract_rejects_bad_data(problem):
    df = simulate().iloc[:4].copy()
    if problem == "missing":
        df = df.drop(columns=FEATURE_COLUMNS[0])
    elif problem in ("nan", "infinity"):
        df.loc[0, FEATURE_COLUMNS[0]] = np.nan if problem == "nan" else np.inf
    elif problem == "nonnumeric":
        df[FEATURE_COLUMNS[0]] = "broken"
    elif problem == "unsorted":
        df = df.iloc[::-1]
    elif problem == "gap":
        df = df.drop(index=1)
    elif problem == "duplicate":
        df.loc[1, "ts"] = df.loc[0, "ts"]
    elif problem == "missing_time":
        df.loc[1, "ts"] = pd.NaT
    else:
        df.loc[0, "incident"] = 2
    with pytest.raises(ValueError):
        validate_data(df, labeled=True)


def test_test_labels_do_not_affect_threshold_or_model(trained):
    df = simulate()
    test = df.ts >= "2026-10-21"
    df.loc[test, "incident"] = 1 - df.loc[test, "incident"]
    other = train_and_score(df)
    assert other.threshold == trained.threshold
    assert other.tuned_rule_threshold == trained.tuned_rule_threshold
    np.testing.assert_array_equal(other.scores, trained.scores)


def test_training_requires_all_splits_and_both_training_classes():
    with pytest.raises(ValueError, match="splits must all be nonempty"):
        train_and_score(simulate().iloc[:500])
    df = simulate()
    df.loc[df.ts < "2026-10-15", "incident"] = 0
    with pytest.raises(ValueError, match="both incident classes"):
        train_and_score(df)


def test_saved_model_reproduces_test_predictions(trained, tmp_path):
    path = save_model(trained, 42, .9, tmp_path)
    artifact = load_model(path)
    df = simulate()
    df = df[df.ts >= "2026-10-21"].drop(columns="incident")
    result = shadow_score(artifact, df, "2026-10-31T00:00:00Z")
    np.testing.assert_array_equal(result.decision_score, trained.scores)
    np.testing.assert_array_equal(result.alert, alert_policy(trained.scores >= trained.threshold))
    assert artifact["metadata"]["seed"] == 42
    assert artifact["metadata"]["data_sha256"] == trained.data_sha256
    assert len(artifact["metadata"]["source_sha256"]) == 64
    assert artifact["features"] == FEATURE_COLUMNS
    assert save_model(trained, 42, .9, tmp_path) != path
    assert len(list(tmp_path.glob("*.pkl"))) == 2


@pytest.mark.parametrize("problem", ["versions", "features", "pending_windows", "step_minutes"])
def test_model_load_rejects_incompatible_contract(trained, tmp_path, problem):
    path = save_model(trained, 42, .9, tmp_path)
    with path.open("rb") as f:
        artifact = pickle.load(f)
    if problem == "versions":
        artifact["metadata"]["versions"]["python"] = "0.0.0"
    elif problem == "features":
        artifact["features"] = list(reversed(FEATURE_COLUMNS))
    else:
        artifact[problem] = 999
    with path.open("wb") as f:
        pickle.dump(artifact, f)
    with pytest.raises(ValueError, match="differ"):
        load_model(path)


@pytest.mark.parametrize("as_of", ["2026-11-01T00:00:00Z", "2026-10-30T23:50:00Z", "NaT"])
def test_shadow_scoring_rejects_stale_and_future_data(trained, as_of):
    with pytest.raises(ValueError, match="latest telemetry"):
        shadow_score({"model": trained.model, "features": FEATURE_COLUMNS,
                      "threshold": trained.threshold}, simulate(), as_of)


def test_benchmark_reports_ranges_and_infeasible_runs(monkeypatch):
    import benchmark
    real_train = benchmark.train_and_score

    def fail_second(df, precision):
        if df.p99_latency_ms.iloc[0] == simulate(1).p99_latency_ms.iloc[0]:
            raise ValueError("no alert threshold reaches target")
        return real_train(df, precision)

    monkeypatch.setattr(benchmark, "train_and_score", fail_second)
    report = run_benchmark(2, .9)
    assert report["infeasible_classifier_runs"] == 1
    assert report["summary"]["classifier"]["evaluated_runs"] == 1
    assert "error" in report["runs"][1]
    assert report["summary"]["classifier"]["metrics"]["incident_recall"]["min"] >= 0
