"""Repeat the complete train/validation/test experiment across simulation seeds."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import alert_policy, incident_metrics, train_and_score, window_metrics
from simulate import simulate


def run_benchmark(seeds: int, precision: float) -> dict:
    if seeds < 1 or not 0 < precision <= 1:
        raise ValueError("seeds must be positive and precision must be in (0, 1]")
    runs = []
    for seed in range(seeds):
        try:
            r = train_and_score(simulate(seed), precision)
        except ValueError as e:
            if not str(e).startswith("no alert threshold"):
                raise
            runs.append({"seed": seed, "error": str(e)})
            continue
        methods = {"classifier": alert_policy(r.scores >= r.threshold),
                   "fixed_rule": r.rule, "tuned_rule": r.tuned_rule}
        row = {"seed": seed, "threshold": r.threshold,
               "tuned_rule_threshold": r.tuned_rule_threshold, "methods": {}}
        for name, pred in methods.items():
            if pred is None:
                continue
            w, i = window_metrics(r.y, pred), incident_metrics(r.y, pred)
            row["methods"][name] = {
                "window_precision": float(w["precision"]), "window_recall": float(w["recall"]),
                "episode_precision": float(i["precision"]), "incident_recall": i["recall"],
                "false_episodes_per_day": i["false_alert_episodes_per_day"],
                "mean_detection_delay_minutes": i["mean_detection_delay_minutes"],
                "false_alert_minutes": i["false_alert_minutes"],
            }
        runs.append(row)
    summary = {}
    for name in ("classifier", "fixed_rule", "tuned_rule"):
        metrics = [r["methods"][name] for r in runs if name in r.get("methods", {})]
        summary[name] = {"evaluated_runs": len(metrics), "metrics": {}}
        for metric in metrics[0] if metrics else []:
            values = [m[metric] for m in metrics if m[metric] is not None]
            if values:
                summary[name]["metrics"][metric] = {
                    "min": float(np.min(values)), "median": float(np.median(values)),
                    "max": float(np.max(values)),
                }
        summary[name]["below_test_precision_target"] = sum(
            m["window_precision"] < precision or m["episode_precision"] < precision
            for m in metrics)
    return {"seeds": seeds, "target_precision": precision,
            "infeasible_classifier_runs": sum("error" in r for r in runs),
            "summary": summary, "runs": runs}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=30, help="run seeds 0 through N-1")
    ap.add_argument("--precision", type=float, default=0.90)
    ap.add_argument("--output", type=Path, default=Path("data/robustness.json"))
    args = ap.parse_args()
    try:
        report = run_benchmark(args.seeds, args.precision)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))
    print(f"Per-seed results written to {args.output}")


if __name__ == "__main__":
    main()
