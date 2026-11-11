"""Run the whole experiment: simulate data, train, evaluate, plot.

    python main.py                 # default: threshold for >= 90% precision
    python main.py --precision 0.8 # try a different precision target
"""
import argparse
import os

from evaluate import incident_metrics, train_and_score, window_metrics
from plots import precision_recall_vs_threshold, precision_vs_recall, threshold_tradeoff
from simulate import simulate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", type=float, default=0.90,
                    help="precision target used to pick the threshold (default 0.90)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    df = simulate(args.seed)
    df.to_csv("data/k8s_checkout_metrics.csv", index=False)
    try:
        r = train_and_score(df, args.precision)
    except ValueError as e:
        raise SystemExit(f"error: {e}")

    pred = r.scores >= r.threshold
    rows = [
        (f"Classifier @ {r.threshold:.2f}", window_metrics(r.y, pred), incident_metrics(r.y, pred)),
        ("Rule: 5xx (15m) > 1%", window_metrics(r.y, r.rule), incident_metrics(r.y, r.rule)),
    ]
    print(f"Test set: {len(r.y)} windows, {r.y.sum()} incident windows\n")
    print(f"{'':24} {'per window':>22} {'per incident':>30}")
    print(f"{'':24} {'P':>6} {'R':>6} {'false':>8} {'P':>8} {'R':>6} {'caught':>9} {'false':>6}")
    for name, w, i in rows:
        print(f"{name:24} {w['precision']:>6.0%} {w['recall']:>6.0%} {w['false_alarm_windows']:>8}"
              f" {i['precision']:>8.0%} {i['recall']:>6.0%}"
              f" {i['caught']:>4}/{i['incidents']:<4} {i['false_alert_episodes']:>6}")

    precision_recall_vs_threshold(r, "figures/precision_recall_vs_threshold.png")
    precision_vs_recall(r, "figures/precision_vs_recall.png")
    threshold_tradeoff(r, "figures/threshold_tradeoff.png",
                       {"Low": r.threshold - 2, "Chosen": r.threshold, "High": r.threshold + 2})
    print("\nFigures written to figures/")


if __name__ == "__main__":
    main()
