"""Figures in the style of Géron, Hands-On ML, chapter 3."""
import matplotlib.pyplot as plt
import numpy as np

from evaluate import Results, alert_policy, window_metrics

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def _grid(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def precision_recall_vs_threshold(r: Results, path: str):
    """Géron Fig. 3-5."""
    m = window_metrics(r.y, alert_policy(r.scores >= r.threshold))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(r.thresholds, r.precisions[:-1], "--", color=BLUE, lw=2,
            label="Window precision")
    ax.plot(r.thresholds, r.recalls[:-1], "-", color=ORANGE, lw=2,
            label="Window recall")
    ax.axvline(r.threshold, color=MUTED, ls=":", lw=1.5)
    for val, c in ((m["precision"], BLUE), (m["recall"], ORANGE)):
        ax.plot([r.threshold], [val], "o", color=c, ms=8, mec=SURFACE, mew=2)
    ax.annotate(f"threshold = {r.threshold:.2f}\n"
                f"precision {m['precision']:.0%}, recall {m['recall']:.0%}",
                xy=(r.threshold, m["recall"]), xytext=(r.threshold + 1.0, 0.35),
                color=INK, fontsize=10, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xlim(-8, 8); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Alert threshold (classifier decision score)")
    ax.set_ylabel("Score")
    _grid(ax)
    ax.legend(loc="center left", frameon=False)
    ax.set_title("checkout: window precision and recall vs. alert threshold",
                 loc="left", color=INK, fontsize=13)
    fig.text(0.01, 0.01, f"10 held-out days: {r.y.sum()}/{len(r.y)} incident windows. "
             "All curves use the two-sample pending policy.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def precision_vs_recall(r: Results, path: str):
    """Géron Fig. 3-6, with the single-metric rule for comparison."""
    m = window_metrics(r.y, alert_policy(r.scores >= r.threshold))
    b = window_metrics(r.y, r.rule)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot(r.recalls, r.precisions, color=BLUE, lw=2, label="Multi-signal classifier")
    ax.plot([m["recall"]], [m["precision"]], "o", color=BLUE, ms=8, mec=SURFACE, mew=2)
    ax.annotate(f"chosen threshold\nP {m['precision']:.0%} / R {m['recall']:.0%}",
                (m["recall"], m["precision"]), xytext=(-120, -50),
                textcoords="offset points", fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.plot([b["recall"]], [b["precision"]], "s", color=ORANGE, ms=9, mec=SURFACE, mew=2,
            label="Rule: 5xx rate (15m) > 1%")
    ax.annotate(f"P {b['precision']:.0%} / R {b['recall']:.0%}", (b["recall"], b["precision"]),
                xytext=(10, -18), textcoords="offset points", fontsize=10, color=INK)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    if r.tuned_rule is not None:
        tuned = window_metrics(r.y, r.tuned_rule)
        ax.plot([tuned["recall"]], [tuned["precision"]], "D", color=INK, ms=7,
                label=f"Tuned 5xx rule: >= {r.tuned_rule_threshold:.2f}%")
    ax.set_xlabel("Window recall")
    ax.set_ylabel("Window precision")
    _grid(ax)
    ax.legend(loc="lower left", frameon=False)
    ax.set_title("Window precision vs. recall (with pending)", loc="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def threshold_tradeoff(r: Results, path: str, settings: dict[str, float]):
    """False alarms vs. missed incident time at a few thresholds (for part 2)."""
    names = list(settings)
    ms = [window_metrics(r.y, alert_policy(r.scores >= t)) for t in settings.values()]
    fa = [m["false_alarm_windows"] for m in ms]
    missed = [m["missed_windows"] for m in ms]

    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars_fa = ax.bar(x - w / 2 - 0.01, fa, w, color=BLUE, label="False alarms (5-min windows)")
    bars_mi = ax.bar(x + w / 2 + 0.01, missed, w, color=ORANGE,
                     label="Missed incident time (5-min windows)")
    for bars in (bars_fa, bars_mi):
        ax.bar_label(bars, padding=3, color=INK, fontsize=10)
    labels = [f"{n}\nthreshold {t:.2f}\nP {m['precision']:.0%} / R {m['recall']:.0%}"
              for (n, t), m in zip(settings.items(), ms)]
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Count over 10 test days")
    ax.set_ylim(0, max(fa + missed) * 1.18)
    ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Moving the threshold trades false alarms for missed incidents",
                 loc="left", color=INK, fontsize=13, pad=24)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
