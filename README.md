# Your alert rule is a classifier

An alert rule decides "incident" or "no incident", so you can evaluate it like any classifier:
measure its precision and recall, then choose the threshold on purpose instead of by gut feeling.

This repo applies the precision/recall tools from chapter 3 of Géron's *Hands-On Machine Learning*
to alerting on a **simulated** Kubernetes service, and adapts them to how alerting data behaves.

![Window precision and recall vs. alert threshold](figures/precision_recall_vs_threshold.png)

## The scenario (all data is simulated)

- 30 days of 5-minute windows for a `checkout` Deployment: p99 latency, 5xx error rate, pod restarts,
  CPU throttling, memory working set and ready-replica ratio, with a daily traffic cycle.
- 45 labelled incidents of four kinds (OOM, failing dependency, crashloop, slow database), some deliberately subtle.
- Noise that looks bad but isn't an incident: 120 rolling deploys and 60 nightly batch jobs.
- Time-based split: train on days 1–14, pick the threshold on days 15–20, test on days 21–30.
  The test set never influences a choice. About 6% of test windows are incident time.

The model is a logistic regression on nine explicitly listed features. Its decision score is what the threshold
moves along, as with `decision_function` in the book.

### Alert policy and threshold selection

An alert fires on the **second consecutive above-threshold sample** and clears on the first sample below
threshold. With 5-minute samples, that waits 5 minutes after the first positive sample. Each validation,
test, or shadow-scoring batch starts inactive; no pending state carries between batches.

Thresholds are selected on validation data **after applying this policy**. The default requires at least
90% precision for both windows and alert episodes. Window precision prevents a permanently active alert
from winning on episode overlap alone. Among feasible thresholds, select the highest incident recall;
break ties with fewer false episodes/day, shorter mean detection delay, less false alert time, then the
lower threshold. An unreachable target fails explicitly. These are empirical validation targets, not
guarantees about future precision.

The fixed 1% error-rate rule and a validation-tuned error-rate rule use the same pending policy. The tuned
rule uses the same objectives and precision targets as the classifier; infeasible tuning is reported
explicitly rather than relaxing its target. All plotted curves also include the pending policy.

## Results on the 10 test days

|  | Precision (window) | Recall (window) | False-alarm windows | Precision (alert episode) | Recall (incident) | False alert episodes |
|---|---|---|---|---|---|---|
| Classifier, threshold −1.26 | 94% | 76% | 8 | 87% | 16 / 17 | 3 |
| Rule: 5xx rate over 15 min > 1% | 60% | 35% | 40 | 41% | 11 / 17 | 20 |
| Tuned rule: 5xx rate over 15 min ≥ 1.64% | 80% | 16% | 7 | 60% | 6 / 17 | 4 |

![Window precision vs. recall](figures/precision_vs_recall.png)

At these validation-selected operating points, the classifier catches more test incidents than either
single-metric rule. Keeping both baselines separates comparison with the familiar 1% rule from comparison
with a rule that has also had its threshold tuned.

### Moving the threshold

![False alarms vs. missed incident time at three thresholds](figures/threshold_tradeoff.png)

No setting is "correct". Which one to use depends on what a missed incident costs compared with a false page.

### Per window vs. per incident

The same predictions look different depending on what counts as a positive:

- **Per window:** every 5-minute window is scored on its own. Catching 3 windows of a 6-window outage gives 50% recall.
- **Per incident:** an incident counts as caught if any window is alerted. An *alert episode* is a run of
  consecutive alerted windows, and it's a false alert if it overlaps no incident.

At the chosen threshold, the classifier catches 16 of 17 test incidents even though per-window recall is 76%.
Alert-episode precision is 87%: 8 false windows make 3 separate false alert episodes. These episodes do not
model actual notification delivery, repeat notifications, or routing.

Overlap scores alone can be misleading: an alert that stays active throughout the test gets 100% episode
precision and incident recall whenever there is at least one incident. An alert in the last window of an
incident also counts as a catch. The evaluation therefore reports timing alongside overlap:

|  | Mean detection delay (min) | Alert-active time (min) | Alert-active time outside incidents (min) | False episodes/day |
|---|---|---|---|---|
| Classifier, threshold −1.26 | 5.0 | 685 | 40 | 0.30 |
| Rule: 5xx rate over 15 min > 1% | 15.5 | 495 | 200 | 2.00 |
| Tuned rule: 5xx rate over 15 min ≥ 1.64% | 20.0 | 175 | 35 | 0.40 |

Detection delay runs from incident onset to its first alerted window, measured in 5-minute steps.
The mean covers caught incidents only; if none are caught, it is reported as `N/A`. An alert already active
at incident onset has zero delay. Read delay alongside incident recall, since missed incidents are excluded.
Alert-active time sums all alerted windows; time outside incidents exposes persistent false alerting even
when one long alert episode overlaps real incidents. These metrics assume contiguous 5-minute windows.

### Robustness across simulation seeds

`uv run benchmark.py --seeds 30` repeats training, validation selection, and held-out evaluation for seeds
0–29, writing per-seed results and min/median/max summaries for all three methods to `data/robustness.json`.
It reports infeasible classifier runs and the number of evaluated runs per method; infeasible runs are
not turned into successful predictions. Baseline summaries cover successful classifier runs, and the
tuned baseline additionally requires a feasible validation threshold.

For the classifier at the default 90% target, all 30 validation selections were feasible:

| Held-out metric | Minimum | Median | Maximum |
|---|---|---|---|
| Window precision | 88.1% | 97.1% | 100% |
| Window recall | 46.4% | 66.1% | 84.1% |
| Alert-episode precision | 68.2% | 90.0% | 100% |
| Incident recall | 61.5% | 92.6% | 100% |
| False episodes/day | 0.0 | 0.2 | 0.7 |
| Mean detection delay (min) | 5.0 | 10.0 | 18.3 |

In 15 of 30 runs, at least one held-out precision measure was below 90%. Fixed seeds make experiments
repeatable, not certain. These ranges describe variation under this simulator; they are not confidence
intervals for real production traffic. Delay summaries exclude runs with no caught incidents. Real-data
evaluation still needs repeated chronological holdouts and uncertainty estimates grouped by incident or day.

## Run it

With [uv](https://docs.astral.sh/uv/):

```bash
uv run main.py                  # 90% validation precision in both units, after pending
uv run main.py --precision 0.8  # try another precision target
uv run benchmark.py --seeds 30 # measure variation across seeds 0–29
```

`uv run` creates the virtual environment and installs the locked dependencies on first use.
Run the tests with `uv run pytest`.

`main.py` writes the dataset to `data/`, the figures to `figures/`, and prints the tables above.
It also saves a uniquely named model bundle under `artifacts/`, preserving earlier bundles for rollback.
The bundle contains the fitted scaler and classifier, threshold, feature order, pending policy, cadence,
seed, split dates, creation time, Python/library versions, and data/source fingerprints. The seed is fixed
by default. Data fingerprints cover normalized timestamps, selected features and labels; source fingerprints
cover the simulator, feature contract, evaluator, main command and scorer, including uncommitted edits.

### Shadow scoring

Use the exact model path printed by `main.py`:

```bash
uv run score.py artifacts/model-<run>.pkl data/k8s_checkout_metrics.csv data/shadow_predictions.csv \
  --as-of 2026-10-31T00:00:00Z
```

The scorer writes timestamps, decision scores, and alert states to a new CSV; it never sends notifications
and refuses to overwrite an existing output. Omit `--as-of` for fresh telemetry; the latest timestamp must
be within the last 5 minutes and must not be in the future. `--as-of` explicitly sets the reference time
for historical replay. Missing or invalid telemetry stops scoring rather than being treated as healthy.

Training and scoring require `ts` and the nine prepared features listed in `features.py`. Training also
requires binary `incident` labels and nonempty train/validation/test periods. Inputs must be finite numeric
values, with unique increasing timestamps at a contiguous 5-minute cadence. Timestamps are normalized to
UTC; naive timestamps are interpreted as UTC. Additional columns are ignored, including incident metadata.
The three `_15m` features must use trailing three-sample means, as in `simulate.py`; callers supply that
causal history, including before the start of a scoring batch.

Only load model bundles you trust: Python pickle files can execute code. Loading checks that feature,
policy, cadence, and runtime versions match. Scoring is batch-based shadow evaluation, with no cross-batch
alert state or automatic retraining. Use the same pinned environment as training, validate real telemetry
in shadow mode, and retain an existing operational alert until real-data performance has been assessed.

## Layout

| File | What it does |
|---|---|
| `simulate.py` | Generates the fake Kubernetes metrics, incidents and noise |
| `features.py` | Explicit feature list and shared input validation |
| `evaluate.py` | Time-based split, training, pending policy, operational threshold selection and metrics |
| `plots.py` | The three figures |
| `main.py` | Runs the experiment, prints results and saves a model bundle |
| `benchmark.py` | Repeats the experiment across seeds and reports variation |
| `score.py` | Loads a trusted bundle and writes shadow predictions |
| `tests/` | Metrics, policy, threshold choice, input validation and model round-trip tests |

## Caveats

- All numbers depend on how the data was simulated. They demonstrate the method, not a benchmark.
- Incident boundaries are exact here. Real labels from post-mortems are approximate.
- Logistic regression is deliberately simple. The point is how to evaluate and choose, not the model.

## License

MIT
