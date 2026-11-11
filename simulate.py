"""Simulate 30 days of 5-minute metrics for a fake Kubernetes `checkout` Deployment.

Real incidents (label=1) are mixed with noise that looks bad but is not an incident:
rolling deploys and nightly batch jobs. All data is synthetic; the seed is fixed.
"""
import numpy as np
import pandas as pd


def simulate(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    DAYS, STEP_MIN = 30, 5
    n = DAYS * 24 * 60 // STEP_MIN
    ts = pd.date_range("2026-10-01", periods=n, freq=f"{STEP_MIN}min")
    hour = (ts.hour + ts.minute / 60).to_numpy()
    traffic = 0.6 + 0.4 * np.sin((hour - 9) / 24 * 2 * np.pi)  # daily cycle

    # Healthy baseline
    p99_ms = 180 + 60 * traffic + rng.gamma(2, 12, n)
    err_pct = np.clip(0.15 + 0.1 * traffic + rng.normal(0, 0.08, n), 0, None)
    restarts = rng.poisson(0.03, n).astype(float)
    cpu_throttle = np.clip(5 + 10 * traffic + rng.normal(0, 4, n), 0, 100)
    mem_pct = np.clip(55 + 8 * traffic + rng.normal(0, 3, n), 0, 100)
    ready_ratio = np.ones(n)
    label = np.zeros(n, dtype=int)

    def episode(length_range):
        start = rng.integers(0, n - 30)
        return slice(start, start + rng.integers(*length_range))

    # Real incidents (the things you WANT to be paged for)
    kinds = ["oom", "bad_dependency", "crashloop", "slow_db"]
    for _ in range(45):
        s = episode((3, 18))  # 15 min - 90 min
        k = rng.choice(kinds)
        sev = rng.uniform(0.4, 1.5)  # some incidents are subtle
        m = s.stop - s.start
        label[s] = 1
        if k == "oom":
            mem_pct[s] += 30 * sev; restarts[s] += rng.poisson(2 * sev, m)
            err_pct[s] += 1.5 * sev * rng.random(m); ready_ratio[s] -= 0.2 * sev
        elif k == "bad_dependency":
            err_pct[s] += 4 * sev * rng.random(m); p99_ms[s] += 400 * sev
        elif k == "crashloop":
            restarts[s] += rng.poisson(4 * sev, m); ready_ratio[s] -= 0.4 * sev
            err_pct[s] += 2 * sev * rng.random(m)
        else:  # slow_db
            p99_ms[s] += 900 * sev * rng.random(m); err_pct[s] += 0.8 * sev * rng.random(m)

    # Noise: things that LOOK bad but aren't incidents (the false positives)
    for _ in range(120):  # rolling deploys: restarts + short error blips
        s = episode((1, 4)); m = s.stop - s.start
        restarts[s] += rng.poisson(1.5, m); err_pct[s] += rng.uniform(0.3, 2.5, m)
        ready_ratio[s] -= rng.uniform(0.05, 0.25, m)
    for _ in range(60):  # nightly batch jobs: CPU throttling + latency
        s = episode((4, 12)); m = s.stop - s.start
        cpu_throttle[s] += rng.uniform(20, 50, m); p99_ms[s] += rng.uniform(100, 600, m)

    df = pd.DataFrame({
        "ts": ts, "p99_latency_ms": p99_ms, "error_rate_pct": err_pct,
        "pod_restarts": restarts, "cpu_throttle_pct": np.clip(cpu_throttle, 0, 100),
        "mem_working_set_pct": np.clip(mem_pct, 0, 100),
        "ready_ratio": np.clip(ready_ratio, 0, 1), "incident": label,
    })
    # What an alert expression would actually use: short rolling windows
    for c in ["p99_latency_ms", "error_rate_pct", "pod_restarts"]:
        df[c + "_15m"] = df[c].rolling(3, min_periods=1).mean()
    return df
