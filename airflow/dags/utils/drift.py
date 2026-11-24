"""Pure helpers for the drift monitor DAG (no Airflow imports, unit-tested).

Drift-triggered retraining needs BOTH conditions:
    1. drift is detected: the same two metrics, thresholds and 15-minute
       window as the Grafana alerts
       (grafana/provisioning/alerting/alert_rules.yml)
    2. enough new labelled data: at least TRAINING_MIN_NEW_ROWS rows in
       good_data, counted exactly like training_dag.load_data
plus a cooldown, so a lasting drift starts ONE training run, not one every
10 minutes.
"""

import glob
import json
import os
from dataclasses import dataclass

import pandas as pd
from utils.training import label_rows

# same values as the Grafana alert rules: keep them in sync
GRID_ZSCORE_THRESHOLD = 1.0
RATE_DELTA_THRESHOLD = 0.2
WINDOW_MINUTES = 15

# same default as training_dag: below this the training run would skip anyway
MIN_NEW_ROWS = int(os.getenv("TRAINING_MIN_NEW_ROWS", "100"))
COOLDOWN_SECONDS = 3600  # no new drift trigger within an hour of the last one

# Both Grafana queries in one: live window of prediction_logs versus the
# champion's baseline in training_stats (WHERE is_baseline).
DRIFT_SQL = f"""
WITH grid_base AS (
    SELECT
        MAX(CASE WHEN stat = 'mean' THEN value END) AS mean_val,
        NULLIF(MAX(CASE WHEN stat = 'std' THEN value END), 0) AS std_val
    FROM training_stats
    WHERE feature = 'Grid' AND is_baseline = true
),
rate_base AS (
    SELECT value AS base_rate
    FROM training_stats
    WHERE feature = 'top10' AND stat = 'positive_rate' AND is_baseline = true
    LIMIT 1
),
live AS (
    SELECT
        AVG(grid::float) AS live_mean,
        AVG(CASE WHEN prediction THEN 1.0 ELSE 0.0 END) AS live_rate,
        COUNT(*) AS n_predictions
    FROM prediction_logs
    WHERE created_at >= NOW() - INTERVAL '{WINDOW_MINUTES} minutes'
)
SELECT
    COALESCE(ABS(live.live_mean - grid_base.mean_val) / grid_base.std_val, 0)
        AS grid_zscore,
    COALESCE(ABS(live.live_rate - rate_base.base_rate), 0) AS rate_delta,
    live.n_predictions
FROM live CROSS JOIN grid_base CROSS JOIN rate_base
"""


@dataclass(frozen=True)
class Decision:
    retrain: bool
    drift: bool
    enough_data: bool
    cooling_down: bool
    reason: str


def parse_metrics(row) -> dict:
    """The (grid_zscore, rate_delta, n_predictions) row of DRIFT_SQL as a
    dict. No row (no baseline yet) or NULLs mean no drift."""
    if not row:
        return {"grid_zscore": 0.0, "rate_delta": 0.0, "n_predictions": 0}
    grid_zscore, rate_delta, n_predictions = row
    return {
        "grid_zscore": float(grid_zscore or 0.0),
        "rate_delta": float(rate_delta or 0.0),
        "n_predictions": int(n_predictions or 0),
    }


def decide_retrain(
    metrics: dict,
    new_rows: int,
    last_trigger_ts: float | None,
    now_ts: float,
    min_new_rows: int = MIN_NEW_ROWS,
    cooldown_s: int = COOLDOWN_SECONDS,
) -> Decision:
    """Retrain only if drift AND enough new data AND not in cooldown."""
    z = metrics.get("grid_zscore") or 0.0
    delta = metrics.get("rate_delta") or 0.0

    drifted = []
    if z > GRID_ZSCORE_THRESHOLD:
        drifted.append(f"Grid z-score {z:.2f} > {GRID_ZSCORE_THRESHOLD}")
    if delta > RATE_DELTA_THRESHOLD:
        drifted.append(f"top-10 rate change {delta:.2f} > {RATE_DELTA_THRESHOLD}")
    drift = bool(drifted)

    enough = new_rows >= min_new_rows
    age = None if last_trigger_ts is None else now_ts - last_trigger_ts
    cooling = age is not None and age < cooldown_s

    if not drift:
        reason = f"no drift (Grid z-score {z:.2f}, top-10 rate change {delta:.2f})"
    elif not enough:
        reason = (
            f"drift ({'; '.join(drifted)}) but only {new_rows} of the "
            f"{min_new_rows} new labelled rows needed"
        )
    elif cooling:
        reason = (
            f"drift and {new_rows} new rows, but retraining was triggered "
            f"{int(age // 60)} min ago (cooldown {cooldown_s // 60} min)"
        )
    else:
        reason = f"drift ({'; '.join(drifted)}) and {new_rows} new labelled rows"

    return Decision(
        retrain=drift and enough and not cooling,
        drift=drift,
        enough_data=enough,
        cooling_down=cooling,
        reason=reason,
    )


def count_new_labelled_rows(good_dir: str, labels_path: str) -> int:
    """New labelled rows waiting in good_data: the same count as
    training_dag.load_data (files already used by a promoted model were
    archived, so they are not counted again)."""
    files = sorted(glob.glob(os.path.join(good_dir, "*.csv")))
    if not files or not os.path.exists(labels_path):
        return 0
    rows = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    return len(label_rows(rows, pd.read_csv(labels_path)))


def load_last_trigger(path: str) -> float | None:
    """When retraining was last triggered by drift (None if never)."""
    try:
        with open(path) as f:
            return float(json.load(f)["last_trigger_ts"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_last_trigger(path: str, ts: float) -> None:
    with open(path, "w") as f:
        json.dump({"last_trigger_ts": ts}, f)
