"""Drift monitor DAG: retrain when the data drifts AND enough new data exists.

compute_drift ---+
                 +--> decide --> trigger_training --> record_trigger
count_new_rows --+   (skips the run         |
                      unless both           +-------> notify (ML Alerts card)
                      conditions hold
                      and no cooldown)

The decision logic lives in utils/drift.py (unit-tested). training_dag keeps
its own hourly schedule; this DAG only adds an early start when drift shows.
"""

import os
import time
from datetime import datetime, timedelta

from airflow.exceptions import AirflowSkipException
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import dag, task
from utils.drift import (
    COOLDOWN_SECONDS,
    DRIFT_SQL,
    MIN_NEW_ROWS,
    count_new_labelled_rows,
    decide_retrain,
    load_last_trigger,
    parse_metrics,
    save_last_trigger,
)

DATA = "/opt/airflow/data"
GOOD = os.path.join(DATA, "good_data")
LABELS = os.path.join(DATA, "inference_labels.csv")
# remembers when drift last started a training run (cooldown)
STATE_FILE = os.path.join(DATA, ".drift_state.json")

APP_DB_CONN_ID = "app_db"
ML_ALERTS_WEBHOOK_URL = os.getenv("ML_ALERTS_WEBHOOK_URL", "")


@dag(
    schedule="*/10 * * * *",  # every 10 minutes
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=10)},
    tags=["drift", "training"],
)
def drift_monitor_dag():

    @task
    def compute_drift() -> dict:
        """The two Grafana drift metrics over the last 15 minutes."""
        from airflow.providers.postgres.hooks.postgres import PostgresHook

        row = PostgresHook(postgres_conn_id=APP_DB_CONN_ID).get_first(DRIFT_SQL)
        metrics = parse_metrics(row)
        print(f"[compute_drift] {metrics}")
        return metrics

    @task
    def count_new_rows() -> int:
        """New labelled rows in good_data (same count as training_dag)."""
        n = count_new_labelled_rows(GOOD, LABELS)
        print(f"[count_new_rows] {n} new labelled rows (need {MIN_NEW_ROWS})")
        return n

    @task
    def decide(metrics: dict, new_rows: int) -> dict:
        """Skip the run unless drift AND enough data AND no recent trigger."""
        d = decide_retrain(
            metrics, new_rows, load_last_trigger(STATE_FILE), time.time()
        )
        print(f"[decide] retrain={d.retrain}: {d.reason}")
        if not d.retrain:
            raise AirflowSkipException(d.reason)
        return {"metrics": metrics, "new_rows": new_rows, "reason": d.reason}

    trigger = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="training_dag",
        conf={"trigger": "drift"},
        wait_for_completion=False,
    )

    @task
    def record_trigger() -> None:
        """Start the cooldown (after the trigger really happened)."""
        save_last_trigger(STATE_FILE, time.time())
        print(f"[record_trigger] cooldown {COOLDOWN_SECONDS // 60} min started")

    @task
    def notify(decision: dict) -> None:
        from utils.teams import adaptive_card, send

        m = decision["metrics"]
        send(
            ML_ALERTS_WEBHOOK_URL,
            adaptive_card(
                "Retraining triggered by drift",
                [
                    ("Why", decision["reason"]),
                    ("Grid z-score", f"{m['grid_zscore']:.2f}"),
                    ("Top-10 rate change", f"{m['rate_delta']:.2f}"),
                    ("New labelled rows", str(decision["new_rows"])),
                    ("Started", "training_dag (conf trigger=drift)"),
                ],
                "Warning",
            ),
            "[notify]",
        )

    decision = decide(compute_drift(), count_new_rows())
    decision >> trigger
    trigger >> record_trigger()
    trigger >> notify(decision)


drift_monitor_dag()
