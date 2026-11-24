import os
import random
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
from airflow.exceptions import AirflowSkipException
from airflow.sdk import dag, task
from utils.ge_helpers import validate_dataframe
from utils.ingestion import (
    STATS_COLUMNS,
    build_stats_row,
    build_teams_alert,
    split_good_bad,
)

RAW = "/opt/airflow/data/raw_data"
GOOD = "/opt/airflow/data/good_data"
BAD = "/opt/airflow/data/bad_data"

TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL", "")
DATA_DOCS_BASE_URL = os.getenv("DATA_DOCS_BASE_URL", "")
APP_DB_CONN_ID = "app_db"  # Airflow Connection to the project database


@dag(
    schedule="* * * * *",  # every minute (course requirement)
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    # one retry absorbs transient Airflow API errors (seen: an XCom push that
    # timed out, was retried and hit a duplicate-key 409). Tasks read their
    # input from upstream XComs, so a retry re-processes the same file.
    default_args={"retries": 1, "retry_delay": timedelta(seconds=10)},
    tags=["ingestion"],
)
def ingestion_dag():

    @task
    def read_data() -> dict:
        files = [f for f in os.listdir(RAW) if f.endswith(".csv")]
        if not files:
            raise AirflowSkipException("no files in raw_data")
        chosen = random.choice(files)
        path = os.path.join(RAW, chosen)
        df = pd.read_csv(path)
        os.remove(path)
        return {"file_name": chosen, "data": df.to_json(orient="records")}

    @task
    def validate_data(payload: dict) -> dict:
        df = pd.read_json(StringIO(payload["data"]), orient="records")
        result = validate_dataframe(df)
        result["file_name"] = payload["file_name"]
        result["data"] = payload["data"]
        return result

    @task
    def save_statistics(v: dict) -> None:
        from airflow.providers.postgres.hooks.postgres import PostgresHook

        hook = PostgresHook(postgres_conn_id=APP_DB_CONN_ID)
        placeholders = ",".join(["%s"] * len(STATS_COLUMNS))
        hook.run(
            f"INSERT INTO ingestion_stats ({', '.join(STATS_COLUMNS)}) "
            f"VALUES ({placeholders})",
            parameters=build_stats_row(v),
        )
        print(
            f"[save_statistics] saved stats for {v['file_name']} "
            f"(criticality={v['criticality']})"
        )

    @task
    def send_alerts(v: dict) -> None:
        from utils.teams import send

        if v["criticality"] not in ("medium", "high"):
            print(f"[send_alerts] criticality={v['criticality']}, no alert")
            return

        send(
            TEAMS_WEBHOOK_URL, build_teams_alert(v, DATA_DOCS_BASE_URL), "[send_alerts]"
        )

    @task
    def split_and_save_data(v: dict) -> None:
        df = pd.read_json(StringIO(v["data"]), orient="records")
        file_name = v["file_name"]

        os.makedirs(GOOD, exist_ok=True)
        os.makedirs(BAD, exist_ok=True)

        if v["invalid_rows"] == 0:
            df.to_csv(os.path.join(GOOD, file_name), index=False)
            print(f"[split_and_save_data] all rows clean -> {GOOD}/{file_name}")

        elif v["invalid_rows"] == v["total_rows"]:
            df.to_csv(os.path.join(BAD, file_name), index=False)
            print(f"[split_and_save_data] all rows bad -> {BAD}/{file_name}")

        else:
            good_df, bad_df = split_good_bad(df, v["bad_row_indices"])
            good_df.to_csv(os.path.join(GOOD, file_name), index=False)
            bad_df.to_csv(os.path.join(BAD, file_name), index=False)
            print(
                f"[split_and_save_data] split: {len(good_df)} good, "
                f"{len(bad_df)} bad rows -> {file_name}"
            )

    v = validate_data(read_data())
    save_statistics(v)
    send_alerts(v)
    split_and_save_data(v)


ingestion_dag()
