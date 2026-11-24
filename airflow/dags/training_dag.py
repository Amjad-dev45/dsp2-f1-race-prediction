"""Training DAG: retrain on new ingested data, promote if the candidate is at
least as good as the champion.

load_data -> train_model -> save_training_stats -> evaluate_candidate
    -> (only if it passes) promote_to_champion -> notify_api_reload
                                               -> archive_data
"""

import glob
import os
import shutil
import time
from datetime import datetime, timedelta

import pandas as pd
from airflow.exceptions import AirflowSkipException
from airflow.sdk import dag, task

DATA = "/opt/airflow/data"
GOOD = os.path.join(DATA, "good_data")
ARCHIVED = os.path.join(DATA, "archived_data")
BASE_TRAINING = os.path.join(DATA, "training_data.csv")
POOL = os.path.join(DATA, "inference.csv")
LABELS = os.path.join(DATA, "inference_labels.csv")

API_URL = os.getenv("API_URL", "http://api:8000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "f1-top10")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "f1-top10-training")
CHAMPION = "champion"
APP_DB_CONN_ID = "app_db"
ML_ALERTS_WEBHOOK_URL = os.getenv("ML_ALERTS_WEBHOOK_URL", "")

# retrain only once this many new LABELLED rows sit in good_data
MIN_NEW_ROWS = int(os.getenv("TRAINING_MIN_NEW_ROWS", "100"))
# promotion: inference on the whole test set must stay under this
MAX_INFERENCE_MS = float(os.getenv("TRAINING_MAX_INFERENCE_MS", "1000"))


def _read_csvs(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _champion_version(client) -> int | None:
    from mlflow.exceptions import MlflowException

    try:
        return int(client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version)
    except MlflowException:
        return None  # no registered model or no champion alias yet


@dag(
    schedule="0 * * * *",  # hourly; also trigger by hand for the demo
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=10)},
    tags=["training"],
)
def training_dag():

    @task
    def load_data() -> dict:
        """Skip unless enough new labelled rows arrived since the last
        promotion. Without a champion, train anyway (bootstrap)."""
        from mlflow import MlflowClient
        from utils.training import label_rows

        files = sorted(glob.glob(os.path.join(GOOD, "*.csv")))
        labels = pd.read_csv(LABELS)
        new_rows = len(label_rows(_read_csvs(files), labels))
        has_champion = _champion_version(MlflowClient()) is not None

        if has_champion and new_rows < MIN_NEW_ROWS:
            raise AirflowSkipException(
                f"{new_rows} new labelled rows in good_data (< {MIN_NEW_ROWS})"
            )
        print(
            f"[load_data] {len(files)} files, {new_rows} new labelled rows, "
            f"champion exists: {has_champion}"
        )
        return {"files": files, "new_rows": new_rows}

    @task
    def train_model(data: dict) -> dict:
        """Train on seed + archived + new data, log and register in MLflow."""
        import mlflow
        from mlflow.models import infer_signature
        from utils.training import (
            FEATURES,
            TARGET,
            build_pipeline,
            label_rows,
            training_set,
            training_stats,
        )

        labels = pd.read_csv(LABELS)
        archived = sorted(glob.glob(os.path.join(ARCHIVED, "*.csv")))
        ingested = _read_csvs(archived + data["files"])
        train = training_set(pd.read_csv(BASE_TRAINING), label_rows(ingested, labels))

        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run() as run:
            model = build_pipeline().fit(train[FEATURES], train[TARGET])
            mlflow.log_params(
                {
                    "train_rows": len(train),
                    "new_rows": data["new_rows"],
                    "new_files": len(data["files"]),
                    "archived_files": len(archived),
                    **{
                        k: v
                        for k, v in model.named_steps["clf"].get_params().items()
                        if k in ("n_estimators", "min_samples_leaf", "random_state")
                    },
                }
            )
            info = mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=MODEL_NAME,
                signature=infer_signature(
                    train[FEATURES].head(50), model.predict(train[FEATURES].head(50))
                ),
                input_example=train[FEATURES].head(3),
                # skops (MLflow's default sklearn format) only loads types
                # it is told to trust; the tree storage of RandomForest is
                # one. Safe here: models only come from this pipeline.
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
        version = int(info.registered_model_version)
        print(f"[train_model] {len(train)} rows -> {MODEL_NAME} v{version}")
        return {
            "version": version,
            "run_id": run.info.run_id,
            "stats": training_stats(train),
        }

    @task
    def save_training_stats(candidate: dict) -> dict:
        """Store the candidate's feature stats; they only become the drift
        baseline (is_baseline) if the candidate is promoted."""
        from airflow.providers.postgres.hooks.postgres import PostgresHook

        rows = [
            (
                candidate["version"],
                s["feature"],
                s["stat"],
                s["category"],
                s["value"],
                False,
            )
            for s in candidate["stats"]
        ]
        PostgresHook(postgres_conn_id=APP_DB_CONN_ID).insert_rows(
            "training_stats",
            rows,
            target_fields=[
                "model_version",
                "feature",
                "stat",
                "category",
                "value",
                "is_baseline",
            ],
        )
        print(f"[save_training_stats] {len(rows)} stats for v{candidate['version']}")
        return candidate

    @task.short_circuit
    def evaluate_candidate(candidate: dict) -> bool:
        """Candidate vs champion on the fixed test set. False -> downstream
        tasks are skipped (no promotion) and ML Alerts gets a message."""
        import mlflow
        from mlflow import MlflowClient
        from sklearn.metrics import accuracy_score, f1_score
        from utils.teams import adaptive_card, send
        from utils.training import FEATURES, TARGET, holdout_set, promotion_decision

        test = holdout_set(pd.read_csv(POOL), pd.read_csv(LABELS))
        X, y = test[FEATURES], test[TARGET]
        version = candidate["version"]

        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
        start = time.perf_counter()
        pred = model.predict(X)
        inference_ms = (time.perf_counter() - start) * 1000
        cand_acc = accuracy_score(y, pred)

        client = MlflowClient()
        champ_version = _champion_version(client)
        champ_acc = None
        if champ_version is not None:
            champ = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION}")
            champ_acc = accuracy_score(y, champ.predict(X))

        ok, reason = promotion_decision(
            cand_acc, champ_acc, inference_ms, MAX_INFERENCE_MS
        )
        metrics = {
            "test_accuracy": cand_acc,
            "test_f1": f1_score(y, pred),
            "test_inference_ms": inference_ms,
            "test_rows": len(test),
        }
        if champ_acc is not None:
            metrics["champion_test_accuracy"] = champ_acc
        with mlflow.start_run(run_id=candidate["run_id"]):
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "promotion": "passed" if ok else "failed",
                    "promotion_reason": reason,
                    "compared_to_champion": str(champ_version),
                }
            )
        print(f"[evaluate_candidate] v{version}: {'PASS' if ok else 'FAIL'} - {reason}")

        if not ok:
            send(
                ML_ALERTS_WEBHOOK_URL,
                adaptive_card(
                    f"ML Alert - model v{version} not promoted",
                    [
                        ("Reason", reason),
                        ("Candidate accuracy", f"{cand_acc:.4f}"),
                        (
                            "Champion accuracy",
                            "n/a" if champ_acc is None else f"{champ_acc:.4f}",
                        ),
                        ("Inference time", f"{inference_ms:.0f} ms"),
                        ("Champion version", str(champ_version)),
                    ],
                    "Warning",
                ),
                "[evaluate_candidate]",
            )
        return ok

    @task
    def promote_to_champion(candidate: dict) -> int:
        """Move the @champion alias and make this version's training stats
        the drift baseline."""
        from airflow.providers.postgres.hooks.postgres import PostgresHook
        from mlflow import MlflowClient

        version = candidate["version"]
        MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION, version)
        PostgresHook(postgres_conn_id=APP_DB_CONN_ID).run(
            "UPDATE training_stats SET is_baseline = (model_version = %s)",
            parameters=(version,),
        )
        print(f"[promote_to_champion] {MODEL_NAME}@{CHAMPION} -> v{version}")
        return version

    @task
    def notify_api_reload(version: int) -> None:
        import requests

        resp = requests.post(f"{API_URL}/reload-model", timeout=60)
        resp.raise_for_status()
        print(f"[notify_api_reload] API reloaded v{version}: {resp.json()}")

    @task
    def archive_data(version: int, data: dict) -> None:
        """Files used by the promoted model leave good_data."""
        os.makedirs(ARCHIVED, exist_ok=True)
        moved = 0
        for path in data["files"]:
            if not os.path.exists(path):
                continue
            dest = os.path.join(ARCHIVED, os.path.basename(path))
            if os.path.exists(dest):  # never overwrite earlier training data
                stem, ext = os.path.splitext(dest)
                dest = f"{stem}_v{version}{ext}"
            shutil.move(path, dest)
            moved += 1
        print(f"[archive_data] v{version}: moved {moved} files to archived_data")

    data = load_data()
    candidate = save_training_stats(train_model(data))
    passed = evaluate_candidate(candidate)
    version = promote_to_champion(candidate)
    passed >> version
    notify_api_reload(version)
    archive_data(version, data)


training_dag()
