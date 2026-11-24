# DSP F1 Race Prediction

An end-to-end MLOps pipeline that predicts whether a Formula 1 driver
finishes in the top 10, built for the EPITA Data Science in Production
course. A trained model is served through an API and a web app, while an
Airflow-orchestrated pipeline continuously ingests new race data,
validates it with Great Expectations, alerts on data quality problems,
and makes scheduled predictions on the clean rows.

## Architecture

Everything runs from one Docker Compose file:

| Service | Container | Purpose |
|---|---|---|
| PostgreSQL | f1_postgres | One instance, two databases: app data (predictions, ingestion statistics) and Airflow metadata |
| FastAPI | f1_api | Model service: /predict, /past-predictions, /options, /health, /reload-model |
| Streamlit | f1_webapp | Web UI: single and batch predictions, past-prediction browsing |
| Grafana | f1_grafana | Data-quality and drift dashboards with provisioned alerts |
| Nginx | f1_nginx | Browser entry point and Great Expectations report hosting |
| Airflow (x4) | airflow-init, airflow-api-server, airflow-dag-processor, airflow-scheduler | Orchestrates the ingestion and prediction DAGs |
| MLflow | f1_mlflow | Tracking server + model registry (`f1-top10`, `@champion` alias); SQLite metadata and artifacts on named volumes |

The webapp only talks to the API; the API and the DAGs use the shared
SQLAlchemy models in `database/`; all connection settings come from
environment variables.

## Prerequisites

- Git
- Docker with Compose v2 (Docker Desktop on macOS/Windows, Docker Engine
  on Linux), running
- Python 3.10+ (only for the data/model scripts, not for the stack)

Windows only, once, BEFORE cloning:

```
git config --global core.autocrlf false
```

## 1. Get the code

```
git clone https://github.com/ribbzz/dsp2-f1-race-prediction.git
cd dsp2-f1-race-prediction
```

## 2. Configure the environment

```
cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
```

Open `.env` and fill in every value:

| Variable | What to put |
|---|---|
| POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD | App database name and credentials — any values, created on first boot |
| AIRFLOW_DB_PASSWORD | Password for Airflow's database user. Letters and digits ONLY |
| AIRFLOW_JWT_SECRET / AIRFLOW_SECRET_KEY | Generate with the python one-liners in the comments of `.env.example` |
| AIRFLOW_UID | Linux: output of `id -u`. macOS/Windows: leave empty |
| TEAMS_WEBHOOK_URL | Teams **Workflows** webhook of the `Data Quality Alerts` channel (ingestion alerts); may stay empty (alerts are then written to the task log) |
| ML_ALERTS_WEBHOOK_URL | Teams Workflows webhook of the `ML Alerts` channel (failed model promotion, drift); may stay empty |
| MLFLOW_PORT | Host port of the MLflow UI (default 5000; on macOS use e.g. 5001, AirPlay uses 5000) |
| MLFLOW_MODEL_NAME | Registered model name (default `f1-top10`) |

`.env` is gitignored — real values never go into the repository.

## 3. Start the stack

```
docker compose up -d --build
```

The first run builds all images (several minutes). On a fresh volume,
postgres automatically creates both databases.

If this machine ran an older version of the project before, reset the
database volume once so it is recreated with your current `.env`:

```
docker compose down -v
docker compose up -d --build
```

## 4. Verify

`docker compose ps` should show:

| Container | State |
|---|---|
| f1_postgres | healthy |
| f1_api | healthy |
| f1_webapp | running |
| airflow-init | Exited (0) — one-shot migration job, exiting is success |
| airflow-api-server | healthy |
| airflow-scheduler, airflow-dag-processor | running |

| Service | URL |
|---|---|
| Web app | http://localhost:8501 |
| API (Swagger docs) | http://localhost:8000/docs |
| Airflow UI | http://localhost:8021 |
| MLflow UI | http://localhost:5000 (`MLFLOW_PORT`) |

Airflow login: user `admin`, generated password printed in the log:

```
docker compose logs airflow-api-server | grep -i password
```

(Windows PowerShell: `docker compose logs airflow-api-server | Select-String password`)

## 5. Prepare data for the pipeline

The trained model (`artifacts/model.pkl`) and the data pools
(`data/training_data.csv`, `data/inference.csv`) are committed, so the
API and webapp work immediately. What a fresh clone lacks is ingestion
input: batch files in `data/raw_data/`.

From `scripts/` (macOS/Linux, or Git Bash on Windows):

```
cd scripts
./get_data.sh            # download the Kaggle dataset
./train_model.sh         # create the venv (writes model_new.pkl, never overwrites the live model)
./make_demo_data.sh 30   # 30 x 10-row batches into data/raw_data/, with injected errors
./make_defense_files.sh  # the 5 defense demo files in data/defense_files/
cd ..
```

`make_demo_data.sh` splits the clean pool into 10-row files and corrupts
each file independently (every error type hits each row with probability
0.05 by default; pass a second argument to change it), so every batch has
its own mix of errors. The last batch has a dropped column (schema error).

`make_defense_files.sh` creates the 5 files the defense instructions ask
for, outside `raw_data`: `1_all_errors.csv`, `2_clean.csv`, `3_clean.csv`,
`4_clean.csv`, `5_mixed.csv`. Copy them into `data/raw_data/` during the demo.

Details on every script, error types and native Windows equivalents:
`scripts/README.md`.

## 6. Run the pipeline

Both DAGs are scheduled. New DAGs start **paused**, so unpause them once
in the Airflow UI (http://localhost:8021) or from the command line:

```
docker compose exec airflow-scheduler airflow dags unpause ingestion_dag
docker compose exec airflow-scheduler airflow dags unpause prediction_dag
```

1. `ingestion_dag` runs **every minute**. Each run picks one random file
   from `data/raw_data/` (and deletes it there), validates it with Great
   Expectations, saves statistics to the database, sends a Teams alert on
   medium/high criticality, and splits the rows into `data/good_data/`
   and/or `data/bad_data/`. With no file in `raw_data` the run skips.
2. `prediction_dag` runs **every 2 minutes**. It collects the files added
   to `data/good_data/` since its last run, sends all their rows to
   `POST /predict` in one call with `source="scheduled"`, and the API
   stores the predictions. No new files = the whole run is skipped
   (expected, not an error). It never moves files out of `good_data`.

3. `training_dag` runs **every hour** (unpause it like the others):
   `load_data → train_model → save_training_stats → evaluate_candidate →
   promote_to_champion → notify_api_reload + archive_data`.
   - skips unless `good_data` holds at least `TRAINING_MIN_NEW_ROWS`
     (default 100) new **labelled** rows. Labels come from
     `data/inference_labels.csv` (race results arriving after the race).
     With no champion yet it trains anyway (first model)
   - trains on `training_data.csv` + `archived_data` + the new rows, and
     logs and registers the model in MLflow as `f1-top10`
   - stores the training-set feature stats in `training_stats`. Only the
     champion's rows have `is_baseline = true` (drift dashboard baseline)
   - compares candidate vs `@champion` on a **fixed held-out test set**
     (1 in 5 rows of the labelled 2022–2024 pool, chosen by a hash of the
     row key and never trained on). It promotes if accuracy is >= the
     champion's and the inference time is < `TRAINING_MAX_INFERENCE_MS`
     (default 1000 ms). Otherwise it logs the reason in MLflow and alerts
     the ML Alerts channel
   - on promotion it moves `@champion`, calls the API's `POST /reload-model`
     and moves the used files from `good_data` to `archived_data`

4. `drift_monitor_dag` runs **every 10 minutes** and starts `training_dag`
   early when the data drifts (unpause it too). Retraining starts only if
   **both** conditions hold:
   - **drift** over the last 15 minutes of `prediction_logs`, against the
     champion's baseline in `training_stats` (`is_baseline`): Grid z-score
     > 1.0 **or** top-10 rate change > 0.2. These are the same two metrics,
     queries and thresholds as the Grafana drift alerts
   - **enough new data**: at least `TRAINING_MIN_NEW_ROWS` (default 100) new
     labelled rows in `good_data`, counted exactly like `training_dag`

   A **cooldown** (1 hour) stops it from triggering again while the drift
   lasts. A trigger runs `training_dag` with `conf={"trigger": "drift"}` and
   sends a card to the ML Alerts channel. The decision logic is in
   `airflow/dags/utils/drift.py` (unit-tested in `tests/test_drift.py`).

   To see it work: set `DRIFT_MODE=true` in `.env`, run
   `docker compose up -d feeder`, and wait about 15 minutes for the feeder's
   shifted data to fill the window. Then check the `drift_monitor_dag` log
   (`[decide] retrain=True: ...`) and the new `training_dag` run.

Trigger a run immediately with the ▶ button in the UI, or
`docker compose exec airflow-scheduler airflow dags trigger ingestion_dag`.

The Great Expectations Data Docs report for each validation run is
generated under
`airflow/great_expectations/gx/uncommitted/data_docs/local_site/` — open
the newest HTML file under `validations/` in a browser. The Teams alert
shows that report's path relative to `local_site/`.

If the validation rules change, rebuild the expectation suite and
checkpoint (allowed categorical values come from `data/training_data.csv`
and `data/inference.csv`):

```
docker compose exec airflow-scheduler python /opt/airflow/dags/utils/build_expectation_suite.py
```

## 7. See the results

- Web app, Prediction page: single prediction via the form (dropdowns
  offer the values known to the model, new values are accepted; the
  result is shown as a table with the features and the model version), or
  batch prediction via CSV upload with columns
  `Season, Round, Grid, CircuitID, DriverID, ConstructorName`.
- Web app, Past Predictions page: filter by date range and source
  (`webapp` / `scheduled`).
- API: interactive Swagger at http://localhost:8000/docs. `/predict`
  validates every sample (wrong types or out-of-range values → 422) and
  each stored prediction records the `model_version` that produced it.

Or straight from the database:

```
docker compose exec postgres psql -U <POSTGRES_USER> -d <POSTGRES_DB> \
  -c "SELECT file_name, total_rows, valid_rows, invalid_rows, criticality FROM ingestion_stats ORDER BY ingested_at DESC LIMIT 10;"
```

## Tests and linting

CI (GitHub Actions) runs black, isort, flake8 and pytest on every push
and pull request. Locally:

```
python3 -m venv .venv-dev
.venv-dev/bin/pip install -r requirements-dev.txt
.venv-dev/bin/black . && .venv-dev/bin/isort . && .venv-dev/bin/flake8
DATABASE_URL=sqlite:///./test.db .venv-dev/bin/pytest -q
```

## Releases (CD)

`.github/workflows/release.yml` turns every push to `main` that passes CI
into a GitHub release: it tags the commit with the next version
(`v1.0.0` first, then a patch bump: `v1.0.1`, ...) and generates the
release notes from the merged pull requests. A commit that already has a
version tag is not released twice. For a minor or major release, run the
workflow by hand from the Actions tab on `main` and pick the bump.

## Grafana, nginx, and Data Docs

Set GRAFANA_ADMIN_PASSWORD and TEAMS_WEBHOOK_URL in .env before starting the stack. Use a long, private Grafana password. PUBLIC_BASE_URL should be http://localhost:8080 for local access. On a shared demo machine, replace localhost with a host name or IP that teammates can reach.

Nginx is the browser entry point on port 8080:

- / opens the Streamlit app.
- FastAPI documentation remains available directly at http://localhost:8000/docs.
- /grafana/ opens Grafana. The direct Grafana port is configurable with GRAFANA_PORT.
- /data-docs/ serves the Great Expectations validation reports generated by ingestion.

Grafana provisions the F1 Data Quality dashboard from ingestion_stats. It shows invalid-row rate, schema failures, row volume, validation error categories, run criticality, and recent ingestion batches. Provisioned Teams alerts fire for a medium invalid-row rate (10–50%), a high rate (>50%), or a schema failure. If TEAMS_WEBHOOK_URL is empty, the alerts cannot be delivered to Teams.
## Stop / reset

```
docker compose down        # stop, keep all data
docker compose down -v     # stop and wipe both databases
```

`scripts/clean.sh` removes everything the scripts generate
(`--all` also removes the venv).

## Project structure

```
airflow/            Airflow image, DAGs (ingestion, prediction), Great Expectations project
artifacts/          trained model (model.pkl)
data/               data pools + raw_data / good_data / bad_data folders
database/           shared SQLAlchemy models and connection handling
model_service/      FastAPI model service
scripts/            data download, model training, error injection, demo data
tests/              pytest suite (runs against SQLite in CI)
webapp/             Streamlit application
docker-compose.yml  the whole stack
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| "password authentication failed for user airflow" | `AIRFLOW_DB_PASSWORD` has special characters, or the volume predates your `.env`: fix the password, then `docker compose down -v && docker compose up -d --build` |
| ingestion_dag PermissionError on data files (Linux) | set `AIRFLOW_UID=$(id -u)` in `.env`, then `docker compose up -d --build` |
| Every file shows as modified after cloning (Windows) | clone happened before `git config --global core.autocrlf false`: set it, delete the clone, re-clone |
| Port already in use | free ports 8000 / 8501 / 8021 (postgres is not published on the host) |
| API container unhealthy after pulling new code | the `prediction_logs` table gained a `model_version` column: reset the volume once with `docker compose down -v` |
| Webapp cannot reach the API | `docker compose ps` must show f1_api healthy; inside the network the webapp uses `http://api:8000` |
