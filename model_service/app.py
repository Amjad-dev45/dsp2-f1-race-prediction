import hashlib
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database.database import Base, get_engine, get_session_local
from database.models import PredictionLog

engine = get_engine()


def get_db():
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "f1-top10")
CHAMPION = "champion"
ml = {}


def file_model_version(path: str) -> str:
    """Version id of a model file: MODEL_VERSION env var if set, otherwise
    the first 8 hex chars of the file's sha256 (changes whenever the model
    file changes). Defense 2 replaces this with the MLflow registry version."""
    if os.getenv("MODEL_VERSION"):
        return os.environ["MODEL_VERSION"]
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return f"file-{digest.hexdigest()[:8]}"


def load_champion():
    """(model, version) of <MLFLOW_MODEL_NAME>@champion from the registry."""
    import mlflow
    from mlflow import MlflowClient

    mv = MlflowClient().get_model_version_by_alias(MLFLOW_MODEL_NAME, CHAMPION)
    model = mlflow.sklearn.load_model(f"models:/{MLFLOW_MODEL_NAME}/{mv.version}")
    return model, str(mv.version)


def load_startup_model():
    """Champion from MLflow when configured, else the committed model file
    (fresh stack before the first training run, or tests)."""
    if os.getenv("MLFLOW_TRACKING_URI"):
        try:
            return load_champion()
        except Exception as e:
            print(f"[startup] no MLflow champion ({e}), using {MODEL_PATH}")
    return joblib.load(MODEL_PATH), file_model_version(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)  # create the predictions table if missing
    ml["model"], ml["version"] = load_startup_model()  # load once at startup
    yield
    ml.clear()


app = FastAPI(title="F1 Prediction API", lifespan=lifespan)


class F1Features(BaseModel):
    """One sample: the six model features, named as in the dataset CSVs.
    Unknown categorical values are allowed (the encoder ignores them)."""

    Season: int = Field(ge=1950, le=2100)
    Round: int = Field(ge=1, le=30)
    CircuitID: str = Field(min_length=1)
    DriverID: str = Field(min_length=1)
    ConstructorName: str = Field(min_length=1)
    Grid: int = Field(ge=0, le=40)


class PredictRequest(BaseModel):
    features: list[F1Features]
    source: Literal["webapp", "scheduled"] = "webapp"


class PredictionOut(BaseModel):
    id: int
    season: int
    round: int
    circuit_id: str
    driver_id: str
    constructor_name: str
    grid: int
    prediction: bool
    probability: float | None
    source: str
    model_version: str | None
    created_at: datetime


def to_dict(r):
    return {
        "id": r.id,
        "season": r.season,
        "round": r.round,
        "circuit_id": r.circuit_id,
        "driver_id": r.driver_id,
        "constructor_name": r.constructor_name,
        "grid": r.grid,
        "prediction": r.prediction,
        "probability": r.probability,
        "source": r.source,
        "model_version": r.model_version,
        "created_at": r.created_at,
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reload-model")
def reload_model():
    """Called by the training DAG after a promotion: serve the new champion.
    On failure the current model keeps serving."""
    try:
        model, version = load_champion()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"reload failed: {e}")
    previous = ml.get("version")
    ml["model"], ml["version"] = model, version  # swap only after a full load
    return {"status": "reloaded", "previous_version": previous, "version": version}


@app.post("/predict", response_model=list[PredictionOut])
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    if not req.features:
        raise HTTPException(status_code=400, detail="features cannot be empty")
    df = pd.DataFrame([f.model_dump() for f in req.features])
    try:
        # one snapshot per request: a /reload-model in the middle cannot mix
        # two models (or a model and the wrong version) into one answer
        model, version = ml["model"], ml.get("version")
        preds = model.predict(df)
        probas = model.predict_proba(df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"prediction failed: {e}")

    rows = []
    for feat, p, prob in zip(req.features, preds, probas):
        row = PredictionLog(
            season=feat.Season,
            round=feat.Round,
            circuit_id=feat.CircuitID,
            driver_id=feat.DriverID,
            constructor_name=feat.ConstructorName,
            grid=feat.Grid,
            prediction=bool(p),
            probability=float(prob),
            source=req.source,
            model_version=version,
        )
        db.add(row)
        rows.append(row)
    db.commit()
    for row in rows:
        db.refresh(row)
    return [to_dict(r) for r in rows]


@app.get("/past-predictions", response_model=list[PredictionOut])
def past_predictions(
    start: date | None = None,
    end: date | None = None,
    source: str = Query("all", pattern="^(webapp|scheduled|all)$"),
    limit: int | None = Query(None, ge=1, le=10000),
    db: Session = Depends(get_db),
):
    if start and end and start > end:
        raise HTTPException(status_code=400, detail="start must be before end")

    q = db.query(PredictionLog)
    if source != "all":
        q = q.filter(PredictionLog.source == source)
    if start:
        q = q.filter(PredictionLog.created_at >= start)
    if end:
        q = q.filter(PredictionLog.created_at < end + timedelta(days=1))
    q = q.order_by(PredictionLog.created_at.desc())
    if limit:
        q = q.limit(limit)
    return [to_dict(r) for r in q.all()]


@app.get("/options")
def options():
    prep = ml["model"].named_steps["preprocessor"]
    encoder = prep.named_transformers_["cat"]
    cats = dict(zip(encoder.feature_names_in_, encoder.categories_))
    return {col: sorted(v.tolist()) for col, v in cats.items()}
