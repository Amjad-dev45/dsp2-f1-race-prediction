from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
)
from sqlalchemy.sql import false, func

from database.database import Base


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    season = Column(Integer, nullable=False)

    round = Column(Integer, nullable=False)

    circuit_id = Column(String(100), nullable=False)

    driver_id = Column(String(100), nullable=False)

    constructor_name = Column(String(100), nullable=False)

    grid = Column(Integer, nullable=False)

    prediction = Column(Boolean, nullable=False)

    source = Column(String(20), nullable=False)

    probability = Column(Float)

    # which model produced the prediction (dashboards compare before/after
    # retraining); file hash in Defense 1, MLflow registry version later
    model_version = Column(String(50), index=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class IngestionStat(Base):
    __tablename__ = "ingestion_stats"

    id = Column(Integer, primary_key=True, index=True)

    file_name = Column(String(255), nullable=False)

    ingested_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    total_rows = Column(Integer)

    valid_rows = Column(Integer)

    invalid_rows = Column(Integer)

    criticality = Column(String(10))

    null_count = Column(Integer)

    type_error_count = Column(Integer)

    range_count = Column(Integer)

    consistency_count = Column(Integer)

    format_count = Column(Integer)

    duplicate_count = Column(Integer)

    schema_error = Column(Boolean, default=False)


class TrainingStat(Base):
    """Feature statistics of each trained model's training set: the drift
    dashboard's baseline. Long format, one row per (feature, stat[, category]):

        feature='Grid'            stat in mean|std|min|max|p25|p50|p75
        feature='Round'           same numeric stats
        feature='ConstructorName' stat='share', category=<team>, value=0..1
        feature='CircuitID'       stat='share', category=<circuit>
        feature='top10'           stat='positive_rate' (label rate)
        feature='_all'            stat='row_count'

    Written by the training DAG for every candidate; only the current
    champion's rows have is_baseline = true (set on promotion), so
    dashboards compare serving data with:  WHERE is_baseline
    """

    __tablename__ = "training_stats"

    id = Column(Integer, primary_key=True, index=True)

    model_version = Column(Integer, nullable=False, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    feature = Column(String(50), nullable=False)

    stat = Column(String(30), nullable=False)

    category = Column(String(100))

    value = Column(Float, nullable=False)

    is_baseline = Column(
        Boolean, nullable=False, default=False, server_default=false(), index=True
    )
