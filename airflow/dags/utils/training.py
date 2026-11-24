"""Pure helpers for the training DAG (no Airflow / MLflow imports).

Data used for (re)training:
    data/training_data.csv     labelled seed, seasons 2000-2021
    data/archived_data/*.csv   ingested rows already used by a promoted model
    data/good_data/*.csv       new ingested rows since the last promotion
Ingested rows carry no label: their ground truth (race results) comes from
data/inference_labels.csv, joined on Season + Round + DriverID.

Test set (fair candidate vs champion comparison): a FIXED slice of the
labelled 2022-2024 pool, chosen by a hash of the row key. It never changes
between runs and its rows are excluded from every training set, so both
models are scored on the same data that neither was trained on.
"""

import zlib

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = ["Season", "Round", "Grid"]
CATEGORICAL_FEATURES = ["CircuitID", "DriverID", "ConstructorName"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "top10"
LABEL_KEYS = ["Season", "Round", "DriverID"]
TEST_SHARE = 5  # 1 row in 5 of the labelled pool is held out for testing

# features summarised for the drift dashboard baseline
STATS_NUMERIC = ["Grid", "Round"]
STATS_CATEGORICAL = ["ConstructorName", "CircuitID"]
STATS_QUANTILES = {"p25": 0.25, "p50": 0.5, "p75": 0.75}


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Same model as the original artifacts/model.pkl. The step names
    'preprocessor' and 'cat' are read by the API's /options endpoint."""
    return Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore"),
                            CATEGORICAL_FEATURES,
                        )
                    ],
                    remainder="passthrough",
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=5,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Feature columns with clean dtypes: numerics as int (ingested files can
    hold 2022.0 after cleaning, which would change the row key), rows with
    unusable numerics dropped."""
    df = df[FEATURES].copy()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=NUMERIC_FEATURES)
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(int)
    return df


def is_test_row(df: pd.DataFrame) -> pd.Series:
    """Deterministic test-set membership from the row key (stable across
    runs and machines, unlike Python's salted hash())."""
    keys = df[LABEL_KEYS].astype(str).agg("|".join, axis=1)
    return keys.map(lambda k: zlib.crc32(k.encode()) % TEST_SHARE == 0)


def holdout_set(pool: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The fixed held-out test set: labelled pool rows selected by is_test_row."""
    df = normalise(pool).merge(labels, on=LABEL_KEYS, how="inner")
    return df[is_test_row(df)].reset_index(drop=True)


def label_rows(rows: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach ground truth to ingested rows and drop test-set rows (they must
    never be trained on). Unlabelled rows are dropped."""
    if rows.empty:
        return pd.DataFrame(columns=FEATURES + [TARGET])
    df = normalise(rows).merge(labels, on=LABEL_KEYS, how="inner")
    return df[~is_test_row(df)].reset_index(drop=True)


def training_set(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Base training data plus the new labelled rows, with an int target.
    An empty `new` frame has object columns, and concatenating it would
    turn the target into object dtype, which sklearn rejects."""
    parts = [base] if new.empty else [base, new]
    train = pd.concat(parts, ignore_index=True)
    train[TARGET] = train[TARGET].astype(int)
    return train


def training_stats(df: pd.DataFrame) -> list[dict]:
    """Feature statistics of a training set, in the long format of the
    training_stats table: one row per (feature, stat[, category])."""
    rows = [
        {
            "feature": "_all",
            "stat": "row_count",
            "category": None,
            "value": float(len(df)),
        }
    ]
    for col in STATS_NUMERIC:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        values = {"mean": s.mean(), "std": s.std(), "min": s.min(), "max": s.max()}
        values.update({k: s.quantile(q) for k, q in STATS_QUANTILES.items()})
        rows += [
            {"feature": col, "stat": k, "category": None, "value": float(v)}
            for k, v in values.items()
        ]
    for col in STATS_CATEGORICAL:
        shares = df[col].value_counts(normalize=True)
        rows += [
            {"feature": col, "stat": "share", "category": str(c), "value": float(v)}
            for c, v in shares.items()
        ]
    if TARGET in df:
        rows.append(
            {
                "feature": TARGET,
                "stat": "positive_rate",
                "category": None,
                "value": float(df[TARGET].mean()),
            }
        )
    return rows


def promotion_decision(
    candidate_metric: float,
    champion_metric: float | None,
    inference_ms: float,
    max_inference_ms: float,
) -> tuple[bool, str]:
    """Promote when candidate >= champion on the primary metric AND the
    inference time is under the threshold. No champion yet -> only the
    inference time is checked."""
    if inference_ms >= max_inference_ms:
        return False, (
            f"inference time {inference_ms:.0f} ms >= threshold "
            f"{max_inference_ms:.0f} ms"
        )
    if champion_metric is None:
        return True, "no champion yet (first model)"
    if candidate_metric < champion_metric:
        return False, (
            f"candidate accuracy {candidate_metric:.4f} < champion "
            f"{champion_metric:.4f}"
        )
    return True, (
        f"candidate accuracy {candidate_metric:.4f} >= champion "
        f"{champion_metric:.4f}"
    )
