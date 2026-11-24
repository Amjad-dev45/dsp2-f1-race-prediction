"""Continuous F1 data feeder.

Every FEEDER_INTERVAL_SECONDS it samples a batch from data/inference.csv,
corrupts it with the course error types (scripts/inject_errors.py) and writes
it to data/raw_data/, so the ingestion -> prediction pipeline and both Grafana
dashboards keep getting fresh data. DRIFT_MODE=true shifts Grid and
ConstructorName to trigger the drift alerts. It pauses while too many files
are waiting in raw_data (FEEDER_MAX_PENDING_FILES).
"""

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from inject_errors import ROW_ERRORS, inject

DEFAULT_SOURCE = Path(os.getenv("FEEDER_SOURCE_CSV", "/app/data/inference.csv"))
DEFAULT_RAW_DIR = Path(os.getenv("FEEDER_RAW_DIR", "/app/data/raw_data"))
DEFAULT_BATCH_SIZE = int(os.getenv("FEEDER_BATCH_SIZE", "20"))
DEFAULT_INTERVAL = int(os.getenv("FEEDER_INTERVAL_SECONDS", "60"))
DEFAULT_DRIFT = os.getenv("DRIFT_MODE", "false").lower() in ("1", "true", "yes")
DEFAULT_MAX_PENDING = int(os.getenv("FEEDER_MAX_PENDING_FILES", "5"))
# per-row probability of each row-level error type, and per-file probability
# of a missing column (schema error); 0 disables
ERROR_PROB = float(os.getenv("FEEDER_ERROR_PROB", "0.015"))
MISSING_COLUMN_PROB = float(os.getenv("FEEDER_MISSING_COLUMN_PROB", "0.005"))


def apply_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Induce covariate shift (Grid + ConstructorName) to trigger drift alerts."""
    drifted = df.copy()
    drifted["Grid"] = rng.integers(16, 21, size=len(drifted))
    backmarkers = ["Haas F1 Team", "Williams", "Alfa Romeo", "AlphaTauri"]
    drifted["ConstructorName"] = rng.choice(backmarkers, size=len(drifted))
    return drifted


def generate_batch(
    pool: pd.DataFrame,
    raw_dir: Path,
    batch_size: int,
    drift: bool,
    rng: np.random.Generator,
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    replace = len(pool) < batch_size
    indices = rng.choice(len(pool), size=batch_size, replace=replace)
    batch = pool.iloc[indices].reset_index(drop=True)

    if drift:
        batch = apply_drift(batch, rng)

    probs = {e: ERROR_PROB for e in ROW_ERRORS}
    probs["missing_column"] = MISSING_COLUMN_PROB
    batch, _ = inject(batch, probs, rng)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "drift" if drift else "live"
    out_path = raw_dir / f"feed_{tag}_{ts}.csv"
    batch.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--drift", action="store_true", default=DEFAULT_DRIFT)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-pending", type=int, default=DEFAULT_MAX_PENDING)
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source pool not found: {args.source}")

    pool = pd.read_csv(args.source)
    rng = np.random.default_rng()

    print(
        f"[feeder] started | source={args.source} ({len(pool)} rows) | "
        f"raw_dir={args.raw_dir} | batch_size={args.batch_size} | "
        f"interval={args.interval}s | drift={args.drift}"
    )

    while True:
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        pending = len(list(args.raw_dir.glob("*.csv")))
        if not args.once and pending >= args.max_pending:
            print(
                f"[feeder] {pending} files pending in {args.raw_dir} "
                f"(>= {args.max_pending}), waiting {args.interval}s..."
            )
        else:
            out_file = generate_batch(
                pool=pool,
                raw_dir=args.raw_dir,
                batch_size=args.batch_size,
                drift=args.drift,
                rng=rng,
            )
            print(
                f"[feeder] wrote {out_file.name} "
                f"({args.batch_size} rows, drift={args.drift})"
            )

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
