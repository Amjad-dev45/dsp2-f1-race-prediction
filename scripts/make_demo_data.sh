#!/usr/bin/env bash
# Prepare demo batches for the ingestion DAG:
#   split the clean incoming pool into N 10-row files, then corrupt EACH file
#   independently (per-row error probabilities -> every batch gets its own
#   mix, duplicates land inside a batch), and make the last batch a
#   schema-error file (missing column -> High criticality alert).
# Usage: ./make_demo_data.sh [n_files] [error_prob]   (defaults 30, 0.05)
set -e
cd "$(dirname "$0")"
N="${1:-30}"
P="${2:-0.05}"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "run ./train_model.sh first (creates the venv)"; exit 1; }

$PY split_data.py ../data/inference.csv ../data/raw_data "$N" --seed "$RANDOM"

for f in ../data/raw_data/batch_*.csv; do
    $PY inject_errors.py "$f" "$f" \
        --p-missing-value "$P" --p-wrong-type "$P" --p-out-of-range "$P" \
        --p-invalid-category "$P" --p-bad-format "$P" --p-duplicate-rows "$P" \
        > /dev/null
done

LAST=$(printf '../data/raw_data/batch_%03d.csv' "$N")
$PY inject_errors.py "$LAST" "$LAST" \
    --p-missing-column 1 --p-missing-value 0 --p-wrong-type 0 \
    --p-out-of-range 0 --p-invalid-category 0 --p-bad-format 0 \
    --p-duplicate-rows 0 > /dev/null
echo "done: $N batches in data/raw_data/ (batch_$(printf '%03d' "$N") has a dropped column)"
