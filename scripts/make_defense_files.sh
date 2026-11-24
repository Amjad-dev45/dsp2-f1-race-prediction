#!/usr/bin/env bash
# The 5 demo files the defense instructions ask for, kept OUTSIDE raw_data
# (copy them into data/raw_data/ one by one during the demo):
#   1_all_errors.csv            every row invalid  -> high alert, all to bad_data
#   2_clean.csv 3_clean.csv 4_clean.csv   valid    -> good_data, then predictions
#   5_mixed.csv                 some rows invalid  -> file split good/bad
# Usage: ./make_defense_files.sh [output_dir]   (default ../data/defense_files)
set -e
cd "$(dirname "$0")"
OUT="${1:-../data/defense_files}"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "run ./train_model.sh first (creates the venv)"; exit 1; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

$PY split_data.py ../data/inference.csv "$TMP" 5 --seed "$RANDOM" > /dev/null
mkdir -p "$OUT"
none="--p-missing-value 0 --p-wrong-type 0 --p-out-of-range 0 --p-invalid-category 0 --p-bad-format 0 --p-duplicate-rows 0"

# shellcheck disable=SC2086
$PY inject_errors.py "$TMP/batch_001.csv" "$OUT/1_all_errors.csv" $none --p-invalid-category 1 > /dev/null
cp "$TMP/batch_002.csv" "$OUT/2_clean.csv"
cp "$TMP/batch_003.csv" "$OUT/3_clean.csv"
cp "$TMP/batch_004.csv" "$OUT/4_clean.csv"
# shellcheck disable=SC2086
$PY inject_errors.py "$TMP/batch_005.csv" "$OUT/5_mixed.csv" $none \
    --p-wrong-type 0.2 --p-out-of-range 0.2 --p-bad-format 0.2 --seed 5 > /dev/null
echo "done: 5 defense files in $OUT"
