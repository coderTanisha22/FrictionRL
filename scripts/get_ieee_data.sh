#!/usr/bin/env bash
# Download Kaggle's IEEE-CIS Fraud Detection data and build data/ieee_features.npz.
#
#   bash scripts/get_ieee_data.sh                     # default rate: Rs 85 per USD
#   bash scripts/get_ieee_data.sh --inr-per-usd 84.2  # any extra options go to frictionrl.data_ieee
#
# Needs: a Kaggle account, an API token (~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY),
# and the competition rules accepted at https://www.kaggle.com/c/ieee-fraud-detection/rules
# Run it from the repository root. Safe to re-run: finished steps are skipped.
set -euo pipefail

RAW=data/ieee-fraud-detection
OUT=data/ieee_features.npz
ZIP=$RAW/ieee-fraud-detection.zip
PY=${PYTHON:-$(command -v python || command -v python3)}

fail() { echo "error: $*" >&2; exit 1; }

[ -f frictionrl/data_ieee.py ] || fail "run this from the FrictionRL repository root"
[ -n "$PY" ] || fail "no python found (set PYTHON=/path/to/python)"

# 1. Kaggle CLI and credentials
command -v kaggle >/dev/null || fail "kaggle CLI not found. Install it with: pip install kaggle"
if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_USERNAME:-}${KAGGLE_KEY:-}" ]; then
  fail "no Kaggle API token found.
  1. kaggle.com -> your profile -> Settings -> API -> Create New Token (downloads kaggle.json)
  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
fi

# 2. Download (about 118 MB)
mkdir -p "$RAW"
if [ -f "$RAW/train_transaction.csv" ] && [ -f "$RAW/train_identity.csv" ]; then
  echo "[1/3] raw CSVs already present, skipping download"
else
  if [ ! -f "$ZIP" ]; then
    echo "[1/3] downloading ieee-fraud-detection from Kaggle..."
    kaggle competitions download -c ieee-fraud-detection -p "$RAW" || fail "download failed.
  If the error is 403 Forbidden, accept the competition rules first:
  https://www.kaggle.com/c/ieee-fraud-detection/rules"
  fi
  echo "[1/3] unzipping the two files we need..."
  unzip -o -q "$ZIP" train_transaction.csv train_identity.csv -d "$RAW"
fi

# 3. Build the feature pools
echo "[2/3] building $OUT ..."
"$PY" -m frictionrl.data_ieee --raw "$RAW" --out "$OUT" "$@"

# 4. Sanity check + comparison figure
echo "[3/3] checking the data ..."
"$PY" scripts/check_ieee.py --data "$OUT"

cat <<'EOF'

Done. Next:
  python scripts/run_all.py --source ieee --out results_ieee      # full run, about 35-45 min
  python scripts/make_figures.py --results results_ieee
EOF
