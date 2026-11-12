#!/usr/bin/env bash
# Build the wheel that notebooks/kaggle_inference.ipynb installs offline. Nothing is uploaded.
set -euo pipefail

OUT="${1:-dist/kaggle}"
OWNER="${KAGGLE_USERNAME:-your-kaggle-username}"

rm -rf "$OUT"
mkdir -p "$OUT"
python3 -m pip wheel . --no-deps -w "$OUT" -q
cat > "$OUT/dataset-metadata.json" <<JSON
{
  "title": "rogii-wellbore-code",
  "id": "$OWNER/rogii-wellbore-code",
  "licenses": [{"name": "MIT"}]
}
JSON
echo "wheel and metadata in $OUT"
echo "upload with: kaggle datasets create -p $OUT"
