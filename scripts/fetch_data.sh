#!/usr/bin/env bash
# Download the competition files into data/raw (needs the Kaggle CLI and accepted competition rules).
set -euo pipefail

SLUG="rogii-wellbore-geology-prediction"
DEST="${1:-data/raw}"

mkdir -p "$DEST"
kaggle competitions download -c "$SLUG" -p "$DEST"
unzip -q -o "$DEST/$SLUG.zip" -d "$DEST"
rm "$DEST/$SLUG.zip"
echo "competition data in $DEST"
