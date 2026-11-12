#!/usr/bin/env bash
# Download the third-party Kaggle datasets the pipeline reads at inference time into artifacts/.
# Each one is governed by its own Kaggle license; see docs/artifacts.md.
set -euo pipefail

DEST="${1:-artifacts}"
DATASETS=(
  "ravaghi/wellbore-geology-prediction-artifacts"
  "fleongg/rogii-claude-models-pub"
  "pilkwang/rogii-model-package"
  "phongnguyn23021656/koolbox-offline"
)

for ref in "${DATASETS[@]}"; do
  name="${ref#*/}"
  mkdir -p "$DEST/$name"
  kaggle datasets download -d "$ref" -p "$DEST/$name" --unzip
done
echo "artifacts in $DEST"
