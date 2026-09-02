#!/usr/bin/env bash
# Download the training Parquet files directly from Hugging Face.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_DIR="$REPO_ROOT/data/reference"

mkdir -p "$REFERENCE_DIR/hep-config-sft" "$REFERENCE_DIR/hep-signature-backgrounds"

hf download ho22joshua/hep-config-sft \
  --repo-type dataset \
  --local-dir "$REFERENCE_DIR/hep-config-sft" \
  --max-workers 8 \
  --include 'data/*.parquet'

hf download ho22joshua/hep-signature-backgrounds \
  --repo-type dataset \
  --local-dir "$REFERENCE_DIR/hep-signature-backgrounds" \
  --max-workers 8 \
  --include 'hep_sft/*.parquet'

echo "Reference training datasets are ready in data/reference/."
