#!/usr/bin/env bash
# Download exactly the 2025 diphoton files used by the H->gamma gamma fixture.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT_DIR="$REPO_ROOT/trex_fitter/inputs"
DATA_DIR="$INPUT_DIR/Data"
MC_DIR="$INPUT_DIR/MC"
STAGING_DIR="$(mktemp -d "$INPUT_DIR/.hyy-download.XXXXXX")"
DATASET="ho22joshua/atlas_opendata"

cleanup() {
  find "$STAGING_DIR" -depth -delete 2>/dev/null || true
}
trap cleanup EXIT

if [[ -e "$DATA_DIR" || -e "$MC_DIR" ]]; then
  echo "Expected empty inputs directory; remove existing Data/ and MC/ first." >&2
  exit 1
fi

mkdir -p "$DATA_DIR" "$MC_DIR"

hf download "$DATASET" \
  --repo-type dataset \
  --local-dir "$STAGING_DIR" \
  --max-workers 16 \
  --include 'GamGam/2025/data/*.root' \
  --include 'GamGam/2025/MC/ODEO_FEB2025_v0_GamGam_mc_343981.*.root' \
  --include 'GamGam/2025/MC/ODEO_FEB2025_v0_GamGam_mc_346214.*.root' \
  --include 'GamGam/2025/MC/ODEO_FEB2025_v0_GamGam_mc_345317.*.root' \
  --include 'GamGam/2025/MC/ODEO_FEB2025_v0_GamGam_mc_345318.*.root' \
  --include 'GamGam/2025/MC/ODEO_FEB2025_v0_GamGam_mc_345319.*.root' \
  --include 'GamGam/2025/MC/ODEO_FEB2025_v0_GamGam_mc_346525.*.root' \
  --include 'GamGam/2025/MC/ODEO_FEB2025_v0_GamGam_mc_364350.*.root' \
  --include 'GamGam/2025/MC/ODEO_FEB2025_v0_GamGam_mc_364351.*.root' \
  --include 'GamGam/2025/MC/ODEO_FEB2025_v0_GamGam_mc_364352.*.root' \
  --include 'GamGam/2025/MC/ODEO_FEB2025_v0_GamGam_mc_364353.*.root'

find "$STAGING_DIR/GamGam/2025/data" -maxdepth 1 -type f -name '*.root' -exec mv -t "$DATA_DIR" {} +
find "$STAGING_DIR/GamGam/2025/MC" -maxdepth 1 -type f -name '*.root' -exec mv -t "$MC_DIR" {} +

data_count=$(find "$DATA_DIR" -maxdepth 1 -type f -name '*.root' | wc -l)
mc_count=$(find "$MC_DIR" -maxdepth 1 -type f -name '*.root' | wc -l)
if [[ "$data_count" != 16 || "$mc_count" != 10 ]]; then
  echo "Expected 16 data files and 10 MC files; got $data_count and $mc_count." >&2
  exit 1
fi

echo "Downloaded H->gamma gamma inputs: $data_count data files and $mc_count MC files."
