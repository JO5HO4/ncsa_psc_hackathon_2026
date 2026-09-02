#!/usr/bin/env bash
# Download selected ATLAS Open Data skims directly from Hugging Face.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${ATLAS_OPENDATA_DIR:-$REPO_ROOT/data/atlas_opendata}"
DATASET="ho22joshua/atlas_opendata"

usage() {
  cat <<'EOF'
Usage: bash data/fetch_atlas_opendata.sh SKIM [SKIM ...]

Download one or more complete skims into data/atlas_opendata/ using hf download.
Available skims:
  1lepMET30  2J2LMET  2to4lep  4lep  GamGam  GamGam-2020  all

Examples:
  bash data/fetch_atlas_opendata.sh 2to4lep
  bash data/fetch_atlas_opendata.sh 1lepMET30 2J2LMET
  bash data/fetch_atlas_opendata.sh GamGam
  bash data/fetch_atlas_opendata.sh all

GamGam means the 2025 ODEO collection. GamGam-2020 is the five-file legacy
MC subset. The all option downloads every ROOT file in the Hub dataset.
Set ATLAS_OPENDATA_DIR to use a different local destination.
EOF
}

if [[ "$#" -eq 0 ]]; then
  usage >&2
  exit 2
fi

patterns=()
for skim in "$@"; do
  case "$skim" in
    1lepMET30|2J2LMET|2to4lep|4lep)
      patterns+=("$skim/*.root")
      ;;
    GamGam)
      patterns+=("GamGam/2025/*.root")
      ;;
    GamGam-2020)
      patterns+=("GamGam/2020/*.root")
      ;;
    all)
      patterns=('*.root')
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown skim: $skim" >&2
      usage >&2
      exit 2
      ;;
  esac
done

args=()
for pattern in "${patterns[@]}"; do
  args+=(--include "$pattern")
done

mkdir -p "$DESTINATION"
hf download "$DATASET" \
  --repo-type dataset \
  --local-dir "$DESTINATION" \
  --max-workers 16 \
  "${args[@]}"

echo "Downloaded selected Open Data skims to: $DESTINATION"
