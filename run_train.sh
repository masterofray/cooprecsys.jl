#!/usr/bin/env bash
# =============================================================================
# run_train.sh — Train the HybridRecommender pipeline
#
# Usage:
#   chmod +x run_train.sh
#   ./run_train.sh [OPTIONS]
#
# Options (all optional — defaults shown):
#   --data          data/train.csv
#   --model-dir     models
#   --components    50
#   --min-support   0.01
#   --min-conf      0.10
#   --top-k         100
#   --chunk-size    256
#   --threads       auto   (Julia thread count)
#
# Examples:
#   ./run_train.sh
#   ./run_train.sh --data /mnt/data/train.csv --components 80
#   ./run_train.sh --threads 8 --min-support 0.005
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA="data/train.csv"
MODEL_DIR="models"
COMPONENTS=50
MIN_SUPPORT=0.01
MIN_CONF=0.10
TOP_K=100
CHUNK_SIZE=256
THREADS="auto"

# ── Parse overrides ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)         DATA="$2";        shift 2 ;;
    --model-dir)    MODEL_DIR="$2";   shift 2 ;;
    --components)   COMPONENTS="$2";  shift 2 ;;
    --min-support)  MIN_SUPPORT="$2"; shift 2 ;;
    --min-conf)     MIN_CONF="$2";    shift 2 ;;
    --top-k)        TOP_K="$2";       shift 2 ;;
    --chunk-size)   CHUNK_SIZE="$2";  shift 2 ;;
    --threads)      THREADS="$2";     shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; BOLD='\033[1m'; RESET='\033[0m'
echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║         HybridRecommender — TRAINING PIPELINE        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo "  Data        : $DATA"
echo "  Model dir   : $MODEL_DIR"
echo "  Components  : $COMPONENTS"
echo "  Min support : $MIN_SUPPORT"
echo "  Min conf    : $MIN_CONF"
echo "  Top-K CF    : $TOP_K"
echo "  Chunk size  : $CHUNK_SIZE"
echo "  Threads     : $THREADS"
echo ""

# ── Prerequisite checks ────────────────────────────────────────────────────────
if ! command -v julia &> /dev/null; then
  echo -e "${BOLD}ERROR: julia not found in PATH. Install Julia from https://julialang.org${RESET}" >&2
  exit 1
fi

if [ ! -f "$DATA" ]; then
  echo -e "${BOLD}ERROR: data file not found: $DATA${RESET}" >&2
  exit 1
fi

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p "$MODEL_DIR" plots logs

# ── Copy data to expected location if needed ──────────────────────────────────
if [ "$DATA" != "data/train.csv" ]; then
  mkdir -p data
  cp "$DATA" data/train.csv
fi

# ── Set Julia depot and threads ───────────────────────────────────────────────
export JULIA_NUM_THREADS="$THREADS"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$HOME/.julia}"

# ── Install / precompile dependencies ─────────────────────────────────────────
echo -e "${GREEN}[$(date +%T)] Installing Julia dependencies…${RESET}"
julia --project=. -e "
  import Pkg
  Pkg.instantiate()
  Pkg.precompile()
" 2>&1 | tee logs/install.log

# ── Run training ──────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_${TIMESTAMP}.log"

echo -e "${GREEN}[$(date +%T)] Starting training… (log: $LOG_FILE)${RESET}"

julia --threads "$THREADS" --project=. scripts/train.jl \
  --data         "$DATA"        \
  --model-dir    "$MODEL_DIR"   \
  --components   "$COMPONENTS"  \
  --min-support  "$MIN_SUPPORT" \
  --min-conf     "$MIN_CONF"    \
  --top-k        "$TOP_K"       \
  --chunk-size   "$CHUNK_SIZE"  \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
  echo -e "\n${BOLD}${GREEN}✅ Training complete!${RESET}"
  echo "   Models saved to: $MODEL_DIR/"
  echo "   Plots saved to:  plots/"
  echo "   Log saved to:    $LOG_FILE"
else
  echo -e "\n${BOLD}\033[0;31m❌ Training failed (exit code $EXIT_CODE). Check $LOG_FILE${RESET}" >&2
  exit $EXIT_CODE
fi
