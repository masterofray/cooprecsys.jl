#!/usr/bin/env bash
# =============================================================================
# run_predict.sh — Run predictions or start the HTTP API server
#
# Usage:
#   chmod +x run_predict.sh
#   ./run_predict.sh [MODE] [OPTIONS]
#
# Modes:
#   single   — Predict for one user (default)
#   batch    — Batch predict from a JSON file
#   server   — Start HTTP API server (persistent, suitable for cron restart)
#
# Single mode options:
#   --user-id   <id>        User ID to predict for   (required)
#   --cart      <ids>       Comma-separated cart IDs  (optional)
#   --top-k     <n>         Recommendations to return (default: 20)
#   --output    <file>      JSON output file           (default: stdout)
#
# Batch mode options:
#   --input     <file>      JSON file with {requests:[...]} (required)
#   --output    <file>      Output JSON file
#   --top-k     <n>         Recommendations per user   (default: 20)
#
# Server mode options:
#   --host      <host>      Bind host  (default: 0.0.0.0)
#   --port      <port>      Bind port  (default: 8080)
#
# Common options:
#   --svd-model <path>      models/svd_model.jls
#   --fp-model  <path>      models/fp_model.jls
#   --threads   <n|auto>    Julia threads (default: auto)
#
# Cron examples (crontab -e):
#   # Nightly batch recommendations at 2 AM
#   0 2 * * * /path/to/run_predict.sh batch --input /data/nightly_requests.json --output /data/recommendations.json
#
#   # Keep server alive — restart if it crashes (every 5 min)
#   */5 * * * * pgrep -f "predict.jl.*server" || /path/to/run_predict.sh server --port 8080 &
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODE="single"
SVD_MODEL="models/svd_model.jls"
FP_MODEL="models/fp_model.jls"
USER_ID=""
CART=""
INPUT=""
OUTPUT=""
TOP_K=20
HOST="0.0.0.0"
PORT=8080
THREADS="auto"

# ── Parse positional mode first ───────────────────────────────────────────────
if [[ $# -gt 0 ]] && [[ "$1" =~ ^(single|batch|server)$ ]]; then
  MODE="$1"; shift
fi

# ── Parse options ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --svd-model)   SVD_MODEL="$2"; shift 2 ;;
    --fp-model)    FP_MODEL="$2";  shift 2 ;;
    --user-id)     USER_ID="$2";   shift 2 ;;
    --cart)        CART="$2";      shift 2 ;;
    --input)       INPUT="$2";     shift 2 ;;
    --output)      OUTPUT="$2";    shift 2 ;;
    --top-k)       TOP_K="$2";     shift 2 ;;
    --host)        HOST="$2";      shift 2 ;;
    --port)        PORT="$2";      shift 2 ;;
    --threads)     THREADS="$2";   shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; BOLD='\033[1m'; RESET='\033[0m'
echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║      HybridRecommender — PREDICT ($(printf '%-12s' ${MODE^^}))         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Prerequisite checks ────────────────────────────────────────────────────────
if ! command -v julia &> /dev/null; then
  echo -e "${BOLD}ERROR: julia not found in PATH.${RESET}" >&2; exit 1
fi

if [ ! -f "$SVD_MODEL" ]; then
  echo -e "${BOLD}ERROR: SVD model not found: $SVD_MODEL${RESET}" >&2
  echo "Run ./run_train.sh first." >&2; exit 1
fi

if [ ! -f "$FP_MODEL" ]; then
  echo -e "${BOLD}ERROR: FP model not found: $FP_MODEL${RESET}" >&2
  echo "Run ./run_train.sh first." >&2; exit 1
fi

export JULIA_NUM_THREADS="$THREADS"

# ── Build Julia args ───────────────────────────────────────────────────────────
JULIA_ARGS=(
  "--mode"      "$MODE"
  "--svd-model" "$SVD_MODEL"
  "--fp-model"  "$FP_MODEL"
  "--top-k"     "$TOP_K"
)

case "$MODE" in
  single)
    [ -z "$USER_ID" ] && { echo "ERROR: --user-id required for single mode" >&2; exit 1; }
    JULIA_ARGS+=("--user-id" "$USER_ID")
    [ -n "$CART"   ] && JULIA_ARGS+=("--cart"   "$CART")
    [ -n "$OUTPUT" ] && JULIA_ARGS+=("--output" "$OUTPUT")
    echo "  User ID  : $USER_ID"
    echo "  Cart     : ${CART:-<empty>}"
    ;;
  batch)
    [ -z "$INPUT" ] && { echo "ERROR: --input required for batch mode" >&2; exit 1; }
    JULIA_ARGS+=("--input" "$INPUT")
    [ -n "$OUTPUT" ] && JULIA_ARGS+=("--output" "$OUTPUT")
    echo "  Input    : $INPUT"
    echo "  Output   : ${OUTPUT:-stdout}"
    ;;
  server)
    JULIA_ARGS+=("--host" "$HOST" "--port" "$PORT")
    echo "  Host     : $HOST"
    echo "  Port     : $PORT"
    ;;
esac

echo "  SVD model: $SVD_MODEL"
echo "  FP model : $FP_MODEL"
echo "  Top-K    : $TOP_K"
echo "  Threads  : $THREADS"
echo ""

# ── Run ────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

if [ "$MODE" = "server" ]; then
  LOG_FILE="logs/server_${TIMESTAMP}.log"
  echo -e "${GREEN}[$(date +%T)] Starting server on ${HOST}:${PORT}… (log: $LOG_FILE)${RESET}"
  julia --threads "$THREADS" --project=. scripts/predict.jl "${JULIA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
else
  LOG_FILE="logs/predict_${MODE}_${TIMESTAMP}.log"
  echo -e "${GREEN}[$(date +%T)] Running prediction… (log: $LOG_FILE)${RESET}"
  julia --threads "$THREADS" --project=. scripts/predict.jl "${JULIA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

  EXIT_CODE=${PIPESTATUS[0]}
  if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${BOLD}${GREEN}✅ Prediction complete!${RESET}"
    [ -n "$OUTPUT" ] && echo "   Output: $OUTPUT"
  else
    echo -e "\n${BOLD}\033[0;31m❌ Prediction failed. Check $LOG_FILE${RESET}" >&2
    exit $EXIT_CODE
  fi
fi
