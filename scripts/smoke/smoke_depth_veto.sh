#!/usr/bin/env bash
# A/B smoke: DEPTH_STOP_VETO alone (BACKTRACK forced OFF). Verifies caveat (2):
# depth_stop_veto events carry a non-null depth_reading_m == the depth sensor is read.
#
# Runs the canonical false-stop ep377 (conf-1.00 @ 5.42m) so the veto is GUARANTEED to be
# evaluated on a stop that SHOULD be vetoed (5.4m > 3.0m). Add more ids via SMOKE_EPISODE_IDS.
#
# PREREQS (single GPU, serial): the full baseline must have exited, and the qwen
# transformers-serve backend must still be up. This script refuses to run otherwise.
# Never enable both switches in one run -- attribution is the whole point.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

# Mirror the runtime env that run_OpenNav.bash (the baseline launcher) exports.
# Critically OPENNAV_LLM_MODEL: `transformers serve` is model-pinned and 400s any
# request whose model != the loaded weights, so the client label MUST be 9B to
# match the served 9B backend (parity with clean_baseline_v1). Missing this is
# why the 07-05/07-18 smokes defaulted to 4B and were rejected.
export OPENNAV_LLM_BASE_URL="${OPENNAV_LLM_BASE_URL:-http://127.0.0.1:23333/v1}"
export OPENNAV_LLM_MODEL="${OPENNAV_LLM_MODEL:-/root/models/Qwen3.5-9B}"
export OPENNAV_SIGLIP_PATH="${OPENNAV_SIGLIP_PATH:-/root/models/google-siglip-so400m-patch14-384}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASELINE_PID="${BASELINE_PID:-6378}"
if ps -p "$BASELINE_PID" >/dev/null 2>&1; then
  echo "REFUSING: baseline PID $BASELINE_PID still running (single GPU is busy). Wait for it to finish."
  exit 1
fi

SMOKE_EPISODE_IDS="${SMOKE_EPISODE_IDS:-377}"
EPISODE_COUNT="${EPISODE_COUNT:-3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="smoke_depthveto_${STAMP}"
SINCE="$(date +%s)"

echo "=== DEPTH_STOP_VETO smoke: exp=$EXP_NAME episodes=[$SMOKE_EPISODE_IDS] ==="
OPENNAV_EPISODE_IDS="$SMOKE_EPISODE_IDS" \
python run.py \
  --exp_name "$EXP_NAME" \
  --exp-config run_OpenNav.yaml \
  --llm Qwen/Qwen3.5-9B --api_key not-needed \
  SIMULATOR_GPU_IDS [0] TORCH_GPU_ID 0 TORCH_GPU_IDS [0] \
  EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT "$EPISODE_COUNT" \
  OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.DEPTH_STOP_VETO.ENABLED True \
  OPENNAV_HARNESS.BACKTRACK.ENABLED False

echo "=== verifying ==="
python scripts/smoke/verify_smoke.py --mode depth --since "$SINCE"
