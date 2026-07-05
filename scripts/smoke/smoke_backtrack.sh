#!/usr/bin/env bash
# A/B smoke: BACKTRACK alone (DEPTH_STOP_VETO forced OFF). Verifies caveat (1):
# a backtrack_offer(offered=True) is followed by a backtrack_apply(applied=True) ==
# MOVE_BACK survives test_decisions and reaches the single action-4 emitter.
#
# Backtrack fires on ego-stall or dead_end, so this targets wandering / step-limit
# episodes (Group-B random-walk-far cases 1084, 1106) where a stall is likely. If the
# verifier reports INCONCLUSIVE (never offered), widen SMOKE_EPISODE_IDS.
#
# PREREQS (single GPU, serial): the full baseline must have exited, and the qwen
# transformers-serve backend must still be up. This script refuses to run otherwise.
# Never enable both switches in one run -- attribution is the whole point.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

BASELINE_PID="${BASELINE_PID:-6378}"
if ps -p "$BASELINE_PID" >/dev/null 2>&1; then
  echo "REFUSING: baseline PID $BASELINE_PID still running (single GPU is busy). Wait for it to finish."
  exit 1
fi

SMOKE_EPISODE_IDS="${SMOKE_EPISODE_IDS:-1084 1106}"
EPISODE_COUNT="${EPISODE_COUNT:-5}"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="smoke_backtrack_${STAMP}"
SINCE="$(date +%s)"

echo "=== BACKTRACK smoke: exp=$EXP_NAME episodes=[$SMOKE_EPISODE_IDS] ==="
OPENNAV_EPISODE_IDS="$SMOKE_EPISODE_IDS" \
python run.py \
  --exp_name "$EXP_NAME" \
  --exp-config run_OpenNav.yaml \
  --llm Qwen/Qwen3.5-9B --api_key not-needed \
  SIMULATOR_GPU_IDS [0] TORCH_GPU_ID 0 TORCH_GPU_IDS [0] \
  EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT "$EPISODE_COUNT" \
  OPENNAV_HARNESS.BACKTRACK.ENABLED True \
  OPENNAV_HARNESS.VISUAL_TARGET_VERIFIER.DEPTH_STOP_VETO.ENABLED False

echo "=== verifying ==="
python scripts/smoke/verify_smoke.py --mode backtrack --since "$SINCE"
