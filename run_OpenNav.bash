#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HABITAT_LAB_DIR="${HABITAT_LAB_DIR:-${PROJECT_ROOT}/external/habitat-lab-v0.1.7}"

cd "$PROJECT_ROOT"

EPISODE_COUNT="${EPISODE_COUNT:-100}"
EXP_NAME="${EXP_NAME:-ep${EPISODE_COUNT}_series_qwen_siglip_local_$(date +%Y%m%d_%H%M%S)}"

if [[ -d "${HABITAT_LAB_DIR}/habitat" ]]; then
  export PYTHONPATH="${HABITAT_LAB_DIR}:${PROJECT_ROOT}:${PYTHONPATH:-}"
else
  echo "Warning: habitat-lab not found at ${HABITAT_LAB_DIR}; Python may fail to import habitat." >&2
  export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
fi

export MAGNUM_LOG="${MAGNUM_LOG:-verbose}"
export HABITAT_SIM_LOG="${HABITAT_SIM_LOG:-verbose}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENNAV_LLM_BASE_URL="${OPENNAV_LLM_BASE_URL:-http://127.0.0.1:23333/v1}"
export OPENNAV_LLM_MODEL="${OPENNAV_LLM_MODEL:-/root/models/Qwen3.5-4B}"
export OPENNAV_SIGLIP_PATH="${OPENNAV_SIGLIP_PATH:-/root/models/google-siglip-so400m-patch14-384}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

python run.py \
  --exp_name "$EXP_NAME" \
  --exp-config run_OpenNav.yaml \
  --llm Qwen/Qwen3.5-4B \
  --api_key not-needed \
  SIMULATOR_GPU_IDS "[0]" \
  TORCH_GPU_ID 0 \
  TORCH_GPU_IDS "[0]" \
  EVAL.SPLIT val_unseen \
  EVAL.EPISODE_COUNT "$EPISODE_COUNT" \
  "$@"
