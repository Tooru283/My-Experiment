#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HABITAT_LAB_DIR="${HABITAT_LAB_DIR:-${PROJECT_ROOT}/external/habitat-lab-v0.1.7}"

cd "$PROJECT_ROOT"

EPISODE_COUNT="${EPISODE_COUNT:-10}"
EXP_NAME="${EXP_NAME:-ep${EPISODE_COUNT}_series_m4$(date +%Y%m%d_%H%M%S)}"
# Cloud credentials are opt-in. The default runtime is local Qwen3.5-4B.
API_CONFIG="${OPENNAV_API_CONFIG:-}"
if [[ -n "${API_CONFIG}" ]]; then
  if [[ ! -f "${API_CONFIG}" ]]; then
    echo "ERROR: OPENNAV_API_CONFIG does not exist: ${API_CONFIG}" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${API_CONFIG}"
fi

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
# Local Qwen3.5-4B served by ./serve_qwen.sh (12GB-class GPU).
export OPENNAV_LLM_MODEL="${OPENNAV_LLM_MODEL:-/root/models/Qwen3.5-4B}"
# Client backend selector; the request payload still uses OPENNAV_LLM_MODEL above.
export OPENNAV_LLM_CLIENT_MODEL="${OPENNAV_LLM_CLIENT_MODEL:-Qwen/Qwen3.5-4B}"
export OPENNAV_LLM_API_KEY="${OPENNAV_LLM_API_KEY:-not-needed}"
# Override BASE_URL, MODEL, and CLIENT_MODEL together for another backend.
export OPENNAV_SIGLIP_PATH="${OPENNAV_SIGLIP_PATH:-/root/models/google-siglip-so400m-patch14-384}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ "${OPENNAV_LLM_BASE_URL}" == *dashscope* ]] && [[ -z "${OPENNAV_LLM_API_KEY}" ]]; then
  echo "ERROR: set DASHSCOPE_API_KEY or OPENNAV_LLM_API_KEY before using the Bailian endpoint." >&2
  exit 2
fi

if [[ "${OPENNAV_SKIP_LLM_PREFLIGHT:-0}" != "1" ]]; then
  python - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

base_url = os.environ["OPENNAV_LLM_BASE_URL"].rstrip("/")
model = os.environ["OPENNAV_LLM_MODEL"]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "ping"}],
    "temperature": 0,
    "max_tokens": 1,
}
request = urllib.request.Request(
    f"{base_url}/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": "Bearer " + (os.environ.get("OPENNAV_LLM_API_KEY") or "not-needed"),
        "Content-Type": "application/json",
    },
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()
except urllib.error.HTTPError as exc:
    detail = exc.read().decode("utf-8", errors="replace")
    print(
        f"ERROR: LLM preflight failed for model {model} at {base_url}: "
        f"HTTP {exc.code} {detail}",
        file=sys.stderr,
    )
    if "pinned to" in detail:
        print(
            "Restart the service on port 23333 with the same model as "
            "OPENNAV_LLM_MODEL.",
            file=sys.stderr,
        )
    sys.exit(1)
except Exception as exc:
    print(
        f"ERROR: LLM preflight failed for model {model} at {base_url}: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"LLM preflight OK: {model} at {base_url}")
PY
fi

# LLM identifier passed to run.py for logging/naming.
# Backend selector passed to llmClient; the served model ID is OPENNAV_LLM_MODEL.
LLM_NAME="${LLM_NAME:-${OPENNAV_LLM_CLIENT_MODEL}}"
# Override OPENNAV_LLM_CLIENT_MODEL and OPENNAV_LLM_MODEL together when switching families.

python run.py \
  --exp_name "$EXP_NAME" \
  --exp-config run_OpenNav.yaml \
  --llm "$LLM_NAME" \
  --api_key "${OPENNAV_LLM_API_KEY:-not-needed}" \
  SIMULATOR_GPU_IDS "[0]" \
  TORCH_GPU_ID 0 \
  TORCH_GPU_IDS "[0]" \
  EVAL.SPLIT val_unseen \
  EVAL.EPISODE_COUNT "$EPISODE_COUNT" \
  "$@"
