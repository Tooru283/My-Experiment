#!/usr/bin/env bash
# =============================================================================
# serve_qwen.sh — start/stop/status the local Qwen3.5 OpenAI-compatible server
# that Open-Nav's navigator/VLM client talks to (OPENNAV_LLM_BASE_URL, port 23333).
#
# Backend: HuggingFace `transformers serve` (greedy decoding only — this is
# intentional and correct for reproducible, comparable eval baselines; do NOT
# swap in a sampling backend for baseline runs). Runs in the `qwen35-serve`
# conda env. The eval itself (run_OpenNav.bash) runs in the `opennav` env.
#
# Usage:
#   ./serve_qwen.sh start      # launch in background, wait until ready
#   ./serve_qwen.sh stop       # kill the running server
#   ./serve_qwen.sh status     # is it up? which model?
#   ./serve_qwen.sh restart
#
# Override via env vars, e.g.:
#   MODEL=/root/models/Qwen3.5-4B ./serve_qwen.sh start
#   PORT=23333 DEVICE=cuda:0 DTYPE=bfloat16 ./serve_qwen.sh start
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- configuration (all overridable) ----------------------------------------
MODEL="${MODEL:-/root/models/Qwen3.5-9B}"     # 9B default; set to Qwen3.5-4B for the 4B
PORT="${PORT:-23333}"                          # must match OPENNAV_LLM_BASE_URL
HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"                      # bf16 = full-fidelity baseline
REASONING="${REASONING:-off}"
SERVE_ENV="${SERVE_ENV:-qwen35-serve}"          # conda env holding `transformers serve`
READY_TIMEOUT="${READY_TIMEOUT:-600}"           # seconds to wait for model load

LOG_DIR="${PROJECT_ROOT}/logs/qwen_server"
PID_FILE="${LOG_DIR}/serve_${PORT}.pid"
BASE_URL="http://127.0.0.1:${PORT}/v1"

mkdir -p "$LOG_DIR"

# ---- helpers ----------------------------------------------------------------
_server_pid() {
  # A live `transformers serve` bound to $PORT, if any.
  pgrep -f "transformers serve .*--port ${PORT}" 2>/dev/null | head -1
}

_ready() {
  # True when the server answers a chat/completions request for $MODEL.
  MODEL="$MODEL" BASE_URL="$BASE_URL" python - <<'PY' 2>/dev/null
import json, os, sys, urllib.request, urllib.error
req = urllib.request.Request(
    f"{os.environ['BASE_URL']}/chat/completions",
    data=json.dumps({
        "model": os.environ["MODEL"],
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0, "max_tokens": 1,
    }).encode(),
    headers={"Authorization": "Bearer not-needed", "Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

cmd_status() {
  local pid; pid="$(_server_pid || true)"
  if [[ -n "$pid" ]]; then
    echo "server process: PID $pid (port $PORT)"
  else
    echo "server process: not running"
  fi
  if _ready; then
    echo "readiness ping : OK ($MODEL at $BASE_URL)"
  else
    echo "readiness ping : NOT READY"
  fi
}

cmd_stop() {
  # Match every process bound to this port — the `conda run` wrapper AND the
  # actual `transformers serve` child — so we don't orphan the real server.
  local pids; pids="$(pgrep -f "transformers serve .*--port ${PORT}" 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    echo "nothing to stop on port $PORT"
    rm -f "$PID_FILE"
    return 0
  fi
  echo "stopping server PID(s): $(echo "$pids" | tr '\n' ' ')..."
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 20); do
    _server_pid >/dev/null || { echo "stopped."; rm -f "$PID_FILE"; return 0; }
    sleep 0.5
  done
  echo "did not exit gracefully; sending SIGKILL"
  # shellcheck disable=SC2086
  kill -9 $(pgrep -f "transformers serve .*--port ${PORT}" 2>/dev/null || true) 2>/dev/null || true
  rm -f "$PID_FILE"
}

cmd_start() {
  if [[ -n "$(_server_pid || true)" ]]; then
    echo "server already running on port $PORT (PID $(_server_pid)); use 'restart' to reload."
    cmd_status
    return 0
  fi

  local ts log
  ts="$(date +%Y%m%d_%H%M%S)"
  log="${LOG_DIR}/qwen_serve_${PORT}_${ts}.log"

  echo "starting: transformers serve $MODEL"
  echo "  host=$HOST port=$PORT device=$DEVICE dtype=$DTYPE reasoning=$REASONING env=$SERVE_ENV"
  echo "  log -> $log"

  nohup conda run -n "$SERVE_ENV" --no-capture-output \
    transformers serve "$MODEL" \
      --host "$HOST" \
      --port "$PORT" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --reasoning "$REASONING" \
      --log-level info \
    > "$log" 2>&1 &

  echo "$!" > "$PID_FILE"
  echo -n "waiting for model to load (up to ${READY_TIMEOUT}s) "
  local waited=0
  while (( waited < READY_TIMEOUT )); do
    if _ready; then
      echo ""
      echo "READY: $MODEL at $BASE_URL"
      return 0
    fi
    # Fail fast if the process died during load.
    if [[ -z "$(_server_pid || true)" ]] && ! _ready; then
      echo ""
      echo "ERROR: server process exited before becoming ready. Last log lines:" >&2
      tail -30 "$log" >&2
      return 1
    fi
    echo -n "."
    sleep 5
    waited=$((waited + 5))
  done
  echo ""
  echo "ERROR: not ready after ${READY_TIMEOUT}s. Last log lines:" >&2
  tail -30 "$log" >&2
  return 1
}

# ---- dispatch ---------------------------------------------------------------
case "${1:-start}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  status)  cmd_status ;;
  restart) cmd_stop; cmd_start ;;
  *) echo "usage: $0 {start|stop|status|restart}" >&2; exit 2 ;;
esac
