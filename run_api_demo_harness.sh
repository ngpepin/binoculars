#!/usr/bin/env bash
set -euo pipefail

# Harness/API constants.
API_PORT=8765
API_PROFILE="fast"
API_STARTUP_TIMEOUT_S=90
API_POLL_INTERVAL_S=0.5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINOCULARS_SH="${SCRIPT_DIR}/binoculars.sh"
HARNESS_PY="${SCRIPT_DIR}/api_demo_harness.py"
VENV_PY="${SCRIPT_DIR}/venv/bin/python"
API_LOG_FILE="${TMPDIR:-/tmp}/binoculars-api-demo.log"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<EOF
Usage: ./run_api_demo_harness.sh

Starts Binoculars API (if not already running on API_PORT), then opens the GUI demo harness.

Constants:
  API_PORT=${API_PORT}
  API_PROFILE=${API_PROFILE}
EOF
  exit 0
fi

if [[ ! -x "${BINOCULARS_SH}" ]]; then
  echo "ERROR: missing executable ${BINOCULARS_SH}" >&2
  exit 2
fi
if [[ ! -f "${HARNESS_PY}" ]]; then
  echo "ERROR: missing harness script ${HARNESS_PY}" >&2
  exit 2
fi

if [[ -x "${VENV_PY}" ]]; then
  RUN_PY="${VENV_PY}"
else
  RUN_PY="$(command -v python3 || true)"
fi
if [[ -z "${RUN_PY}" ]]; then
  echo "ERROR: could not find python3 or ${VENV_PY}" >&2
  exit 2
fi

health_ok() {
  "${RUN_PY}" - "${API_PORT}" <<'PY'
import json
import sys
import urllib.error
import urllib.request

port = int(sys.argv[1])
url = f"http://127.0.0.1:{port}/health"
try:
    with urllib.request.urlopen(url, timeout=1.5) as resp:
        body = resp.read().decode("utf-8", errors="replace")
except Exception:
    sys.exit(1)
try:
    parsed = json.loads(body)
except Exception:
    sys.exit(1)
if isinstance(parsed, dict) and parsed.get("ok") is True:
    sys.exit(0)
sys.exit(1)
PY
}

wait_for_health() {
  local timeout_s="${1}"
  local interval_s="${2}"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if health_ok; then
      return 0
    fi
    local now_ts elapsed
    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    if (( elapsed >= timeout_s )); then
      return 1
    fi
    sleep "${interval_s}"
  done
}

API_PID=""
cleanup() {
  if [[ -n "${API_PID}" ]]; then
    kill "${API_PID}" >/dev/null 2>&1 || true
    wait "${API_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if health_ok; then
  echo "[harness] using existing API on port ${API_PORT}" >&2
else
  echo "[harness] starting API on port ${API_PORT} (profile=${API_PROFILE})" >&2
  "${BINOCULARS_SH}" --config "${API_PROFILE}" --api "${API_PORT}" >>"${API_LOG_FILE}" 2>&1 &
  API_PID="$!"

  if ! wait_for_health "${API_STARTUP_TIMEOUT_S}" "${API_POLL_INTERVAL_S}"; then
    echo "ERROR: API did not become healthy within ${API_STARTUP_TIMEOUT_S}s." >&2
    echo "Tail of ${API_LOG_FILE}:" >&2
    tail -n 80 "${API_LOG_FILE}" >&2 || true
    exit 2
  fi
fi

export BINOCULARS_API_BASE_URL="http://127.0.0.1:${API_PORT}"
"${RUN_PY}" "${HARNESS_PY}"
