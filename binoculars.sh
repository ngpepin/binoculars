#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$SCRIPT_DIR/venv/bin/activate"
PYTHON_SCRIPT="$SCRIPT_DIR/binoculars.py"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "ERROR: virtual environment activation script not found: $VENV_ACTIVATE" >&2
  exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "ERROR: script not found: $PYTHON_SCRIPT" >&2
  exit 1
fi

source "$VENV_ACTIVATE"

cleanup() {
  deactivate >/dev/null 2>&1 || true
}

kill_prior_instances() {
  if [[ "${BINOCULARS_DISABLE_AUTO_KILL:-0}" == "1" ]]; then
    return
  fi

  if ! command -v pgrep >/dev/null 2>&1; then
    return
  fi

  local uid
  uid="$(id -u)"
  local patterns=(
    "$SCRIPT_DIR/binoculars.py"
    "$SCRIPT_DIR/binoculars_llamacpp.py"
  )

  local pid
  local -a pids=()
  local -A seen=()
  local pat
  for pat in "${patterns[@]}"; do
    while IFS= read -r pid; do
      [[ -z "$pid" ]] && continue
      [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
      if [[ -z "${seen[$pid]+x}" ]]; then
        seen["$pid"]=1
        pids+=("$pid")
      fi
    done < <(pgrep -u "$uid" -f "$pat" || true)
  done

  if [[ "${#pids[@]}" -eq 0 ]]; then
    return
  fi

  echo "INFO: terminating prior binoculars instance(s): ${pids[*]}" >&2

  local -a alive=()
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
      alive+=("$pid")
    fi
  done

  local deadline=$((SECONDS + 8))
  while [[ "${#alive[@]}" -gt 0 && "$SECONDS" -lt "$deadline" ]]; do
    local -a next_alive=()
    for pid in "${alive[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        next_alive+=("$pid")
      fi
    done
    alive=("${next_alive[@]}")
    [[ "${#alive[@]}" -gt 0 ]] && sleep 0.2
  done

  if [[ "${#alive[@]}" -gt 0 ]]; then
    echo "INFO: force killing stuck instance(s): ${alive[*]}" >&2
    for pid in "${alive[@]}"; do
      kill -KILL "$pid" 2>/dev/null || true
    done
  fi
}

forward_int() {
  if [[ -n "${child_pid:-}" ]]; then
    kill -INT "$child_pid" 2>/dev/null || true
  fi
}

trap cleanup EXIT
trap forward_int INT

kill_prior_instances

python "$PYTHON_SCRIPT" "$@" &
child_pid=$!
wait "$child_pid"
exit_code=$?

exit "$exit_code"
