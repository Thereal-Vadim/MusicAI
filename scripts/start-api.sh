#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV="${ROOT}/.venv311"
if [[ ! -x "${VENV}/bin/uvicorn" ]]; then
  echo "Missing ${VENV}/bin/uvicorn — create .venv311 first." >&2
  exit 1
fi

export PYTHONPATH="${ROOT}/packages:${ROOT}/apps/api:${ROOT}/apps/worker${PYTHONPATH:+:${PYTHONPATH}}"

exec "${VENV}/bin/uvicorn" musicai_api.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --app-dir apps/api \
  "$@"
