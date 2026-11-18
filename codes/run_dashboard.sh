#!/bin/bash
# Start the embedding dashboard at http://localhost:8501 (only reachable from this machine).
# Uses ../.venv if it exists and no virtual environment is active. Extra arguments go to streamlit,
# e.g. ./run_dashboard.sh --server.port 8600
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"
PYTHON="${PYTHON:-python3}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -x ../.venv/bin/python ]; then
    PYTHON=../.venv/bin/python
fi
exec "$PYTHON" -m streamlit run dashboard/app.py "$@"
