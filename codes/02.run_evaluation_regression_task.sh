#!/bin/bash
# Experiment 2 (Table 3): personality regression with public / DP / non-DP embeddings.
# Extra arguments are passed through, e.g. ./02.run_evaluation_regression_task.sh --n_jobs 4
set -euo pipefail
cd "$(dirname "$0")"
# Add dependency onto projects
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"
python3 ./eval_personality_predictions.py "$@"
