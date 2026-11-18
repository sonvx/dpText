#!/bin/bash
# Experiment 1 (Figure 2): semantic-space changes with/without differential privacy.
set -euo pipefail
cd "$(dirname "$0")"
# Add dependency onto projects
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"
python3 ./eval_embedding_space.py
