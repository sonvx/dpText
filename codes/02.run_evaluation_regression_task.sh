#!/bin/bash
# Add dependency onto projects
export PYTHONPATH="$PYTHONPATH:$PWD"
python3 ./eval_personality_predictions.py
