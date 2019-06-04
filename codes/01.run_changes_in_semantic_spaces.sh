#!/bin/bash
# Add dependency onto projects
export PYTHONPATH="$PYTHONPATH:$PWD"
python3 ./eval_embedding_space.py