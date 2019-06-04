#!/bin/bash
# Add dependency onto projects
export PYTHONPATH="$PYTHONPATH:$PWD"
fileInput="../data/text8.zip"
trained_models="../data/text8_data/trained_models_dp_dpsgd"
python3 ./word2vec/tf_dp_word_embedding.py --train_path=$fileInput --trained_models=$trained_models --with_nce_loss=False --with_dp=True --clip_by_norm=True --RESTORE_LAST_CHECK_POINT=True
trained_models="../data/text8_data/trained_models_nodp_dpsgd"
python3 ./word2vec/tf_dp_word_embedding.py --train_path=$fileInput --trained_models=$trained_models --with_nce_loss=False --clip_by_norm=True --RESTORE_LAST_CHECK_POINT=True
