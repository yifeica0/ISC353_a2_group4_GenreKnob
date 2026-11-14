#!/usr/bin/env bash
# Job 004: choose the CLAP checkpoint for zero-shot judging, using validation tracks only.
source ~/gk-env.sh
cd ~/gk
CUDA_VISIBLE_DEVICES=1 $ACE_PY -m genreknob.judge select-clap 2>&1 | grep -v "Warning\|warn("
cat results/clap_selection.csv; echo; cat results/clap_choice.txt
