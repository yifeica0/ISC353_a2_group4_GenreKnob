#!/usr/bin/env bash
# Job 010: Experiment 1. Features in parallel (handcrafted on CPU, MERT on GPU3 in the
# MERT env, CLAP on GPU2 in the ACE env), then classifiers + tests + SHAP (MERT env).
source ~/gk-env.sh
cd ~/gk
mkdir -p results/exp1
echo "CLAP checkpoint: $(cat results/clap_choice.txt 2>/dev/null)"
$MERT_PY -m genreknob.embed hc > results/exp1/log_hc.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 $MERT_PY -m genreknob.embed mert > results/exp1/log_mert.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 $ACE_PY -m genreknob.embed clap > results/exp1/log_clap.txt 2>&1 &
wait
tail -n 3 results/exp1/log_hc.txt results/exp1/log_mert.txt results/exp1/log_clap.txt
$MERT_PY -m genreknob.exp1 2>&1 | tee results/exp1/log_exp1.txt | grep -v "Warning\|warn("
