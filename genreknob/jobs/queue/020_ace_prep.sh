#!/usr/bin/env bash
# Job 020: Exp 2 sources, genre radar, CAA steering vectors (ACE env, GPUs 0-2).
source ~/gk-env.sh
cd ~/gk
set -x
mkdir -p results/exp2
$ACE_PY -m genreknob.exp2 prepare
CUDA_VISIBLE_DEVICES=0 $ACE_PY -m genreknob.judge build-radar > results/exp2/log_radar.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $ACE_PY -m genreknob.steer genres > results/exp2/log_steer_genres.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 $ACE_PY -m genreknob.steer concepts > results/exp2/log_steer_concepts.txt 2>&1 &
wait
grep -v "DEBUG\|INFO\|Warning" results/exp2/log_radar.txt | tail -n 12
grep "\[steer\]" results/exp2/log_steer_genres.txt results/exp2/log_steer_concepts.txt | tail -n 30
grep -A 12 "cosine similarity" results/exp2/log_steer_genres.txt
ls -la ~/genreknob-data/cache
