#!/usr/bin/env bash
# Job 022a: sanity checks of generator, judges and melody metric (validation clips only).
source ~/gk-env.sh
cd ~/gk
mkdir -p results/diag
CUDA_VISIBLE_DEVICES=0 $ACE_PY -m genreknob.diag > results/diag/log.txt 2>&1
grep -v "INFO\|DEBUG\|WARNING" results/diag/log.txt | tail -n 60
