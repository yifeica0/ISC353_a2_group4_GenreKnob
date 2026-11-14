#!/usr/bin/env bash
# Job 022c: diagnostics again (fixed source scoring; adds timestep-shift and caption variants).
source ~/gk-env.sh
cd ~/gk
mkdir -p results/diag
CUDA_VISIBLE_DEVICES=0 $ACE_PY -m genreknob.diag > results/diag/log2.txt 2>&1
grep -v "INFO\|DEBUG\|WARNING" results/diag/log2.txt | tail -n 80
