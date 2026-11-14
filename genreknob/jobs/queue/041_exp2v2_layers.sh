#!/usr/bin/env bash
# Job 041: v2 validation sweep of the steered layer (one layer at a time, alpha = 1).
source ~/gk-env.sh
cd ~/gk
set -x
$ACE_PY -m genreknob.exp2 plan layers
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $ACE_PY -m genreknob.exp2 gen layers --shard $i --nshards 4 > results/exp2/log_gen_layers_$i.txt 2>&1 &
done
wait
grep -h "FAILED\|s/clip" results/exp2/log_gen_layers_*.txt | tail -n 8
CUDA_VISIBLE_DEVICES=0 $ACE_PY -m genreknob.exp2 score layers
$ACE_PY -m genreknob.exp2 choose-layers
cat results/exp2/choices.json
