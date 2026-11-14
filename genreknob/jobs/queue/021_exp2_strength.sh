#!/usr/bin/env bash
# Job 021: Exp 2 validation stage 'strength' (cover strength) -> results/exp2/choices.json
source ~/gk-env.sh
cd ~/gk
set -x
$ACE_PY -m genreknob.exp2 plan strength
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $ACE_PY -m genreknob.exp2 gen strength --shard $i --nshards 4 > results/exp2/log_gen_strength_$i.txt 2>&1 &
done
wait
grep -h "FAILED\|s/clip" results/exp2/log_gen_strength_*.txt | tail -n 12
CUDA_VISIBLE_DEVICES=0 $ACE_PY -m genreknob.exp2 score strength
$ACE_PY -m genreknob.exp2 choose-strength
cat results/exp2/choices.json
