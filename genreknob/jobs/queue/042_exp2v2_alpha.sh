#!/usr/bin/env bash
# Job 042: v2 validation sweep of the steering strength alpha.
source ~/gk-env.sh
cd ~/gk
set -x
$ACE_PY -m genreknob.exp2 plan alpha
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $ACE_PY -m genreknob.exp2 gen alpha --shard $i --nshards 4 > results/exp2/log_gen_alpha_$i.txt 2>&1 &
done
wait
grep -h "FAILED\|s/clip" results/exp2/log_gen_alpha_*.txt | tail -n 8
CUDA_VISIBLE_DEVICES=0 $ACE_PY -m genreknob.exp2 score alpha
$ACE_PY -m genreknob.exp2 choose-alpha
cat results/exp2/choices.json
