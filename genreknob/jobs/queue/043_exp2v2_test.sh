#!/usr/bin/env bash
# Job 043: v2 test stage: 900 pairs x {T, C, CS} + knob curve; CLAP/radar scoring, MERT judge, statistics.
source ~/gk-env.sh
cd ~/gk
set -x
$ACE_PY -m genreknob.exp2 plan test
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $ACE_PY -m genreknob.exp2 gen test --shard $i --nshards 4 > results/exp2/log_gen_test_$i.txt 2>&1 &
done
wait
grep -h "FAILED\|s/clip" results/exp2/log_gen_test_*.txt | tail -n 12
CUDA_VISIBLE_DEVICES=0 $ACE_PY -m genreknob.exp2 score test
CUDA_VISIBLE_DEVICES=1 $MERT_PY -m genreknob.mert_judge score results/exp2/scores_test.csv out results/exp2/scores_test_mert.csv
$ACE_PY -m genreknob.exp2 stats
