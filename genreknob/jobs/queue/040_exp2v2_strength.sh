#!/usr/bin/env bash
# Job 040: Exp 2 run v2 (timestep shift 3, covers start from the noised source).
# Archive the v1 tables, recompute steering vectors with the new settings, then the
# validation sweep of the keep-original strength.
source ~/gk-env.sh
cd ~/gk
set -x
mkdir -p results/exp2_v1_shift1
mv results/exp2/scores_*.csv results/exp2/valid_*.csv results/exp2/choices.json results/exp2/fig_exp2_layers.png results/exp2/log_gen_* results/exp2_v1_shift1/ 2>/dev/null
for f in genre instrument mood; do cp -n $HOME/genreknob-data/cache/${f}_vectors.pt $HOME/genreknob-data/cache/${f}_vectors_v1_shift1.pt 2>/dev/null; done
CUDA_VISIBLE_DEVICES=0 $ACE_PY -m genreknob.steer genres > results/exp2/log_steer_genres.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $ACE_PY -m genreknob.steer concepts > results/exp2/log_steer_concepts.txt 2>&1 &
$ACE_PY -m genreknob.exp2 plan strength
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i $ACE_PY -m genreknob.exp2 gen strength --shard $i --nshards 4 > results/exp2/log_gen_strength_$i.txt 2>&1 &
done
wait
grep -h "FAILED\|s/clip" results/exp2/log_gen_strength_*.txt | tail -n 8
grep "\[steer\]" results/exp2/log_steer_genres.txt | tail -n 14
CUDA_VISIBLE_DEVICES=0 $ACE_PY -m genreknob.exp2 score strength
$ACE_PY -m genreknob.exp2 choose-strength
cat results/exp2/choices.json
