#!/usr/bin/env bash
# Job 046: re-run Exp 2 statistics only (chance level over the T/C/CS clips; knob reference on the same 90 pairs).
source ~/gk-env.sh
cd ~/gk
$ACE_PY -m genreknob.exp2 stats
cat results/exp2/test_knob_reference.csv
