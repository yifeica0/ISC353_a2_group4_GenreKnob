#!/usr/bin/env bash
# Job 022d: how do cover noise, FlowEdit and full repaint trade genre for melody? (+ the code that implements them)
source ~/gk-env.sh
cd ~/gk
A=$GK_ACE_ROOT
grep -rn "cover_noise_strength" $A/acestep --include=*.py | grep -v "^.*#" | head -20
grep -rn "def .*flow_edit\|flow_edit_morph" $A/acestep --include=*.py | head -12
grep -rn "repaint_strength" $A/acestep --include=*.py | head -8
CUDA_VISIBLE_DEVICES=0 $ACE_PY -m genreknob.diag --suite2 > results/diag/log3.txt 2>&1
grep -v "INFO\|DEBUG\|WARNING" results/diag/log3.txt | grep -v "instrumental\|^$\|^#\|^- \|====\|Fill the\|text_prompt\|lyrics_text\|unknown\|endoftext" | tail -n 60
