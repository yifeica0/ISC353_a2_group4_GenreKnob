#!/usr/bin/env bash
# Job 022e: unquantised cover (cover-nofsq) and FlowEdit variants.
source ~/gk-env.sh
cd ~/gk
grep -n "cover-nofsq\|nofsq" $GK_ACE_ROOT/acestep/constants.py | head -12
CUDA_VISIBLE_DEVICES=0 $ACE_PY -m genreknob.diag --suite3 > results/diag/log4.txt 2>&1
grep -v "INFO\|DEBUG\|WARNING" results/diag/log4.txt | grep -v "instrumental\|^$\|^#\|^- \|====\|Fill the\|text_prompt\|lyrics_text\|unknown\|endoftext\|flow-edit" | tail -n 60
