#!/usr/bin/env bash
# Job 022f: style-reference audio (GTZAN training exemplar) and the planning LM.
source ~/gk-env.sh
cd ~/gk
grep -rn "reference_audio\|refer_audio" $GK_ACE_ROOT/acestep/core/generation/handler/generate_music_request.py | head -12
grep -rn "def initialize" -A 25 $GK_ACE_ROOT/acestep/llm_inference.py | head -40
CUDA_VISIBLE_DEVICES=1 $ACE_PY -m genreknob.diag --suite4 > results/diag/log5.txt 2>&1
grep "\[ace\]" results/diag/log5.txt
grep -v "INFO\|DEBUG\|WARNING" results/diag/log5.txt | grep -v "instrumental\|^$\|^#\|^- \|====\|Fill the\|text_prompt\|lyrics_text\|unknown\|endoftext\|flow-edit" | tail -n 60
