#!/usr/bin/env bash
# Job 044: offline demo gallery with the v2 settings (results/demo_cache/index.html + mp3).
source ~/gk-env.sh
cd ~/gk
CUDA_VISIBLE_DEVICES=0 $ACE_PY -m genreknob.app --cache
ls -la results/demo_cache | head -60
