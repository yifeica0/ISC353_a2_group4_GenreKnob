#!/usr/bin/env bash
# Job 049: after adding the test-set showcase examples, start a second demo instance on port 7861 / GPU 1
# (leaves any demo on port 7860 running) and select every example through the API.
source ~/gk-env.sh
cd ~/gk
pkill -f "genreknob.app --host 127.0.0.1 --port 7861" 2>/dev/null; sleep 2
nohup env CUDA_VISIBLE_DEVICES=1 $ACE_PY -m genreknob.app --host 127.0.0.1 --port 7861 > ~/gk_demo_showcase.log 2>&1 < /dev/null &
for i in $(seq 1 100); do
  grep -q "Running on local URL" ~/gk_demo_showcase.log && break
  sleep 3
done
sleep 5
$ACE_PY tools/demo_examples_test.py http://127.0.0.1:7861 results/demo_selftest
pkill -f "genreknob.app --host 127.0.0.1 --port 7861"
grep -i "error\|traceback" ~/gk_demo_showcase.log | head -20
tail -n 5 ~/gk_demo_showcase.log
