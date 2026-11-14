#!/usr/bin/env bash
# Job 045: start the demo bound to this machine only (127.0.0.1, no public link) and drive it through its API.
source ~/gk-env.sh
cd ~/gk
pkill -f "genreknob.app" 2>/dev/null; sleep 2
nohup env CUDA_VISIBLE_DEVICES=0 $ACE_PY -m genreknob.app --host 127.0.0.1 --port 7861 > ~/gk_demo_selftest.log 2>&1 < /dev/null &
for i in $(seq 1 100); do
  grep -q "Running on local URL" ~/gk_demo_selftest.log && break
  sleep 3
done
sleep 5
EX=$(ls ~/gk-work/exp2/sources/classical.*.wav | head -1)
echo "example: $EX"
$ACE_PY tools/demo_selftest.py http://127.0.0.1:7861 "$EX" results/demo_selftest
pkill -f "genreknob.app"
grep -i "error\|traceback" ~/gk_demo_selftest.log | head -20
tail -n 5 ~/gk_demo_selftest.log
