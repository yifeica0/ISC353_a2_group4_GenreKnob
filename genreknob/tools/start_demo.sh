#!/usr/bin/env bash
# Start the GenreKnob demo on the GPU server, listening on the server itself only.
#
# From the Mac, one line starts it AND opens a private SSH tunnel (keep the window open):
#   ssh -L 7860:127.0.0.1:7860 USER@GPU_SERVER 'bash ~/gk/tools/start_demo.sh; sleep 36000'
# then open  http://localhost:7860  in the Mac's browser. Because the page is served
# from "localhost", the browser also allows the microphone (needed for humming).
# Stop it:   ssh USER@GPU_SERVER 'pkill -f genreknob.app'
# Nothing is published on the internet.
source ~/gk-env.sh
cd ~/gk
pkill -f "genreknob.app" 2>/dev/null
sleep 2
nohup env CUDA_VISIBLE_DEVICES=${GK_DEMO_GPU:-0} $ACE_PY -m genreknob.app --host 127.0.0.1 --port 7860 \
  > ~/gk_demo.log 2>&1 < /dev/null &
echo "Loading models (about 1 minute)..."
for i in $(seq 1 100); do
  if grep -q "Running on local URL" ~/gk_demo.log; then break; fi
  sleep 3
done
if grep -q "Running on local URL" ~/gk_demo.log; then
  echo "Ready: open http://localhost:7860 on the Mac (keep this SSH window open)"
else
  echo "Not ready yet - check ~/gk_demo.log on the server"
fi
