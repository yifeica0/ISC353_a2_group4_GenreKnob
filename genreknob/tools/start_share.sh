#!/usr/bin/env bash
# Start GenreKnob with a temporary PUBLIC link (https://xxxx.gradio.live, valid 72 h), behind a password.
# Anyone with the link AND the password can use the demo (and this server's GPU) until you stop it.
#
# From the Mac:   ssh USER@GPU_SERVER 'bash ~/gk/tools/start_share.sh'            (random password)
#            or   ssh USER@GPU_SERVER 'bash ~/gk/tools/start_share.sh MyPassword'
# Stop it:        ssh USER@GPU_SERVER 'pkill -f genreknob.app'
source ~/gk-env.sh
cd ~/gk
PASS=${1:-$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')}
pkill -f "genreknob.app" 2>/dev/null
sleep 2
nohup env CUDA_VISIBLE_DEVICES=${GK_DEMO_GPU:-0} $ACE_PY -m genreknob.app --host 127.0.0.1 --port 7860 \
  --share --auth "group4:$PASS" > ~/gk_demo.log 2>&1 < /dev/null &
echo "Loading models and opening the public link (1-2 minutes)..."
URL=""
for i in $(seq 1 60); do
  URL=$(grep -o 'https://[a-z0-9.-]*\.gradio\.live' ~/gk_demo.log | head -1)
  [ -n "$URL" ] && break
  sleep 3
done
if [ -n "$URL" ]; then
  echo "Public link: $URL   (valid for 72 hours)"
  echo "Login: user group4, password $PASS"
  echo "Stop: pkill -f genreknob.app"
else
  echo "No public link yet. Last lines of ~/gk_demo.log:"
  tail -n 8 ~/gk_demo.log
fi
