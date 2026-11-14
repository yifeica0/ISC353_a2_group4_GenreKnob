#!/usr/bin/env bash
# Job 048: check the new --auth option on a private instance (127.0.0.1:7861, GPU 1, no public link).
source ~/gk-env.sh
cd ~/gk
pkill -f "genreknob.app --host 127.0.0.1 --port 7861" 2>/dev/null; sleep 2
nohup env CUDA_VISIBLE_DEVICES=1 $ACE_PY -m genreknob.app --host 127.0.0.1 --port 7861 --auth "test:pw123" \
  > ~/gk_demo_auth.log 2>&1 < /dev/null &
for i in $(seq 1 100); do grep -q "Running on local URL" ~/gk_demo_auth.log && break; sleep 3; done
sleep 5
$ACE_PY - <<'PY'
from gradio_client import Client
try:
    Client("http://127.0.0.1:7861", verbose=False)
    print("[auth] WITHOUT password: page opened (auth NOT working)")
except Exception as e:
    print("[auth] without password: refused, as expected:", repr(e)[:120])
c = Client("http://127.0.0.1:7861", auth=("test", "pw123"), verbose=False)
f = c.predict("jazz (jazz.00002)", api_name="/load_example")
print("[auth] with password: OK, example loaded:", f)
PY
pkill -f "genreknob.app --host 127.0.0.1 --port 7861"
grep -i "traceback\|error" ~/gk_demo_auth.log | head -10
