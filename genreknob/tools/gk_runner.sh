#!/usr/bin/env bash
# GenreKnob job runner (runs on the GPU server).
# Executes each new ~/gk/jobs/queue/*.sh exactly once, in name order.
# Log -> ~/gk/results/jobs/<name>.log, exit code -> ~/gk/results/jobs/<name>.status
# Stop: kill $(cat ~/gk_runner.pid)
export PATH="$HOME/.local/bin:$PATH"
cd ~/gk || exit 1
mkdir -p jobs/queue results/jobs
while true; do
  date '+%F %T' > results/jobs/_heartbeat
  for job in $(ls jobs/queue/*.sh 2>/dev/null | sort); do
    name=$(basename "$job" .sh)
    [ -f "results/jobs/$name.status" ] && continue
    echo running > "results/jobs/$name.status"
    bash "$job" > "results/jobs/$name.log" 2>&1 < /dev/null
    echo $? > "results/jobs/$name.status"
  done
  sleep 10
done
