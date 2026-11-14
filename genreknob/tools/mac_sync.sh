#!/usr/bin/env bash
# Run on the Mac:  bash ~/ISC353_a2_group4/genreknob/tools/mac_sync.sh user@host
# 1) copies GTZAN audio to the server once, 2) (re)starts the job runner there,
# 3) every 15 s pushes code/jobs to ~/gk and pulls ~/gk/results back.
set -u
SERVER="${1:?usage: bash mac_sync.sh user@host}"
LOCAL="$HOME/ISC353_a2_group4/genreknob"
GTZAN="$HOME/ISC353_a2_group4/gtzan-dataset-music-genre-classification/Data/genres_original"
mkdir -p "$HOME/.ssh" "$LOCAL/results"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%C -o ControlPersist=12h -o ServerAliveInterval=30"
RSH="ssh $SSH_OPTS"

echo "[1/4] connecting to $SERVER (type your password once if asked)"
$RSH "$SERVER" 'mkdir -p ~/gk/jobs/queue ~/gk/results/jobs ~/genreknob-data && echo connected' || exit 1

echo "[2/4] copying GTZAN audio to the server (first time ~1.2 GB)"
rsync -az -e "$RSH" "$GTZAN" "$SERVER:genreknob-data/" || echo "WARNING: GTZAN copy failed"

echo "[3/4] (re)starting the job runner on the server"
rsync -az -e "$RSH" "$LOCAL/tools/gk_runner.sh" "$SERVER:gk_runner.sh"
$RSH "$SERVER" 'for f in ~/gk_runner.pid ~/genreknob_runner.pid; do [ -f $f ] && kill $(cat $f) 2>/dev/null; done; nohup bash ~/gk_runner.sh > ~/gk_runner.out 2>&1 < /dev/null & echo $! > ~/gk_runner.pid; echo "runner pid $(cat ~/gk_runner.pid)"'

echo "[4/4] syncing every 15 s -- keep this window open (Ctrl-C to stop)"
caffeinate -dims -w $$ &
while true; do
  rsync -az -e "$RSH" --exclude results/ "$LOCAL/" "$SERVER:gk/"
  rsync -az -e "$RSH" "$SERVER:gk/results/" "$LOCAL/results/"
  sleep 15
done
