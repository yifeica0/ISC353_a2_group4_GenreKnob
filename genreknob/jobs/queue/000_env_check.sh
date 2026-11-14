#!/usr/bin/env bash
# Job 000: report the server environment (no installs).
set -x
date; hostname; whoami; uname -a
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
df -h ~ /tmp 2>/dev/null
free -g; nproc
which python3 uv git rsync ffmpeg curl wget conda tmux 2>&1
python3 --version
cat /etc/os-release | head -3
ls ~/genreknob-data/genres_original 2>&1 | head
du -sh ~/genreknob-data/genres_original 2>&1
curl -sS -o /dev/null -w 'huggingface %{http_code}\n' -m 15 https://huggingface.co
curl -sS -o /dev/null -w 'pypi %{http_code}\n' -m 15 https://pypi.org/simple/
curl -sS -o /dev/null -w 'github %{http_code}\n' -m 15 https://github.com
curl -sS -o /dev/null -w 'pytorch %{http_code}\n' -m 15 https://download.pytorch.org/whl/cu128/
ls ~ | head -40
