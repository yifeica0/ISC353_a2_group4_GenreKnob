#!/usr/bin/env bash
# Job 001: install everything (no sudo). ~15-25 min, mostly downloads.
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"

# 1) uv (Python package/env manager)
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

# 2) FFmpeg shared libraries via micromamba (safety net for torchaudio/torchcodec)
mkdir -p ~/gk-tools && cd ~/gk-tools
if [ ! -x bin/micromamba ]; then curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba; fi
[ -d ~/gk-tools/ffmpeg ] || ./bin/micromamba create -y -r ~/gk-tools/mamba -p ~/gk-tools/ffmpeg -c conda-forge "ffmpeg>=6,<8" || echo "ffmpeg install failed (non-fatal)"
ls ~/gk-tools/ffmpeg/bin/ffmpeg || true

# 3) ACE-Step 1.5, pinned to the commit the code was written against
mkdir -p ~/gk-ace && cd ~/gk-ace
[ -d ACE-Step-1.5 ] || git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5
git fetch -q origin || true
git checkout -q ca1e85fe9430179831e6bc6be790c332190a3866
uv sync
uv pip install --python .venv/bin/python librosa scikit-learn joblib pandas
.venv/bin/python -c "import torch, torchaudio, transformers, gradio; print('ACE env', torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), transformers.__version__, gradio.__version__)"

# 4) Separate env for MERT (its remote code needs an older transformers)
uv venv -p 3.11 ~/gk-envs/mert
P=~/gk-envs/mert/bin/python
uv pip install --python $P torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python $P "transformers==4.38.2" "numpy<2" "scipy<1.14" nnAudio "librosa==0.10.2.post1" soundfile "scikit-learn==1.5.2" "xgboost==2.1.4" "pandas<2.3" "matplotlib<3.10" joblib accelerate
$P -c "import torch, transformers, xgboost, librosa, sklearn; print('MERT env', torch.__version__, torch.cuda.is_available(), transformers.__version__, xgboost.__version__, librosa.__version__, sklearn.__version__)"

# 5) one env file sourced by every later job
cat > ~/gk-env.sh << 'EOS'
export PATH="$HOME/.local/bin:$PATH"
export ACE_PY="$HOME/gk-ace/ACE-Step-1.5/.venv/bin/python"
export MERT_PY="$HOME/gk-envs/mert/bin/python"
export GK_ROOT="$HOME/gk"
export GK_ACE_ROOT="$HOME/gk-ace/ACE-Step-1.5"
export GK_FFMPEG_LIB="$HOME/gk-tools/ffmpeg/lib"
export HF_HOME="$HOME/gk-hf"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
EOS
cat ~/gk-env.sh
df -h ~ | tail -1
