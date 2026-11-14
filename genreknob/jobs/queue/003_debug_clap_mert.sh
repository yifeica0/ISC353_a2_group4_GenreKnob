#!/usr/bin/env bash
# Job 003: diagnose CLAP zero-shot (uniform output in job 002) in both envs; verify MERT fix.
source ~/gk-env.sh
cd ~/gk
cat > /tmp/gk_clap_debug.py << 'PY'
import os, numpy as np, torch, librosa, transformers
from transformers import ClapModel, ClapProcessor
print("transformers", transformers.__version__)
mid = "laion/larger_clap_music"
m = ClapModel.from_pretrained(mid).cuda().eval()
proc = ClapProcessor.from_pretrained(mid)
fe = proc.feature_extractor
print("fe:", type(fe).__name__, "sr", fe.sampling_rate, "truncation", getattr(fe, "truncation", None),
      "padding", getattr(fe, "padding", None), "max_length_s", getattr(fe, "max_length_s", None))
print("audio cfg enable_fusion:", getattr(m.config.audio_config, "enable_fusion", None))
print("logit_scale_a", float(m.logit_scale_a), "exp", float(m.logit_scale_a.exp()),
      "logit_scale_t", float(m.logit_scale_t))
G = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
texts = [f"This is a {g} song." for g in G]
ti = proc.tokenizer(texts, return_tensors="pt", padding=True)
with torch.no_grad():
    t = m.get_text_features(**{k: v.cuda() for k, v in ti.items()})
t = t if isinstance(t, torch.Tensor) else t.pooler_output
print("text emb", tuple(t.shape), "norms", t.norm(dim=-1)[:3].tolist())
tn = torch.nn.functional.normalize(t.float(), dim=-1)
print("text-text cos (first 3):", (tn[:3] @ tn[:3].T).cpu().numpy().round(3).tolist())
for g in ["classical", "metal", "reggae"]:
    p = os.path.expanduser(f"~/genreknob-data/genres_original/{g}/{g}.00010.wav")
    y, _ = librosa.load(p, sr=fe.sampling_rate, duration=10)
    fi = fe([y], sampling_rate=fe.sampling_rate, return_tensors="pt")
    print(g, {k: (tuple(v.shape), v.dtype) for k, v in fi.items()}, "is_longer", fi.get("is_longer"))
    with torch.no_grad():
        a = m.get_audio_features(**{k: v.cuda() for k, v in fi.items()})
    a = a if isinstance(a, torch.Tensor) else a.pooler_output
    an = torch.nn.functional.normalize(a.float(), dim=-1)
    cos = (an @ tn.T)[0].cpu().numpy()
    print(f"  audio emb norm {float(a.norm()):.3f}; cos to prompts:", dict(zip(G, cos.round(3))))
    pr = torch.softmax(m.logit_scale_a.exp().float() * an @ tn.T, -1)[0].cpu().numpy()
    print("  zero-shot top:", G[int(pr.argmax())], pr.round(3).tolist())
PY
echo "===== CLAP in MERT env"
CUDA_VISIBLE_DEVICES=3 $MERT_PY /tmp/gk_clap_debug.py 2>&1 | grep -v Warning
echo "===== CLAP in ACE env"
CUDA_VISIBLE_DEVICES=1 $ACE_PY /tmp/gk_clap_debug.py 2>&1 | grep -v Warning
echo "===== MERT positional-conv fix"
CUDA_VISIBLE_DEVICES=3 $MERT_PY - << 'PY' 2>&1 | grep -v "Warning\|warn"
import torch
from huggingface_hub import hf_hub_download
from genreknob.embed import load_mert
m, p = load_mert("cuda")
conv = m.encoder.pos_conv_embed.conv
w = conv.weight.detach().float().cpu()
for fname in ("model.safetensors", "pytorch_model.bin"):
    try:
        path = hf_hub_download("m-a-p/MERT-v1-330M", fname); break
    except Exception as e:
        print("no", fname)
if path.endswith(".safetensors"):
    from safetensors.torch import load_file; sd = load_file(path)
else:
    sd = torch.load(path, map_location="cpu")
g, v = sd["encoder.pos_conv_embed.conv.weight_g"].float(), sd["encoder.pos_conv_embed.conv.weight_v"].float()
dim = [d for d in range(v.dim()) if g.shape[d] != 1]
w_ref = torch._weight_norm(v, g, dim=dim[0] if dim else 0)
print("pos-conv weight max abs diff vs checkpoint:", float((w - w_ref).abs().max()), "g shape", tuple(g.shape), "v shape", tuple(v.shape))
PY
