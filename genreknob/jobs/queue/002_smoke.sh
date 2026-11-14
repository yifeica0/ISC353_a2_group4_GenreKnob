#!/usr/bin/env bash
# Job 002: smoke tests in both envs (model downloads happen here).
source ~/gk-env.sh
cd ~/gk
mkdir -p results/smoke
SRC=~/genreknob-data/genres_original/classical/classical.00010.wav

echo "===== MERT env: MERT + CLAP (transformers 4.38) + handcrafted features"
CUDA_VISIBLE_DEVICES=3 $MERT_PY - << 'PY'
import time, torch, librosa, numpy as np, os
from transformers import AutoModel, Wav2Vec2FeatureExtractor
src = os.path.expanduser("~/genreknob-data/genres_original/classical/classical.00010.wav")
t = time.time()
m = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).cuda().eval()
p = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
y, _ = librosa.load(src, sr=p.sampling_rate, duration=10)
inp = {k: v.cuda() for k, v in p(y, sampling_rate=p.sampling_rate, return_tensors="pt").items()}
with torch.no_grad():
    hs = m(**inp, output_hidden_states=True).hidden_states
print("MERT ok: layers", len(hs), tuple(hs[-1].shape), "sr", p.sampling_rate, f"{time.time()-t:.1f}s", flush=True)
from genreknob.embed import ClapModelWrapper
w = ClapModelWrapper()
e = w.audio_embed([librosa.load(src, sr=w.sr, duration=10)[0]])
print("CLAP(4.38) ok:", e.shape, "zero-shot:", dict(zip(["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"], w.zero_shot(e)[0].round(3))), flush=True)
from genreknob.features import hc_segments_of_file
t = time.time(); f = hc_segments_of_file(src); print("HC ok:", len(f), "segments", len(f[0]), "features", f"{time.time()-t:.1f}s")
PY

echo "===== ACE env: generation, tasks, steering hooks, CLAP judge"
CUDA_VISIBLE_DEVICES=0 $ACE_PY - << 'PY'
import time, os, numpy as np, librosa, soundfile as sf, torch
from genreknob.config import RESULTS
out = RESULTS / "smoke"; out.mkdir(parents=True, exist_ok=True)
src_full = os.path.expanduser("~/genreknob-data/genres_original/classical/classical.00010.wav")
y, _ = librosa.load(src_full, sr=48000, offset=10, duration=10)
src = str(out / "source_classical.wav"); sf.write(src, y, 48000)
from genreknob.ace import Ace
t = time.time(); ace = Ace(); print(f"ACE init {time.time()-t:.1f}s", flush=True)
print("cross-attn layers:", ace.steering.layers, flush=True)
print("model class:", type(ace.h.model).__name__, flush=True)
cap = "heavy metal music, instrumental"
for name, kw in [("text2music", dict(task="text2music")),
                 ("cover_s0.6", dict(task="cover", src=src, cover_strength=0.6)),
                 ("cover_s1.0", dict(task="cover", src=src, cover_strength=1.0)),
                 ("repaint_4-8s", dict(task="repaint", src=src, repaint=(4.0, 8.0)))]:
    for rep in range(2):
        t = time.time(); f = ace.generate(out / f"{name}.wav", cap, duration=10, seed=11, **kw)
        dt = time.time() - t
    a, sr = sf.read(f)
    print(f"{name}: {dt:.2f}s/clip (2nd run), shape {a.shape}, sr {sr}, rms {np.sqrt((a**2).mean()):.4f}", flush=True)
store = {}
with ace.steering.capture(store):
    ace.generate(out / "capture.wav", "jazz music, instrumental", duration=10, seed=3)
for l in sorted(store)[:3] + sorted(store)[-2:]:
    s, n, nn = store[l]
    print(f"layer {l}: tokens {n}, mean |out| {nn/n:.3f}, |mean| {float((s/n).norm()):.3f}", flush=True)
mid = ace.steering.layers[len(ace.steering.layers)//2]
d = torch.randn(store[mid][0].shape[0]); d = d / d.norm() * (store[mid][2] / store[mid][1])
t = time.time(); ace.generate(out / "steer_random.wav", cap, task="cover", src=src, cover_strength=0.6, duration=10, seed=11, deltas={mid: d})
print(f"steered generation ok {time.time()-t:.2f}s", flush=True)
from genreknob.judge import Judges
J = Judges(with_radar=False)
files = [src] + [str(out / f"{n}.wav") for n in ("text2music", "cover_s0.6", "cover_s1.0", "repaint_4-8s", "steer_random")]
r = J.score_files(files)
G = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
for f, p in zip(files, r["clap"]):
    print(f"CLAP judge {os.path.basename(f)}: top={G[int(p.argmax())]} p_metal={p[6]:.2f} p_classical={p[1]:.2f}")
from genreknob.melody import load_chroma, chroma_similarity
cs = load_chroma(src)
for f in files[1:]:
    print(f"melody kept {os.path.basename(f)}: {chroma_similarity(cs, load_chroma(f)):.3f}")
PY

echo "===== Gradio UI build"
$ACE_PY -m genreknob.app --build-only
for f in results/smoke/*.wav; do ~/gk-tools/ffmpeg/bin/ffmpeg -loglevel error -y -i "$f" -b:a 128k "${f%.wav}.mp3" && rm "$f"; done
ls -la results/smoke
