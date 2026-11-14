"""Pretrained audio representations: MERT (all layers) and CLAP (audio/text).

CLI:
  python -m genreknob.embed mert   # -> CACHE/mert_crops.npy  (n_crops, 25, 1024) float16
  python -m genreknob.embed clap   # -> CACHE/clap_crops.npy  (n_crops, 512) float32
  python -m genreknob.embed hc     # -> CACHE/hc_segments.csv (n_tracks*10, 57 + ids)
Each track contributes 3 crops of 10 s (0-10, 10-20, 20-30 s) for MERT/CLAP and
10 segments of 3 s for the handcrafted features.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from .config import CACHE, CLAP_ID, CROP_S, GENRE_TEXT, GENRES, MERT_ID, ensure_dirs
from .data import load_audio, load_index, split_equal

N_CROPS = 3
ZS_TEMPERATURE = 100.0


def _as_tensor(x):
    """transformers versions differ in what get_*_features returns."""
    import torch
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("audio_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(x, attr) and getattr(x, attr) is not None:
            return getattr(x, attr)
    return x[0]


class ClapModelWrapper:
    """CLAP audio/text embeddings + zero-shot genre probabilities."""

    def __init__(self, model_id: str = CLAP_ID, device: str | None = None):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = ClapProcessor.from_pretrained(model_id)
        self.sr = self.proc.feature_extractor.sampling_rate  # 48000
        self._text_cache = {}

    def _proc_audio(self, arrays):
        # Call the feature extractor directly: the processor's audio keyword changed
        # across transformers versions (audios= vs audio=).
        return self.proc.feature_extractor(arrays, sampling_rate=self.sr, return_tensors="pt")

    def audio_embed(self, arrays: list[np.ndarray], batch: int = 16) -> np.ndarray:
        """L2-normalised audio embeddings for mono arrays sampled at self.sr."""
        torch = self.torch
        out = []
        for i in range(0, len(arrays), batch):
            inp = self._proc_audio([a.astype(np.float32) for a in arrays[i:i + batch]])
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with torch.no_grad():
                e = _as_tensor(self.model.get_audio_features(**inp))
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    def text_embed(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        key = tuple(texts)
        if key not in self._text_cache:
            inp = self.proc(text=list(texts), return_tensors="pt", padding=True)
            inp = {k: v.to(self.device) for k, v in inp.items()}
            with torch.no_grad():
                e = _as_tensor(self.model.get_text_features(**inp))
            self._text_cache[key] = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
        return self._text_cache[key]

    def genre_prompts(self) -> list[str]:
        return [f"This is a {GENRE_TEXT[g]} song." for g in GENRES]

    def zero_shot(self, audio_emb: np.ndarray) -> np.ndarray:
        """Softmax genre probabilities (n, 10) from L2-normalised audio embeddings."""
        t = self.text_embed(self.genre_prompts())
        logits = ZS_TEMPERATURE * audio_emb @ t.T   # fixed CLIP-style temperature for all checkpoints
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(1, keepdims=True)


def crops_of_track(path: str, sr: int) -> list[np.ndarray]:
    y = load_audio(path, sr)
    return split_equal(y, sr, CROP_S, N_CROPS)


def _crop_index(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"track": t, "crop": c} for t in df["track"] for c in range(N_CROPS)]
    return pd.DataFrame(rows)


def run_clap():
    df = load_index()
    df = df[df.ok].reset_index(drop=True)
    w = ClapModelWrapper()
    embs = []
    t0 = time.time()
    for i, p in enumerate(df["path"]):
        embs.append(w.audio_embed(crops_of_track(p, w.sr)))
        if i % 100 == 0:
            print(f"[clap] {i}/{len(df)} {time.time() - t0:.0f}s", flush=True)
    E = np.concatenate(embs).astype(np.float32)
    np.save(CACHE / "clap_crops.npy", E)
    _crop_index(df).to_csv(CACHE / "clap_crops_index.csv", index=False)
    zs = w.zero_shot(E)
    np.save(CACHE / "clap_zeroshot_crops.npy", zs.astype(np.float32))
    print("[clap] saved", E.shape)


def load_mert(device: str):
    """Load MERT and restore its positional-conv weights.

    MERT checkpoints store the weight-normalised positional convolution as
    weight_g / weight_v; with torch>=2.1 transformers builds it with the new
    parametrisation (original0 / original1) and silently re-initialises it.
    We copy the checkpoint tensors back and verify the result.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    model = AutoModel.from_pretrained(MERT_ID, trust_remote_code=True)
    proc = Wav2Vec2FeatureExtractor.from_pretrained(MERT_ID, trust_remote_code=True)
    sd = None
    for fname in ("model.safetensors", "pytorch_model.bin"):
        try:
            path = hf_hub_download(MERT_ID, fname)
        except Exception:
            continue
        if fname.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(path)
        else:
            sd = torch.load(path, map_location="cpu")
        break
    conv = model.encoder.pos_conv_embed.conv
    g = sd.get("encoder.pos_conv_embed.conv.weight_g")
    v = sd.get("encoder.pos_conv_embed.conv.weight_v")
    if g is not None and hasattr(conv, "parametrizations"):
        with torch.no_grad():
            conv.parametrizations.weight.original0.copy_(g)
            conv.parametrizations.weight.original1.copy_(v)
        print("[mert] restored positional-conv weight_g/weight_v from the checkpoint", flush=True)
    elif g is not None and hasattr(conv, "weight_g"):
        print("[mert] positional conv uses weight_g/weight_v directly (loaded normally)", flush=True)
    else:
        print("[mert] WARNING: could not find positional-conv weights in the checkpoint", flush=True)
    return model.to(device).eval(), proc


def run_mert(batch: int = 6):
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    df = load_index()
    df = df[df.ok].reset_index(drop=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_mert(dev)
    sr = proc.sampling_rate  # 24000
    all_crops = []
    for p in df["path"]:
        all_crops.extend(crops_of_track(p, sr))
    print(f"[mert] {len(all_crops)} crops at {sr} Hz", flush=True)
    out = []
    t0 = time.time()
    for i in range(0, len(all_crops), batch):
        chunk = all_crops[i:i + batch]
        inp = proc(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            hs = model(**inp, output_hidden_states=True).hidden_states
        h = torch.stack([x.float().mean(1) for x in hs], dim=1)  # (B, n_layers, dim)
        out.append(h.cpu().numpy().astype(np.float16))
        if (i // batch) % 50 == 0:
            print(f"[mert] {i}/{len(all_crops)} {time.time() - t0:.0f}s", flush=True)
    M = np.concatenate(out)
    np.save(CACHE / "mert_crops.npy", M)
    _crop_index(df).to_csv(CACHE / "mert_crops_index.csv", index=False)
    print("[mert] saved", M.shape)


def run_hc(n_jobs: int = -1):
    from joblib import Parallel, delayed
    from .features import FEATURE_NAMES, hc_segments_of_file
    df = load_index()
    df = df[df.ok].reset_index(drop=True)
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(hc_segments_of_file)(p) for p in df["path"])
    rows = []
    for track, segs in zip(df["track"], res):
        for j, f in enumerate(segs):
            rows.append({"track": track, "seg": j, **f})
    out = pd.DataFrame(rows)[["track", "seg"] + FEATURE_NAMES]
    out.to_csv(CACHE / "hc_segments.csv", index=False)
    print(f"[hc] saved {out.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ensure_dirs()
    what = sys.argv[1]
    {"clap": run_clap, "mert": run_mert, "hc": run_hc}[what]()
