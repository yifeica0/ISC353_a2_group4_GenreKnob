"""ACE-Step 1.5 wrapper (turbo DiT, no language model) + activation steering.

Steering follows TADA (Staniszewski et al., 2026): add a concept direction to
the output of selected DiT cross-attention modules at every denoising step.
Direction = contrastive activation addition (CAA): the mean cross-attention
output for prompts of genre g minus the mean over all genres.
"""
from __future__ import annotations

import contextlib
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .config import ACE_ROOT


class Steering:
    """Forward hooks on decoder.layers.<i>.cross_attn of the ACE-Step DiT."""

    def __init__(self, model):
        self.modules = {}
        for name, m in model.named_modules():
            mt = re.search(r"(?:^|\.)decoder\.layers\.(\d+)\.cross_attn$", name)
            if mt:
                self.modules[int(mt.group(1))] = m
        if not self.modules:
            raise RuntimeError("no cross-attention modules found in the ACE-Step DiT")

    @property
    def layers(self):
        return sorted(self.modules)

    @contextlib.contextmanager
    def capture(self, store: dict):
        """Accumulate per-layer sums of cross-attention outputs (all tokens, all steps)."""
        def make(layer):
            def hook(_mod, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                x = h.detach().float().reshape(-1, h.shape[-1])
                s = store.setdefault(layer, [torch.zeros(h.shape[-1], dtype=torch.float64), 0, 0.0])
                s[0] += x.sum(0).double().cpu()
                s[1] += x.shape[0]
                s[2] += float(x.norm(dim=-1).sum())
            return hook
        handles = [m.register_forward_hook(make(l)) for l, m in self.modules.items()]
        try:
            yield store
        finally:
            for h in handles:
                h.remove()

    @contextlib.contextmanager
    def steer(self, deltas: dict | None):
        """Add deltas[layer] (a hidden-size vector) to that layer's cross-attention output."""
        if not deltas:
            yield
            return

        def make(d):
            def hook(_mod, _inp, out):
                if isinstance(out, tuple):
                    h = out[0]
                    return (h + d.to(device=h.device, dtype=h.dtype),) + tuple(out[1:])
                return out + d.to(device=out.device, dtype=out.dtype)
            return hook
        handles = [self.modules[l].register_forward_hook(make(d))
                   for l, d in deltas.items() if l in self.modules]
        try:
            yield
        finally:
            for h in handles:
                h.remove()


class GenreVectors:
    """CAA genre directions per layer, saved by genreknob.steer."""

    def __init__(self, path):
        d = torch.load(path, map_location="cpu")
        self.genres = d["genres"]
        self.layers = d["layers"]
        self.v = d["v"].float()      # (G, L, H) centred mean differences
        self.r = d["r"].float()      # (L,) mean token norm of the cross-attention output

    def deltas(self, weights: dict, layers, alpha: float) -> dict:
        """alpha * r_l * normalise(sum_g w_g * unit(v_{g,l})) for each chosen layer."""
        out = {}
        for layer in layers:
            li = self.layers.index(layer)
            vec = torch.zeros(self.v.shape[-1])
            for g, w in weights.items():
                if w == 0:
                    continue
                u = self.v[self.genres.index(g), li]
                vec += w * u / (u.norm() + 1e-8)
            if vec.norm() > 0:
                out[layer] = alpha * float(self.r[li]) * vec / (vec.norm() + 1e-8)
        return out


class Ace:
    """Thin wrapper around ACE-Step's generate_music (DiT only, deterministic seeds)."""

    def __init__(self, device: str = "cuda", config_path: str = "acestep-v15-turbo", ace_root=ACE_ROOT,
                 use_lm: bool = False):
        ace_root = str(ace_root)
        if ace_root not in sys.path:
            sys.path.insert(0, ace_root)
        from acestep.handler import AceStepHandler
        self.h = AceStepHandler()
        res = self.h.initialize_service(project_root=ace_root, config_path=config_path, device=device)
        msg, ok = res if isinstance(res, tuple) and len(res) == 2 else (str(res), True)
        if not ok:
            raise RuntimeError(f"ACE-Step init failed: {msg}")
        self.steering = Steering(self.h.model)
        self.tmp = tempfile.mkdtemp(prefix="gk_ace_")
        self.sample_rate = 48000
        self.llm = self._init_lm(ace_root, device) if use_lm else None

    @staticmethod
    def _init_lm(ace_root, device):
        """Load ACE-Step's 5 Hz planning LM (needed for thinking=True); None if it fails."""
        import inspect
        try:
            from acestep.llm_inference import LLMHandler
            llm = LLMHandler()
            sig = inspect.signature(llm.initialize)
            print("[ace] LLMHandler.initialize", sig, flush=True)
            cand = {"checkpoint_dir": os.path.join(ace_root, "checkpoints"),
                    "lm_model_path": "acestep-5Hz-lm-1.7B", "backend": "pt", "device": device,
                    "project_root": ace_root, "offload_to_cpu": False}
            kw = {k: v for k, v in cand.items() if k in sig.parameters}
            res = llm.initialize(**kw)
            print("[ace] LM init:", str(res)[:300], "initialized =", getattr(llm, "llm_initialized", None), flush=True)
            return llm if getattr(llm, "llm_initialized", False) else None
        except Exception as e:
            print("[ace] LM init failed:", repr(e)[:300], flush=True)
            return None

    def generate(self, out_path, caption: str, task: str = "text2music", src=None,
                 duration: float = 10.0, seed: int = 0, cover_strength: float = 1.0,
                 steps: int = 8, bpm=None, keyscale: str = "", repaint=None,
                 deltas: dict | None = None, extra: dict | None = None, reference=None,
                 thinking: bool = False, shift: float = 3.0) -> str:
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
        import soundfile as sf
        kw = dict(task_type=task, caption=caption, lyrics="[Instrumental]", instrumental=True,
                  duration=float(duration), seed=int(seed), inference_steps=int(steps),
                  thinking=False, use_cot_metas=False, use_cot_caption=False,
                  use_cot_language=False, audio_cover_strength=float(cover_strength), shift=float(shift))
        if thinking and self.llm is not None:
            kw["thinking"] = True
        if src is not None:
            kw["src_audio"] = str(src)
        if reference is not None:
            kw["reference_audio"] = str(reference)
        if bpm:
            kw["bpm"] = int(bpm)
        if keyscale:
            kw["keyscale"] = keyscale
        if repaint is not None:
            kw.update(repainting_start=float(repaint[0]), repainting_end=float(repaint[1]),
                      chunk_mask_mode="explicit")
        if extra:
            kw.update(extra)
        params = GenerationParams(**kw)
        cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[int(seed)], audio_format="wav32")
        with self.steering.steer(deltas):
            r = generate_music(self.h, self.llm, params, cfg, save_dir=self.tmp)
        if not r.success or not r.audios:
            raise RuntimeError(f"generation failed: {r.error or r.status_message}")
        a = r.audios[0]
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        t = a.get("tensor")
        if t is not None:
            arr = t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
            if arr.ndim == 2 and arr.shape[0] <= 2:   # (channels, samples) -> (samples, channels)
                arr = arr.T
            sf.write(str(out_path), arr, int(a.get("sample_rate", 48000)), subtype="PCM_16")
        else:
            data, sr = sf.read(a["path"])
            sf.write(str(out_path), data, sr, subtype="PCM_16")
        p = a.get("path")
        if p and os.path.exists(p) and str(p).startswith(self.tmp):
            os.remove(p)
        return str(out_path)
