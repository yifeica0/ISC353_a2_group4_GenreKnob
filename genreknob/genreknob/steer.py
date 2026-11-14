"""Compute CAA steering vectors inside ACE-Step 1.5 (no training of the generator).

For every concept c in a family (the 10 GTZAN genres; extra families for the demo)
we generate short clips from several prompt templates and record the mean output of
each DiT cross-attention module. The direction for c at layer l is
    v_{c,l} = mean_c(l) - mean_over_family(l)
and r_l is the mean token norm at that layer (used to scale steering strength).

CLI: python -m genreknob.steer genres|concepts
"""
from __future__ import annotations

import os
import sys
import time

import torch

from .config import CACHE, GENRE_TEXT, GENRES, WORK, ensure_dirs

GENRE_TEMPLATES = [
    "{c} music, instrumental",
    "an instrumental {c} track",
    "{c} song without vocals, studio recording",
    "a {c} piece played by a band",
]
FAMILIES = {
    "instrument": ["piano", "electric guitar", "violin", "trumpet", "saxophone", "synthesizer"],
    "mood": ["happy", "sad", "energetic", "calm"],
}
FAMILY_TEMPLATES = {
    "instrument": ["instrumental music featuring {c}", "a song with a prominent {c}",
                   "{c} solo over a band", "music played on {c}"],
    "mood": ["{c} instrumental music", "a {c} song without vocals",
             "music that sounds {c}", "a {c} instrumental track"],
}


def collect(ace, prompts_by_concept: dict, n_seeds: int, duration: float = 10.0):
    means, norms = {}, {}
    tmp = WORK / f"steer_tmp_{os.getpid()}.wav"
    for c, prompts in prompts_by_concept.items():
        store = {}
        t0 = time.time()
        with ace.steering.capture(store):
            for i, p in enumerate(prompts):
                for s in range(n_seeds):
                    ace.generate(tmp, p, task="text2music", seed=1000 + 37 * s + i, duration=duration)
        means[c] = {l: v[0] / v[1] for l, v in store.items()}
        norms[c] = {l: v[2] / v[1] for l, v in store.items()}
        print(f"[steer] {c}: {len(prompts) * n_seeds} clips, {time.time() - t0:.0f}s", flush=True)
    return means, norms


def to_vectors(concepts, means, norms):
    layers = sorted(next(iter(means.values())).keys())
    M = torch.stack([torch.stack([means[c][l] for l in layers]) for c in concepts])  # (C, L, H)
    v = M - M.mean(0, keepdim=True)
    r = torch.tensor([sum(norms[c][l] for c in concepts) / len(concepts) for l in layers])
    return {"genres": list(concepts), "layers": layers, "v": v.float(), "r": r.float()}


def main(what: str):
    from .ace import Ace
    ensure_dirs()
    ace = Ace()
    print("[steer] DiT cross-attention layers:", ace.steering.layers, flush=True)
    if what == "genres":
        prompts = {g: [t.format(c=GENRE_TEXT[g]) for t in GENRE_TEMPLATES] for g in GENRES}
        means, norms = collect(ace, prompts, n_seeds=8)
        d = to_vectors(GENRES, means, norms)
        torch.save(d, CACHE / "genre_vectors.pt")
        cos = torch.nn.functional.normalize(d["v"], dim=-1)
        mid = len(d["layers"]) // 2
        sim = cos[:, mid] @ cos[:, mid].T
        print("[steer] cosine similarity between genre directions at the middle layer:")
        print(sim.numpy().round(2))
    else:
        for fam, concepts in FAMILIES.items():
            prompts = {c: [t.format(c=c) for t in FAMILY_TEMPLATES[fam]] for c in concepts}
            means, norms = collect(ace, prompts, n_seeds=4)
            torch.save(to_vectors(concepts, means, norms), CACHE / f"{fam}_vectors.pt")
    print("[steer] done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
