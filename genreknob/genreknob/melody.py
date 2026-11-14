"""Melody/harmony preservation between a source clip and a generated clip.

Primary metric ("melody"): frame-wise cosine similarity of *centred* chroma.
Chroma is computed with a constant-Q transform on the harmonic part of the signal
(HPSS), lightly median-smoothed, and each frame is centred (its mean over the 12
pitch classes is removed) before L2 normalisation. Centring removes the large
positive offset that raw chroma vectors share, so unrelated music scores near
zero and the score reflects which pitch classes stand out at each moment.
The score is maximised over a global lag of +-1 s.

Secondary metric ("chroma_cens"): plain chroma-CENS cosine similarity (as used
by several editing papers), kept for comparability.
"""
from __future__ import annotations

import numpy as np

SR = 22050
HOP = 1024          # ~43 ms frames
MAX_LAG_S = 1.0


def _features(y: np.ndarray, sr: int = SR) -> dict:
    import librosa
    from scipy.ndimage import median_filter
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    yh = librosa.effects.harmonic(y)
    c = librosa.feature.chroma_cqt(y=yh, sr=SR, hop_length=HOP)
    c = median_filter(c, size=(1, 5))
    c = c - c.mean(axis=0, keepdims=True)
    c = c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-8)
    cens = librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP)
    cens = cens / (np.linalg.norm(cens, axis=0, keepdims=True) + 1e-8)
    return {"centred": c.astype(np.float32), "cens": cens.astype(np.float32)}


def load_features(path: str) -> dict:
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return _features(y, SR)


def lagged_cosine(ca: np.ndarray, cb: np.ndarray, max_lag_s: float = MAX_LAG_S) -> float:
    """Mean frame-wise cosine similarity, maximised over a global lag."""
    max_lag = int(round(max_lag_s * SR / HOP))
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        a, b = (ca[:, lag:], cb) if lag >= 0 else (ca, cb[:, -lag:])
        n = min(a.shape[1], b.shape[1])
        if n < 20:
            continue
        best = max(best, float(np.mean(np.sum(a[:, :n] * b[:, :n], axis=0))))
    return best


def similarity(fa: dict, fb: dict) -> dict:
    return {"melody": lagged_cosine(fa["centred"], fb["centred"]),
            "chroma_cens": lagged_cosine(fa["cens"], fb["cens"])}


def features_parallel(paths, n_jobs: int = -1) -> dict:
    from joblib import Parallel, delayed
    paths = list(dict.fromkeys(paths))
    feats = Parallel(n_jobs=n_jobs, batch_size=4)(delayed(load_features)(p) for p in paths)
    return dict(zip(paths, feats))


# backwards-compatible helpers used by the demo
def load_chroma(path: str) -> dict:
    return load_features(path)


def chroma_similarity(fa: dict, fb: dict) -> float:
    return similarity(fa, fb)["melody"]
