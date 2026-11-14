"""Handcrafted features: the 57 descriptors of the Kaggle GTZAN CSV, re-extracted
with one librosa routine so that real and generated audio are described identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SR_HC, SEG_S

BASE = ["chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth",
        "rolloff", "zero_crossing_rate", "harmony", "perceptr"]
FEATURE_NAMES = (
    [f"{b}_{s}" for b in BASE for s in ("mean", "var")]
    + ["tempo"]
    + [f"mfcc{i}_{s}" for i in range(1, 21) for s in ("mean", "var")]
)
assert len(FEATURE_NAMES) == 57

# Feature groups used for SHAP aggregation and ablation.
GROUPS = {
    "timbre (MFCC)": [f"mfcc{i}_{s}" for i in range(1, 21) for s in ("mean", "var")],
    "spectral shape": [f"{b}_{s}" for b in ("spectral_centroid", "spectral_bandwidth",
                                            "rolloff", "zero_crossing_rate")
                       for s in ("mean", "var")],
    "energy": ["rms_mean", "rms_var"],
    "harmony / pitch": ["chroma_stft_mean", "chroma_stft_var", "harmony_mean", "harmony_var"],
    "rhythm / percussive": ["tempo", "perceptr_mean", "perceptr_var"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURE_NAMES)


def _tempo(y, sr):
    import librosa
    try:
        t = librosa.feature.tempo(y=y, sr=sr)
    except AttributeError:  # librosa < 0.10
        t = librosa.beat.tempo(y=y, sr=sr)
    return float(np.atleast_1d(t)[0])


def hc_features(y: np.ndarray, sr: int = SR_HC) -> dict:
    """57 handcrafted descriptors of one audio segment (mean/var over frames)."""
    import librosa
    f = {}
    mats = {
        "chroma_stft": librosa.feature.chroma_stft(y=y, sr=sr),
        "rms": librosa.feature.rms(y=y),
        "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr),
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr),
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(y),
    }
    harm, perc = librosa.effects.hpss(y)
    mats["harmony"] = harm
    mats["perceptr"] = perc
    for b in BASE:
        m = np.asarray(mats[b], dtype=np.float64)
        f[f"{b}_mean"] = float(m.mean())
        f[f"{b}_var"] = float(m.var())
    f["tempo"] = _tempo(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        f[f"mfcc{i + 1}_mean"] = float(mfcc[i].mean())
        f[f"mfcc{i + 1}_var"] = float(mfcc[i].var())
    return f


def hc_segments_of_file(path: str, n_seg: int = 10, seg_s: float = SEG_S,
                        offset: float = 0.0) -> list[dict]:
    """Features of `n_seg` consecutive `seg_s`-second segments of an audio file."""
    from .data import load_audio, split_equal
    y = load_audio(path, SR_HC, offset=offset, duration=n_seg * seg_s)
    return [hc_features(seg) for seg in split_equal(y, SR_HC, seg_s, n_seg)]


def hc_segments_of_array(y: np.ndarray, sr: int, seg_s: float = SEG_S) -> pd.DataFrame:
    """Features of all whole `seg_s` segments in an array (resampled to SR_HC)."""
    import librosa
    if sr != SR_HC:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR_HC)
    n = int(seg_s * SR_HC)
    n_seg = max(1, len(y) // n)
    from .data import split_equal
    rows = [hc_features(s) for s in split_equal(y, SR_HC, seg_s, n_seg)]
    return pd.DataFrame(rows)[FEATURE_NAMES]
