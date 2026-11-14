"""GTZAN index, the artist-disjoint "fault-filtered" split, and audio helpers."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GENRES, GTZAN_DIR, SPLITS_DIR


def _read_list(name: str) -> list[str]:
    path = SPLITS_DIR / f"{name}_filtered.txt"
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(Path(line).name.replace(".wav", ""))  # e.g. "blues.00029"
    return out


def load_index(check_audio: bool = True) -> pd.DataFrame:
    """Return one row per GTZAN track in the fault-filtered split.

    Columns: track, genre, label (int), split (train/valid/test), path, ok.
    The split (Kereliuk, Sturm & Larsen 2015) keeps artists disjoint between
    train/valid/test and drops 70 faulty excerpts. Unreadable files (the
    Kaggle copy of jazz.00054 is corrupted) are flagged ok=False.
    """
    rows = []
    for split in ("train", "valid", "test"):
        for track in _read_list(split):
            genre = track.split(".")[0]
            rows.append({
                "track": track,
                "genre": genre,
                "label": GENRES.index(genre),
                "split": split,
                "path": str(GTZAN_DIR / genre / f"{track}.wav"),
            })
    df = pd.DataFrame(rows)
    if check_audio:
        import soundfile as sf
        ok = []
        for p in df["path"]:
            try:
                info = sf.info(p)
                ok.append(info.frames > 0)
            except Exception:
                ok.append(False)
        df["ok"] = ok
        bad = df.loc[~df["ok"], "track"].tolist()
        if bad:
            warnings.warn(f"Skipping unreadable GTZAN files: {bad}")
    else:
        df["ok"] = True
    return df.reset_index(drop=True)


def load_audio(path: str, sr: int, mono: bool = True, offset: float = 0.0,
               duration: float | None = None) -> np.ndarray:
    """Load audio as float32 mono at `sr` (librosa, soxr resampling)."""
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=mono, offset=offset, duration=duration)
    return y.astype(np.float32)


def fix_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def split_equal(y: np.ndarray, sr: int, seg_s: float, n_seg: int) -> list[np.ndarray]:
    """Cut `n_seg` non-overlapping segments of `seg_s` seconds (zero-pad the tail)."""
    n = int(round(seg_s * sr))
    y = fix_length(y, n * n_seg)
    return [y[i * n:(i + 1) * n] for i in range(n_seg)]
