"""Paths and constants shared by every GenreKnob script.

All paths can be overridden with environment variables so the same code runs
on the GPU server (~/gk) and on a laptop.
"""
import os
from pathlib import Path

HOME = Path(os.path.expanduser("~"))

# Code root: the folder that contains the `genreknob` package.
GK_ROOT = Path(os.environ.get("GK_ROOT", Path(__file__).resolve().parents[1]))

# Large inputs / caches live outside the code folder so they are never synced.
DATA_ROOT = Path(os.environ.get("GK_DATA", HOME / "genreknob-data"))
GTZAN_DIR = Path(os.environ.get("GK_GTZAN", DATA_ROOT / "genres_original"))
CACHE = Path(os.environ.get("GK_CACHE", DATA_ROOT / "cache"))
WORK = Path(os.environ.get("GK_WORK", HOME / "gk-work"))  # generated audio (GBs)

# Small, shareable outputs (tables, figures, logs, demo clips) are synced back.
RESULTS = Path(os.environ.get("GK_RESULTS", GK_ROOT / "results"))

SPLITS_DIR = GK_ROOT / "splits" / "fault_filtered"
ACE_ROOT = Path(os.environ.get("GK_ACE_ROOT", HOME / "gk-ace" / "ACE-Step-1.5"))

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
# Natural-language names used in text prompts (generation and CLAP judging).
GENRE_TEXT = {
    "blues": "blues", "classical": "classical", "country": "country",
    "disco": "disco", "hiphop": "hip hop", "jazz": "jazz",
    "metal": "heavy metal", "pop": "pop", "reggae": "reggae", "rock": "rock",
}

SR_HC = 22050          # sample rate for handcrafted features (as in GTZAN / Kaggle CSV)
SEG_S = 3.0            # handcrafted features: 3-s segments, 10 per track
CROP_S = 10.0          # embeddings: 10-s crops, 3 per track
SEED = 0

MERT_ID = os.environ.get("GK_MERT", "m-a-p/MERT-v1-330M")


def _clap_id():
    """CLAP checkpoint: env GK_CLAP > results/clap_choice.txt (set by `judge select-clap`) > default."""
    if os.environ.get("GK_CLAP"):
        return os.environ["GK_CLAP"]
    f = RESULTS / "clap_choice.txt"
    if f.exists() and f.read_text().strip():
        return f.read_text().strip()
    return "laion/clap-htsat-unfused"


CLAP_ID = _clap_id()


def ensure_dirs():
    for p in (CACHE, WORK, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
