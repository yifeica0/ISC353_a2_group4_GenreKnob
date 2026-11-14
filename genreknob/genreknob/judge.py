"""Genre judges for generated audio.

* CLAP zero-shot (laion/larger_clap_music): never trained on GTZAN -> primary judge.
* Genre radar: logistic regression on CLAP audio embeddings trained on the GTZAN
  train+valid split (the demo's live genre meter; secondary judge).

CLI: python -m genreknob.judge build-radar   (run in the ACE env, needs a GPU)
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .config import CACHE, GENRES, RESULTS, SEED, ensure_dirs

RADAR_PATH = CACHE / "radar_clap_lr.joblib"            # demo: trained on all GTZAN tracks
RADAR_EVAL_PATH = CACHE / "radar_clap_lr_eval.joblib"  # evaluation: train+valid only


class Judges:
    def __init__(self, device=None, with_radar: bool = True, eval_mode: bool = False):
        from .embed import ClapModelWrapper
        self.clap = ClapModelWrapper(device=device)
        self.sr = self.clap.sr
        self.radar = None
        path = RADAR_EVAL_PATH if eval_mode else RADAR_PATH
        if with_radar and path.exists():
            import joblib
            self.radar = joblib.load(path)

    def load(self, path: str, max_s: float = 10.0) -> np.ndarray:
        import librosa
        y, _ = librosa.load(path, sr=self.sr, mono=True, duration=max_s)
        return y

    def embed_arrays(self, arrays) -> np.ndarray:
        return self.clap.audio_embed(list(arrays))

    def score_embeddings(self, emb: np.ndarray) -> dict:
        out = {"clap": self.clap.zero_shot(emb)}
        if self.radar is not None:
            out["radar"] = self.radar.predict_proba(emb)
        return out

    def score_files(self, paths, batch: int = 16) -> dict:
        embs = []
        for i in range(0, len(paths), batch):
            embs.append(self.embed_arrays([self.load(p) for p in paths[i:i + batch]]))
        emb = np.concatenate(embs)
        res = self.score_embeddings(emb)
        res["emb"] = emb
        return res


def build_radar():
    """Train the demo's genre radar: CLAP embeddings (3 x 10-s crops/track) + LogReg."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .data import load_index
    from .embed import ClapModelWrapper, crops_of_track

    ensure_dirs()
    idx = load_index()
    idx = idx[idx.ok].reset_index(drop=True)
    w = ClapModelWrapper()
    X, y, tr, split = [], [], [], []
    for p, lab, t, s in zip(idx.path, idx.label, idx.track, idx.split):
        e = w.audio_embed(crops_of_track(p, w.sr))
        X.append(e); y += [lab] * len(e); tr += [t] * len(e); split += [s] * len(e)
    X = np.concatenate(X); y = np.array(y); tr = np.array(tr); split = np.array(split)

    def track_acc(model, part):
        m = split == part
        pr = model.predict_proba(X[m])
        import pandas as pd
        df = pd.DataFrame(pr); df["track"] = tr[m]; df["y"] = y[m]
        g = df.groupby("track").mean(numeric_only=True)
        return float((g[list(range(len(GENRES)))].to_numpy().argmax(1) == g["y"].to_numpy()).mean())

    best = None
    for C in (0.001, 0.01, 0.1, 1.0, 10.0):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        m.fit(X[split == "train"], y[split == "train"])
        acc = track_acc(m, "valid")
        print(f"[radar] C={C} valid_acc={acc:.3f}", flush=True)
        if best is None or acc > best[1]:
            best = (C, acc)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best[0], max_iter=5000))
    m.fit(X[np.isin(split, ["train", "valid"])], y[np.isin(split, ["train", "valid"])])
    test_acc = track_acc(m, "test")
    joblib.dump(m, RADAR_EVAL_PATH)   # never saw test tracks -> used to judge Exp 2 outputs
    # final radar used in the demo: trained on every available GTZAN track
    final = make_pipeline(StandardScaler(), LogisticRegression(C=best[0], max_iter=5000)).fit(X, y)
    joblib.dump(final, RADAR_PATH)
    info = {"C": best[0], "valid_acc": best[1], "test_acc_trainvalid_model": test_acc,
            "n_crops": int(len(X))}
    (RESULTS / "exp2").mkdir(parents=True, exist_ok=True)
    (RESULTS / "exp2" / "radar_info.json").write_text(json.dumps(info, indent=2))
    print("[radar]", info)


CLAP_CANDIDATES = ["laion/clap-htsat-unfused", "laion/clap-htsat-fused",
                   "laion/larger_clap_music_and_speech", "laion/larger_clap_general",
                   "laion/larger_clap_music"]


def select_clap(per_genre: int = 5):
    """Pick the zero-shot judge on VALIDATION tracks only (5 per genre, 10-20 s)."""
    import librosa
    from .data import load_index
    from .embed import ClapModelWrapper
    ensure_dirs()
    idx = load_index()
    idx = idx[idx.ok & (idx.split == "valid")]
    sel = idx.groupby("genre", group_keys=False).head(per_genre)
    rows = []
    for mid in CLAP_CANDIDATES:
        try:
            w = ClapModelWrapper(model_id=mid)
        except Exception as e:
            rows.append({"model": mid, "error": str(e)[:200]})
            continue
        t = w.text_embed(w.genre_prompts())
        collapse = float((t @ t.T)[np.triu_indices(len(t), 1)].mean())
        audio = [librosa.load(p, sr=w.sr, offset=10, duration=10)[0] for p in sel.path]
        pr = w.zero_shot(w.audio_embed(audio))
        acc = float((pr.argmax(1) == sel.label.to_numpy()).mean())
        rows.append({"model": mid, "valid_zero_shot_acc": acc, "mean_text_text_cos": collapse,
                     "n": int(len(sel))})
        print(rows[-1], flush=True)
        del w
    import pandas as pd
    df = pd.DataFrame(rows)
    (RESULTS / "clap_selection.csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "clap_selection.csv", index=False)
    ok = df.dropna(subset=["valid_zero_shot_acc"])
    ok = ok[ok.mean_text_text_cos < 0.95]
    best = ok.sort_values("valid_zero_shot_acc", ascending=False).iloc[0]["model"]
    (RESULTS / "clap_choice.txt").write_text(best)
    print("[select-clap] chosen:", best)


if __name__ == "__main__":
    {"build-radar": build_radar, "select-clap": select_clap}[sys.argv[1]]()
