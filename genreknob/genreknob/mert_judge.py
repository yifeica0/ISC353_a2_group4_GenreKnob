"""A second, CLAP-independent genre judge: MERT layer-5 embedding + logistic regression.

Fitted on GTZAN train+valid crops (the Exp 1 setting chosen on validation data:
layer 5, C = 0.001), so it never saw the test tracks. Used to check whether the
CLAP judge's verdicts on generated audio are a CLAP artefact.

Run in the MERT env:
  python -m genreknob.mert_judge fit
  python -m genreknob.mert_judge score <in.csv> <path column> <out.csv>
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from .config import CACHE, GENRES, RESULTS

LAYER, C = 5, 0.001
PATH = CACHE / "mert_judge_lr.joblib"


def fit():
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from .data import load_index
    idx = load_index(check_audio=False)
    M = np.load(CACHE / "mert_crops.npy", mmap_mode="r")
    ci = pd.read_csv(CACHE / "mert_crops_index.csv")
    lab = ci.track.map(idx.set_index("track").label)
    split = ci.track.map(idx.set_index("track").split)
    m = split.isin(["train", "valid"]).to_numpy()
    X = np.asarray(M[m, LAYER, :], dtype=np.float32)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X, lab[m].to_numpy())
    te = (split == "test").to_numpy()
    pr = clf.predict_proba(np.asarray(M[te, LAYER, :], dtype=np.float32))
    df = pd.DataFrame(pr); df["track"] = ci.track[te].to_numpy()
    g = df.groupby("track").mean()
    acc = float((g.to_numpy().argmax(1) == g.index.map(idx.set_index("track").label).to_numpy()).mean())
    joblib.dump(clf, PATH)
    info = {"layer": LAYER, "C": C, "test_track_acc": acc, "n_train_crops": int(m.sum())}
    (RESULTS / "exp2").mkdir(parents=True, exist_ok=True)
    (RESULTS / "exp2" / "mert_judge_info.json").write_text(json.dumps(info, indent=2))
    print("[mert-judge]", info, flush=True)


def score(in_csv, col, out_csv, batch: int = 8):
    import joblib
    import librosa
    import torch
    from .embed import load_mert
    df = pd.read_csv(in_csv)
    df = df[df[col].notna() & (df[col].astype(str) != "")].reset_index(drop=True)
    clf = joblib.load(PATH)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = load_mert(dev)
    sr = proc.sampling_rate
    feats = []
    paths = df[col].tolist()
    for i in range(0, len(paths), batch):
        wavs = [librosa.load(p, sr=sr, mono=True, duration=10.0)[0] for p in paths[i:i + batch]]
        inp = proc(wavs, sampling_rate=sr, return_tensors="pt", padding=True)
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            hs = model(**inp, output_hidden_states=True).hidden_states
        feats.append(hs[LAYER].float().mean(1).cpu().numpy())
    pr = clf.predict_proba(np.concatenate(feats))
    df["top_mert"] = [GENRES[i] for i in pr.argmax(1)]
    if "target" in df:
        tgt = df.target.map(GENRES.index).to_numpy()
        df["p_mert_target"] = pr[np.arange(len(df)), tgt]
        df["hit_mert"] = pr.argmax(1) == tgt
    df.to_csv(out_csv, index=False)
    print(f"[mert-judge] scored {len(df)} files -> {out_csv}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit()
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4])
