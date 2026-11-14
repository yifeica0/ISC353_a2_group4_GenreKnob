"""Diagnostics for Experiment 2: are the generator, the judges and the melody metric sane?

Checks (all on validation sources, never on test tracks):
  1. judges on the untouched source clips (should be ~70-80% right);
  2. reconstruction: repaint only the last second of a source clip, so 90% is the
     VAE-decoded original -> judges should agree with the source genre and the
     melody score should be high;
  3. text2music with the short caption used in Exp 2 vs a richer caption,
     at 10 s and at 30 s (scored on seconds 10-20);
  4. cover with the source's own genre as caption (strength 1.0);
  5. signal statistics (peak, RMS, clipping, spectral flatness) of every output.

Run in the ACE env: python -m genreknob.diag
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .config import GENRE_TEXT, GENRES, RESULTS, WORK

OUT = RESULTS / "diag"
W = WORK / "diag"

RICH = {
    "blues": "blues, electric guitar, slow shuffle, harmonica, instrumental",
    "classical": "classical, orchestra, strings, piano, instrumental",
    "country": "country, acoustic guitar, banjo, fiddle, instrumental",
    "disco": "disco, four-on-the-floor beat, funky bass, strings, instrumental",
    "hiphop": "hip hop, boom bap drums, rap beat, instrumental",
    "jazz": "jazz, swing, upright bass, saxophone, piano, instrumental",
    "metal": "heavy metal, distorted guitars, double bass drums, instrumental",
    "pop": "pop, catchy synths, upbeat, instrumental",
    "reggae": "reggae, offbeat guitar skank, dub bass, instrumental",
    "rock": "rock, electric guitars, drums, bass, instrumental",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stats(path):
    import librosa
    y, sr = librosa.load(path, sr=None, mono=True)
    flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    return {"sr": sr, "dur": len(y) / sr, "peak": float(np.abs(y).max()),
            "rms_db": float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)),
            "clip_frac": float(np.mean(np.abs(y) > 0.98)), "flatness": flat,
            "centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))}


def suite2_jobs(r, g):
    """Swap mechanisms that start from the source instead of from noise (all with shift 3)."""
    jobs = []
    for k, tg in enumerate(GENRES):
        if k % 5 != GENRES.index(g) % 5:
            continue
        cap = f"{GENRE_TEXT[tg]} music, instrumental"
        base = dict(caption=cap, src=r.clip, duration=10.0)
        jobs += [
            ("cover_n03", dict(base, task="cover", cover_strength=1.0, extra={"shift": 3.0, "cover_noise_strength": 0.3}), tg),
            ("cover_n05", dict(base, task="cover", cover_strength=1.0, extra={"shift": 3.0, "cover_noise_strength": 0.5}), tg),
            ("cover_n07", dict(base, task="cover", cover_strength=1.0, extra={"shift": 3.0, "cover_noise_strength": 0.7}), tg),
            ("cover_s05_n05", dict(base, task="cover", cover_strength=0.5, extra={"shift": 3.0, "cover_noise_strength": 0.5}), tg),
            ("flowedit", dict(base, task="text2music", extra={"shift": 3.0, "flow_edit_morph": True,
                                                              "flow_edit_source_caption": f"{GENRE_TEXT[g]} music, instrumental",
                                                              "flow_edit_source_lyrics": "[Instrumental]"}), tg),
            ("repaint_all_05", dict(base, task="repaint", repaint=(0.0, 10.0),
                                    extra={"shift": 3.0, "repaint_strength": 0.5}), tg),
            ("repaint_all_08", dict(base, task="repaint", repaint=(0.0, 10.0),
                                    extra={"shift": 3.0, "repaint_strength": 0.8}), tg),
        ]
    return jobs


def suite3_jobs(r, g):
    """cover-nofsq (unquantised source hints) and FlowEdit on top of cover (all with shift 3)."""
    jobs = []
    for k, tg in enumerate(GENRES):
        if k % 5 != GENRES.index(g) % 5:
            continue
        cap = f"{GENRE_TEXT[tg]} music, instrumental"
        base = dict(caption=cap, src=r.clip, duration=10.0)
        fe = {"flow_edit_morph": True, "flow_edit_source_caption": f"{GENRE_TEXT[g]} music, instrumental",
              "flow_edit_source_lyrics": "[Instrumental]"}
        jobs += [
            ("nofsq_s1", dict(base, task="cover-nofsq", cover_strength=1.0, extra={"shift": 3.0}), tg),
            ("nofsq_s05", dict(base, task="cover-nofsq", cover_strength=0.5, extra={"shift": 3.0}), tg),
            ("nofsq_s02", dict(base, task="cover-nofsq", cover_strength=0.2, extra={"shift": 3.0}), tg),
            ("cover_s1_shift3", dict(base, task="cover", cover_strength=1.0, extra={"shift": 3.0}), tg),
            ("cover_flowedit", dict(base, task="cover", cover_strength=1.0, extra=dict(fe, shift=3.0)), tg),
            ("nofsq_flowedit", dict(base, task="cover-nofsq", cover_strength=1.0, extra=dict(fe, shift=3.0)), tg),
        ]
    return jobs


def exemplars():
    """One 10-s TRAINING clip per genre that CLAP zero-shot finds most typical (style references)."""
    import soundfile as sf
    from .config import CACHE
    from .data import load_audio, load_index
    idx = load_index()
    idx = idx[idx.ok]
    ci = pd.read_csv(CACHE / "clap_crops_index.csv")
    zs = np.load(CACHE / "clap_zeroshot_crops.npy")
    split = ci.track.map(idx.set_index("track").split)
    out = {}
    for k, g in enumerate(GENRES):
        m = (split == "train").to_numpy() & ci.track.str.startswith(g + ".").to_numpy()
        j = np.flatnonzero(m)[zs[m, k].argmax()]
        t, c = ci.track.iloc[j], int(ci.crop.iloc[j])
        y = load_audio(idx.set_index("track").loc[t, "path"], 22050, offset=10.0 * c, duration=10.0)
        path = WORK / "exemplars" / f"{g}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), y, 22050, subtype="PCM_16")
        out[g] = {"path": str(path), "track": t, "crop": c, "p_zero_shot": float(zs[j, k])}
    (OUT / "exemplars.json").write_text(json.dumps(out, indent=1))
    return out


def suite4_jobs(r, g, ex):
    """Style reference audio (a GTZAN training exemplar of the target) and the planning LM."""
    jobs = []
    k0 = GENRES.index(g)
    for tg in (GENRES[(k0 + 3) % 10], GENRES[(k0 + 6) % 10]):     # two targets, never the source genre
        cap = f"{GENRE_TEXT[tg]} music, instrumental"
        ref = ex[tg]["path"]
        jobs += [
            ("t2m_lm", dict(caption=cap, task="text2music", duration=10.0, thinking=True), tg),
            ("t2m_ref", dict(caption=cap, task="text2music", duration=10.0, reference=ref), tg),
            ("cover_s1", dict(caption=cap, task="cover", src=r.clip, cover_strength=1.0), tg),
            ("cover_s05", dict(caption=cap, task="cover", src=r.clip, cover_strength=0.5), tg),
            ("cover_n03", dict(caption=cap, task="cover", src=r.clip, cover_strength=1.0,
                               extra={"cover_noise_strength": 0.3}), tg),
            ("ref_cover_s1", dict(caption=cap, task="cover", src=r.clip, cover_strength=1.0, reference=ref), tg),
            ("ref_cover_s05", dict(caption=cap, task="cover", src=r.clip, cover_strength=0.5, reference=ref), tg),
            ("ref_cover_n03", dict(caption=cap, task="cover", src=r.clip, cover_strength=1.0, reference=ref,
                                   extra={"cover_noise_strength": 0.3}), tg),
        ]
    return jobs


def main():
    import soundfile as sf
    import librosa
    from .ace import Ace
    from .judge import Judges
    from .melody import features_parallel, similarity
    OUT.mkdir(parents=True, exist_ok=True)
    W.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(RESULTS / "exp2" / "sources.csv")
    val = src[src.split == "valid"].groupby("genre").head(1).reset_index(drop=True)   # 1 per genre
    rows = []
    ace = Ace()
    t0 = time.time()
    import sys
    suite2 = "--suite2" in sys.argv
    suite3 = "--suite3" in sys.argv
    suite4 = "--suite4" in sys.argv
    ex = exemplars() if suite4 else None
    if suite4:
        del ace
        ace = Ace(use_lm=True)
    for r in val.itertuples():
        g = r.genre
        if suite4:
            jobs = suite4_jobs(r, g, ex)
        elif suite3:
            jobs = suite3_jobs(r, g)
        elif suite2:
            jobs = suite2_jobs(r, g)
        else:
          jobs = [
            ("recon", dict(caption=f"{GENRE_TEXT[g]} music, instrumental", task="repaint", src=r.clip,
                           repaint=(9.0, 10.0), duration=10.0), g),
            ("cover_same_s1", dict(caption=f"{GENRE_TEXT[g]} music, instrumental", task="cover", src=r.clip,
                                   cover_strength=1.0, duration=10.0), g),
        ]
        for k, tg in enumerate(GENRES):
            if k % 5 != GENRES.index(g) % 5:
                continue                     # 2 targets per source keeps this short
            jobs += [
                ("t2m_short_10s", dict(caption=f"{GENRE_TEXT[tg]} music, instrumental", task="text2music",
                                       duration=10.0), tg),
                ("t2m_rich_10s", dict(caption=RICH[tg], task="text2music", duration=10.0), tg),
                ("t2m_short_30s", dict(caption=f"{GENRE_TEXT[tg]} music, instrumental", task="text2music",
                                       duration=30.0), tg),
                ("t2m_short_10s_shift3", dict(caption=f"{GENRE_TEXT[tg]} music, instrumental", task="text2music",
                                              duration=10.0, extra={"shift": 3.0}), tg),
                ("t2m_rich_30s_shift3", dict(caption=RICH[tg], task="text2music", duration=30.0,
                                             extra={"shift": 3.0}), tg),
                ("cover_s06", dict(caption=f"{GENRE_TEXT[tg]} music, instrumental", task="cover", src=r.clip,
                                   cover_strength=0.6, duration=10.0), tg),
                ("cover_rich_s06_shift3", dict(caption=RICH[tg], task="cover", src=r.clip, cover_strength=0.6,
                                               duration=10.0, extra={"shift": 3.0}), tg),
                ("cover_rich_s03_shift3", dict(caption=RICH[tg], task="cover", src=r.clip, cover_strength=0.3,
                                               duration=10.0, extra={"shift": 3.0}), tg),
            ]
        for name, kw, tg in jobs:
            out = W / f"{name}__{r.track}__{tg}.wav"
            if out.exists():
                rows.append({"variant": name, "track": r.track, "src_genre": g, "target": tg,
                             "out": str(out), "src_clip": r.clip, "error": ""})
                continue
            try:
                ace.generate(out, seed=1234, **kw)
                if kw.get("duration") == 30.0:   # judge seconds 10-20 like the GTZAN clips
                    y, sr = sf.read(str(out))
                    sf.write(str(out), y[int(10 * sr):int(20 * sr)], sr, subtype="PCM_16")
                rows.append({"variant": name, "track": r.track, "src_genre": g, "target": tg,
                             "out": str(out), "src_clip": r.clip, "error": ""})
            except Exception as e:  # keep going, report the failure
                rows.append({"variant": name, "track": r.track, "src_genre": g, "target": tg,
                             "out": "", "src_clip": r.clip, "error": str(e)[:300]})
                log("FAILED", name, r.track, tg, e)
        log(f"{r.track}: {len(rows)} outputs, {time.time() - t0:.0f}s")
    del ace
    import torch
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    ok = df[df.out != ""].reset_index(drop=True)
    J = Judges(eval_mode=True)
    res = J.score_files(ok.out.tolist())
    tgt = ok.target.map(GENRES.index).to_numpy()
    ok["top_clap"] = [GENRES[i] for i in res["clap"].argmax(1)]
    ok["hit_clap"] = res["clap"].argmax(1) == tgt
    ok["p_clap_target"] = res["clap"][np.arange(len(ok)), tgt]
    ok["top_radar"] = [GENRES[i] for i in res["radar"].argmax(1)]
    ok["hit_radar"] = res["radar"].argmax(1) == tgt
    feats = features_parallel(ok.out.tolist() + ok.src_clip.unique().tolist())
    ok["melody"] = [similarity(feats[s], feats[o])["melody"] for s, o in zip(ok.src_clip, ok.out)]
    st = pd.DataFrame([stats(p) for p in ok.out])
    ok = pd.concat([ok, st], axis=1)

    # the untouched validation sources (all 30) as a reference
    allv = src[src.split == "valid"].reset_index(drop=True)
    r2 = J.score_files(allv["clip"].tolist())
    y = allv.genre.map(GENRES.index).to_numpy()
    src_ref = {"clap_acc_sources": float((r2["clap"].argmax(1) == y).mean()),
               "radar_acc_sources": float((r2["radar"].argmax(1) == y).mean()),
               "source_top_clap": pd.Series([GENRES[i] for i in r2["clap"].argmax(1)]).value_counts().to_dict()}
    sst = pd.DataFrame([stats(p) for p in allv["clip"]]).mean(numeric_only=True).round(4).to_dict()

    tag = "4" if suite4 else ("3" if suite3 else ("2" if suite2 else ""))
    ok.to_csv(OUT / f"diag_outputs{tag}.csv", index=False)
    summ = ok.groupby("variant")[["hit_clap", "hit_radar", "p_clap_target", "melody", "peak", "rms_db",
                                  "clip_frac", "flatness", "centroid"]].mean().round(3)
    tops = ok.groupby("variant").top_clap.value_counts().unstack(fill_value=0)
    with open(OUT / f"diag_summary{tag}.txt", "w") as f:
        f.write("judges on 30 validation source clips: " + json.dumps(src_ref) + "\n")
        f.write("source clip stats: " + json.dumps(sst) + "\n\n")
        f.write(summ.to_string() + "\n\n" + tops.to_string() + "\n\n")
        f.write("errors: " + json.dumps(df[df.error != ""].error.value_counts().to_dict()) + "\n")
    log(open(OUT / f"diag_summary{tag}.txt").read())


if __name__ == "__main__":
    main()
