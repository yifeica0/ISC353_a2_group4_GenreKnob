"""Experiment 2 - genre swap with ACE-Step 1.5 (generator frozen, no training).

Question: when a song is re-rendered in another genre,
  H2 (multimodal integration): does giving the model the source audio (cover task:
      text + audio) keep the melody better than text alone, and at what cost in genre?
  H3 (steering): does adding a CAA genre direction inside the DiT raise the genre
      hit rate of covers without losing the melody?
Primary judge: CLAP zero-shot (never trained on GTZAN); genre score = its probability
of the target genre (hit = target is its top label). Secondary judges: genre radar
(CLAP embedding + LogReg, GTZAN train+valid) and MERT layer-5 + LogReg (GTZAN
train+valid, scored in the MERT env). Melody kept: centred-chroma cosine similarity
to the source clip. Settings (keep-original strength, steering layers, steering
strength) are chosen on validation sources only.

Run v2 (after the validation diagnostics in results/diag): ACE-Step timestep shift 3
(the documented turbo setting) and covers that start from the noised source latents
(ACE `cover_noise_strength`, our "keep original" knob) - plain covers kept little
of the melody in the diagnostics.

CLI (see jobs/queue/04*.sh):
  prepare | plan STAGE | gen STAGE --shard I --nshards N | score STAGE
  choose-strength | choose-layers | choose-alpha | stats
"""
from __future__ import annotations

import argparse
import json
import time
import zlib

import numpy as np
import pandas as pd

from . import plots as P
from . import stats as S
from .config import CACHE, GENRE_TEXT, GENRES, RESULTS, SEED, WORK, ensure_dirs

OUT = RESULTS / "exp2"
W2 = WORK / "exp2"
S_GRID = [0.1, 0.2, 0.3, 0.4, 0.5]       # keep-original strength (ACE cover_noise_strength)
RUN = "v2"
A_GRID = [0.5, 1.0, 2.0, 4.0]             # steering strength (x typical activation norm)
CLIP_OFFSET, CLIP_DUR = 10.0, 10.0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def caption(genre: str) -> str:
    return f"{GENRE_TEXT[genre]} music, instrumental"


def seed_of(track: str, target: str) -> int:
    return zlib.crc32(f"{track}|{target}".encode()) % (2 ** 31 - 1)


def choices() -> dict:
    p = OUT / "choices.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_choices(**kw):
    c = choices()
    c.update(kw)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "choices.json").write_text(json.dumps(c, indent=2))


# ------------------------------------------------------------------------- prepare
def prepare():
    import soundfile as sf
    from .data import load_audio, load_index
    ensure_dirs()
    idx = load_index()
    idx = idx[idx.ok]
    rng = np.random.default_rng(SEED)
    rows = []
    for split, n in (("test", 10), ("valid", 3)):
        for g in GENRES:
            pool = idx[(idx.split == split) & (idx.genre == g)].track.to_list()
            for t in sorted(rng.choice(pool, size=min(n, len(pool)), replace=False)):
                path = idx.set_index("track").loc[t, "path"]
                y = load_audio(path, 22050, offset=CLIP_OFFSET, duration=CLIP_DUR)
                out = W2 / "sources" / f"{t}.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(out), y, 22050, subtype="PCM_16")
                rows.append({"track": t, "genre": g, "split": split, "clip": str(out)})
    src = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    src.to_csv(OUT / "sources.csv", index=False)
    log("sources", src.groupby("split").size().to_dict())


def sources():
    return pd.read_csv(OUT / "sources.csv")


def valid_pairs(n_targets=3):
    src = sources()
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for r in src[src.split == "valid"].itertuples():
        others = [g for g in GENRES if g != r.genre]
        for t in rng.choice(others, size=n_targets, replace=False):
            rows.append((r.track, r.genre, r.clip, t))
    return rows


def test_pairs():
    src = sources()
    return [(r.track, r.genre, r.clip, t) for r in src[src.split == "test"].itertuples()
            for t in GENRES if t != r.genre]


def knob_pairs():
    """3 test sources per genre x 3 targets, for the cover-strength curve."""
    src = sources()
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for g in GENRES:
        tr = src[(src.split == "test") & (src.genre == g)].head(3)
        for r in tr.itertuples():
            for t in rng.choice([x for x in GENRES if x != g], size=3, replace=False):
                rows.append((r.track, r.genre, r.clip, t))
    return rows


# ------------------------------------------------------------------------- plans
def _row(stage, cond, pair, task, strength=None, layers=None, alpha=0.0):
    track, sg, clip, tgt = pair
    rid = f"{cond}__{track}__{tgt}"
    return {"id": rid, "stage": stage, "cond": cond, "track": track, "src_genre": sg,
            "src_clip": clip, "target": tgt, "seed": seed_of(track, tgt), "task": task,
            "strength": strength if strength is not None else np.nan,
            "layers": ",".join(map(str, layers)) if layers else "", "alpha": alpha,
            "out": str(W2 / f"out_{RUN}" / stage / cond / f"{track}__{tgt}.wav")}


def make_plan(stage: str):
    ch = choices()
    rows = []
    if stage == "strength":
        for p in valid_pairs():
            rows.append(_row(stage, "T", p, "text2music"))
            for s in S_GRID:
                rows.append(_row(stage, f"C_s{s}", p, "cover", strength=s))
    elif stage == "layers":
        import torch
        layers = torch.load(CACHE / "genre_vectors.pt", map_location="cpu")["layers"]
        for p in valid_pairs()[::3]:              # one target per valid source (30 pairs)
            rows.append(_row(stage, "C", p, "cover", strength=ch["strength"]))
            for l in layers:
                rows.append(_row(stage, f"CS_L{l}", p, "cover", strength=ch["strength"],
                                 layers=[l], alpha=1.0))
    elif stage == "alpha":
        for p in valid_pairs():
            rows.append(_row(stage, "C", p, "cover", strength=ch["strength"]))
            for a in A_GRID:
                rows.append(_row(stage, f"CS_a{a}", p, "cover", strength=ch["strength"],
                                 layers=ch["layers"], alpha=a))
    elif stage == "test":
        for p in test_pairs():
            rows.append(_row(stage, "T", p, "text2music"))
            rows.append(_row(stage, "C", p, "cover", strength=ch["strength"]))
            rows.append(_row(stage, "CS", p, "cover", strength=ch["strength"],
                             layers=ch["layers"], alpha=ch["alpha"]))
        for p in knob_pairs():
            for s in S_GRID:
                rows.append(_row(stage, f"knob_s{s}", p, "cover", strength=s))
    else:
        raise ValueError(stage)
    plan = pd.DataFrame(rows).drop_duplicates("id")
    (W2 / f"plans_{RUN}").mkdir(parents=True, exist_ok=True)
    plan.to_csv(W2 / f"plans_{RUN}" / f"{stage}.csv", index=False)
    log(f"plan {stage}: {len(plan)} generations", plan.cond.value_counts().to_dict())


def load_plan(stage):
    return pd.read_csv(W2 / f"plans_{RUN}" / f"{stage}.csv", keep_default_na=False,
                       na_values={"strength": [""]})


# ------------------------------------------------------------------------- generation
def generate(stage: str, shard: int, nshards: int):
    import os
    from .ace import Ace, GenreVectors
    plan = load_plan(stage)
    mine = plan[[zlib.crc32(i.encode()) % nshards == shard for i in plan.id]]
    todo = mine[[not os.path.exists(o) for o in mine.out]]
    log(f"[gen {stage} shard {shard}/{nshards}] {len(todo)} to do of {len(mine)}")
    if todo.empty:
        return
    ace = Ace()
    vec = GenreVectors(CACHE / "genre_vectors.pt") if (todo.alpha > 0).any() else None
    t0 = time.time()
    for k, r in enumerate(todo.itertuples()):
        deltas = None
        if r.alpha and float(r.alpha) > 0:
            layers = [int(x) for x in str(r.layers).split(",") if x != ""]
            deltas = vec.deltas({r.target: 1.0}, layers, float(r.alpha))
        try:
            extra = {"cover_noise_strength": float(r.strength)} if r.task == "cover" else None
            ace.generate(r.out, caption(r.target), task=r.task,
                         src=r.src_clip if r.task == "cover" else None,
                         duration=CLIP_DUR, seed=int(r.seed), cover_strength=1.0,
                         deltas=deltas, extra=extra)
        except Exception as e:  # keep going; failures are listed in the log
            log(f"FAILED {r.id}: {e}")
        if k % 25 == 0:
            el = time.time() - t0
            log(f"[gen {stage} shard {shard}] {k + 1}/{len(todo)}  {el / (k + 1):.2f}s/clip")


# ------------------------------------------------------------------------- scoring
def score(stage: str):
    import os
    from .judge import Judges
    from .melody import features_parallel, similarity
    plan = load_plan(stage)
    plan = plan[[os.path.exists(o) for o in plan.out]].reset_index(drop=True)
    log(f"score {stage}: {len(plan)} outputs")
    J = Judges(eval_mode=True)
    res = J.score_files(plan.out.tolist())
    tgt = plan.target.map(GENRES.index).to_numpy()
    srcg = plan.src_genre.map(GENRES.index).to_numpy()
    n = np.arange(len(plan))
    df = plan[["id", "stage", "cond", "track", "src_genre", "target", "strength", "layers", "alpha", "out"]].copy()
    df["p_clap_target"] = res["clap"][n, tgt]
    df["p_clap_source"] = res["clap"][n, srcg]
    df["top_clap"] = [GENRES[i] for i in res["clap"].argmax(1)]
    df["hit_clap"] = res["clap"].argmax(1) == tgt
    if "radar" in res:
        df["p_radar_target"] = res["radar"][n, tgt]
        df["hit_radar"] = res["radar"].argmax(1) == tgt
    # melody kept (and a chance level: similarity to an unrelated source clip)
    feats = features_parallel(plan.out.tolist() + plan.src_clip.tolist())
    clips = plan.src_clip.to_numpy()
    tracks = plan.track.to_numpy()
    rows = []
    for i, r in enumerate(plan.itertuples()):
        fo = feats[r.out]
        own = similarity(feats[r.src_clip], fo)
        j = (i + 7) % len(plan)
        while tracks[j] == r.track:
            j = (j + 1) % len(plan)
        ch = similarity(feats[clips[j]], fo)
        rows.append({"melody": own["melody"], "chroma_cens": own["chroma_cens"],
                     "melody_chance": ch["melody"], "chroma_cens_chance": ch["chroma_cens"]})
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"scores_{stage}.csv", index=False)
    log(df.groupby("cond")[["hit_clap", "p_clap_target", "melody", "melody_chance"]].mean().round(3).to_string())

    if stage == "test":  # how the judges see the untouched source clips
        src = sources()
        src = src[src.split == "test"].reset_index(drop=True)
        r2 = J.score_files(src["clip"].tolist())
        y = src.genre.map(GENRES.index).to_numpy()
        out = src[["track", "genre"]].copy()
        out["top_clap"] = [GENRES[i] for i in r2["clap"].argmax(1)]
        out["hit_clap"] = r2["clap"].argmax(1) == y
        if "radar" in r2:
            out["hit_radar"] = r2["radar"].argmax(1) == y
        out.to_csv(OUT / "scores_sources.csv", index=False)
        log("judges on source clips:", out[[c for c in out if c.startswith("hit")]].mean().round(3).to_dict())


def _melody_norm(df):
    c = float(df.melody_chance.mean())
    return ((df.melody - c) / (1 - c)).clip(0, 1), c


def choose_strength():
    df = pd.read_csv(OUT / "scores_strength.csv")
    mn, c = _melody_norm(df)
    df["melody_norm"] = mn
    rows = []
    for cond, g in df.groupby("cond"):
        pt, mel = float(g.p_clap_target.mean()), float(g.melody_norm.mean())
        rows.append({"cond": cond, "p_target": pt, "hit": float(g.hit_clap.mean()),
                     "melody": float(g.melody.mean()), "melody_norm": mel, "balance": S.balance_score(pt, mel)})
    curve = pd.DataFrame(rows)
    curve.to_csv(OUT / "valid_strength_curve.csv", index=False)
    best = curve[curve.cond.str.startswith("C_s")].sort_values("balance", ascending=False).iloc[0]
    s = float(best.cond.split("C_s")[1])
    save_choices(strength=s, melody_chance_valid=c)
    log("strength curve\n", curve.round(3).to_string(), "\nchosen strength", s)


def choose_layers(top_k: int = 2):
    df = pd.read_csv(OUT / "scores_layers.csv")
    base = df[df.cond == "C"].set_index(df[df.cond == "C"].id.str.replace("C__", "", regex=False))
    rows = []
    for cond, g in df[df.cond.str.startswith("CS_L")].groupby("cond"):
        key = g.id.str.replace(f"{cond}__", "", regex=False)
        b = base.loc[key]
        rows.append({"layer": int(cond.split("CS_L")[1]),
                     "delta_p_target": float((g.p_clap_target.to_numpy() - b.p_clap_target.to_numpy()).mean()),
                     "delta_hit": float((g.hit_clap.to_numpy().astype(float) - b.hit_clap.to_numpy().astype(float)).mean()),
                     "delta_melody": float((g.melody.to_numpy() - b.melody.to_numpy()).mean())})
    sweep = pd.DataFrame(rows).sort_values("layer")
    sweep.to_csv(OUT / "valid_layer_sweep.csv", index=False)
    best = sweep.sort_values("delta_p_target", ascending=False).head(top_k).layer.astype(int).tolist()
    save_choices(layers=sorted(best))
    log("layer sweep\n", sweep.round(3).to_string(), "\nchosen layers", sorted(best))
    fig_layers(sweep, sorted(best))


def choose_alpha():
    df = pd.read_csv(OUT / "scores_alpha.csv")
    mn, c = _melody_norm(df)
    df["melody_norm"] = mn
    rows = []
    for cond, g in df.groupby("cond"):
        a = 0.0 if cond == "C" else float(cond.split("CS_a")[1])
        pt, mel = float(g.p_clap_target.mean()), float(g.melody_norm.mean())
        rows.append({"alpha": a, "p_target": pt, "hit": float(g.hit_clap.mean()),
                     "melody": float(g.melody.mean()), "melody_norm": mel, "balance": S.balance_score(pt, mel)})
    curve = pd.DataFrame(rows).sort_values("alpha")
    curve.to_csv(OUT / "valid_alpha_curve.csv", index=False)
    best = curve[curve.alpha > 0].sort_values("balance", ascending=False).iloc[0]
    save_choices(alpha=float(best.alpha))
    log("alpha curve\n", curve.round(3).to_string(), "\nchosen alpha", float(best.alpha))


# ------------------------------------------------------------------------- statistics
GENRE_METRICS = ["p_clap_target", "hit_clap", "hit_radar", "hit_mert", "p_clap_source"]


def load_test_scores():
    df = pd.read_csv(OUT / "scores_test.csv")
    mert = OUT / "scores_test_mert.csv"
    if mert.exists():
        m = pd.read_csv(mert)[["id", "top_mert", "p_mert_target", "hit_mert"]]
        df = df.merge(m, on="id", how="left")
    for c in ("hit_clap", "hit_radar", "hit_mert"):
        if c in df:
            df[c] = df[c].astype(float)
    return df


def stats():
    P.setup()
    df = load_test_scores()
    ch = choices()
    main = df[df.cond.isin(["T", "C", "CS"])].copy()
    main["pair"] = main.track + "|" + main.target
    metrics = [m for m in GENRE_METRICS + ["melody"] if m in main]
    piv = {m: main.pivot_table(index="pair", columns="cond", values=m, aggfunc="first") for m in metrics}
    pairs = piv["melody"].dropna().index
    src_of = pd.Series([p.split("|")[0] for p in pairs], index=pairs)

    def per_source(metric, cond):
        return piv[metric].loc[pairs, cond].astype(float).groupby(src_of).mean()

    summ = []
    for cond in ("T", "C", "CS"):
        for metric in metrics:
            v = per_source(metric, cond)
            lo, hi = S.bootstrap_ci(v.to_numpy())
            summ.append({"cond": cond, "metric": metric, "mean": float(v.mean()), "ci_lo": lo, "ci_hi": hi})
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "test_condition_summary.csv", index=False)

    # primary family (Holm over 4): H2 = cover vs text-only, H3 = cover+steer vs cover
    tests = []
    for name, a, b, metric in (("H2: cover vs text only", "C", "T", "melody"),
                               ("H2: cover vs text only", "C", "T", "p_clap_target"),
                               ("H3: cover + steering vs cover", "CS", "C", "p_clap_target"),
                               ("H3: cover + steering vs cover", "CS", "C", "melody")):
        va, vb = per_source(metric, a), per_source(metric, b)
        w = S.wilcoxon_paired(va.to_numpy(), vb.to_numpy())
        lo, hi = S.paired_bootstrap_diff(va.to_numpy(), vb.to_numpy())
        tests.append({"test": name, "metric": metric, "A": a, "B": b, "mean_A": float(va.mean()),
                      "mean_B": float(vb.mean()), **w, "diff_ci_lo": lo, "diff_ci_hi": hi})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = S.holm(tests.p.to_numpy())
    tests.to_csv(OUT / "test_wilcoxon.csv", index=False)

    # secondary: pair-level hits for every judge (exact McNemar, not corrected)
    mcn = []
    for a, b in (("C", "T"), ("CS", "C")):
        for metric in [m for m in ("hit_clap", "hit_radar", "hit_mert") if m in piv]:
            mcn.append({"A": a, "B": b, "metric": metric,
                        "rate_A": float(piv[metric].loc[pairs, a].mean()), "rate_B": float(piv[metric].loc[pairs, b].mean()),
                        **S.mcnemar_exact(piv[metric].loc[pairs, a].astype(bool).to_numpy(),
                                          piv[metric].loc[pairs, b].astype(bool).to_numpy())})
    mcn = pd.DataFrame(mcn)
    mcn.to_csv(OUT / "test_mcnemar_pairs.csv", index=False)

    mats = {}
    for cond in ("T", "C", "CS"):
        sub = main[main.cond == cond]
        m = sub.pivot_table(index="src_genre", columns="target", values="p_clap_target", aggfunc="mean")
        mats[cond] = m.reindex(index=GENRES, columns=GENRES)
        mats[cond].to_csv(OUT / f"swap_matrix_{cond}.csv")
    top = main.groupby("cond").top_clap.value_counts(normalize=True).unstack(fill_value=0)
    top.to_csv(OUT / "test_top_label_share.csv")
    if "top_mert" in main:
        main.groupby("cond").top_mert.value_counts(normalize=True).unstack(fill_value=0).to_csv(
            OUT / "test_top_label_share_mert.csv")

    knob = df[df.cond.str.startswith("knob_s")].copy()
    knob["strength"] = knob.cond.str.replace("knob_s", "", regex=False).astype(float)
    agg = dict(p_target=("p_clap_target", "mean"), hit=("hit_clap", "mean"), melody=("melody", "mean"),
               melody_chance=("melody_chance", "mean"))
    if "hit_mert" in knob:
        agg["hit_mert"] = ("hit_mert", "mean")
    kc = knob.groupby("strength").agg(**agg).reset_index()
    kc.to_csv(OUT / "test_knob_curve.csv", index=False)
    # T and CS on the same 90 pairs as the knob curve (for a like-for-like reference)
    kp = set(zip(knob.track, knob.target))
    ref = main[[(t, g) in kp for t, g in zip(main.track, main.target)]]
    kref = ref.groupby("cond").agg(**{k: v for k, v in agg.items() if k != "melody_chance"}).reset_index()
    kref["n_pairs"] = ref.groupby("cond").size().reindex(kref.cond).to_numpy()
    kref.to_csv(OUT / "test_knob_reference.csv", index=False)

    chance = float(main.melody_chance.mean())   # unrelated-clip level over the T/C/CS clips
    fig_conditions(summ, chance)
    fig_knob(kc, kref, chance)
    fig_matrix(mats)
    summary = {"choices": ch, "n_pairs": int(len(pairs)), "n_sources": int(src_of.nunique()),
               "conditions": summ.to_dict("records"), "wilcoxon": tests.to_dict("records"),
               "mcnemar_pairs": mcn.to_dict("records"), "melody_chance": chance,
               "knob_curve": kc.to_dict("records"), "knob_reference": kref.to_dict("records"),
               "top_label_share": top.round(4).to_dict("index")}
    src_scores = OUT / "scores_sources.csv"
    if src_scores.exists():
        sc = pd.read_csv(src_scores)
        summary["judges_on_sources"] = {c: float(sc[c].mean()) for c in sc if c.startswith("hit")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log("conditions\n", summ.round(3).to_string())
    log("wilcoxon\n", tests.round(4).to_string())
    log("mcnemar\n", mcn.round(4).to_string())


# ------------------------------------------------------------------------- figures
COND_LABEL = {"T": "Text only", "C": "Cover\n(text + audio)", "CS": "Cover +\nsteering"}


def fig_conditions(summ, chance):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for ax, metric, title in ((axes[0], "p_clap_target", "P(target genre), CLAP judge"),
                              (axes[1], "melody", "Melody kept (chroma similarity)")):
        s = summ[summ.metric == metric].set_index("cond").loc[["T", "C", "CS"]]
        x = np.arange(3)
        ax.errorbar(x, s["mean"], yerr=[s["mean"] - s.ci_lo, s.ci_hi - s["mean"]], fmt="o",
                    color=P.BLUE, ms=7, elinewidth=2, capsize=0)
        for xi, v in zip(x, s["mean"]):
            ax.text(xi + 0.12, v, f"{v:.2f}", va="center", fontsize=8)
        ax.set_xticks(x, [COND_LABEL[c] for c in s.index])
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.5, 2.6)
        ax.grid(axis="x", visible=False)
        if metric == "melody":
            ax.axhline(chance, color=P.GRAY, ls="--", lw=1)
            ax.text(2.55, chance, "unrelated\nclip", fontsize=7, color=P.TEXT2, ha="right", va="bottom")
        else:
            ax.axhline(0.1, color=P.GRAY, ls=":", lw=1)
            ax.text(2.55, 0.1, "chance", fontsize=7, color=P.TEXT2, ha="right", va="bottom")
    P.save(fig, OUT / "fig_exp2_conditions.png")


def fig_knob(kc, kref, chance):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(kc.melody, kc.p_target, color=P.BLUE, marker="o", ms=6, label="cover, keep-original 0.1 to 0.5")
    for r in kc.itertuples():
        ax.annotate(f"{r.strength:g}", (r.melody, r.p_target), xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5)
    s = kref.set_index("cond")
    ax.scatter([s.loc["T", "melody"]], [s.loc["T", "p_target"]], color=P.ORANGE, s=45, zorder=3, label="text only")
    ax.scatter([s.loc["CS", "melody"]], [s.loc["CS", "p_target"]], color=P.AQUA, s=45, zorder=3,
               label="cover + steering")
    ax.axvline(chance, color=P.GRAY, ls="--", lw=1)
    ax.set_xlabel("melody kept (chroma similarity)")
    ax.set_ylabel("P(target genre), CLAP judge")
    ax.legend(fontsize=7.5, loc="best")
    P.save(fig, OUT / "fig_exp2_knob.png")


def fig_matrix(mats):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw={"wspace": 0.35})
    names = {"T": "Text only", "C": "Cover", "CS": "Cover + steering"}
    for ax, (cond, m) in zip(axes, mats.items()):
        a = m.to_numpy(dtype=float)
        ax.imshow(np.nan_to_num(a), cmap=P.SEQ, vmin=0, vmax=1)
        ax.set_xticks(range(10), GENRES, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(10), GENRES, fontsize=7)
        ax.grid(False)
        for i in range(10):
            for j in range(10):
                if not np.isnan(a[i, j]):
                    ax.text(j, i, f"{a[i, j]:.1f}", ha="center", va="center", fontsize=5.5,
                            color=P.ink_on(P.SEQ, a[i, j]))
        ax.set_title(names[cond] + ": P(target)", fontsize=9)
        ax.set_xlabel("target genre"); ax.set_ylabel("source genre")
    P.save(fig, OUT / "fig_exp2_matrix.png")


def fig_layers(sweep, chosen):
    import matplotlib.pyplot as plt
    P.setup()
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    ax.bar(sweep.layer, sweep.delta_p_target, color=[P.BLUE if l in chosen else "#9ec5f4" for l in sweep.layer],
           width=0.75)
    ax.axhline(0, color=P.TEXT2, lw=0.8)
    ax.set_xlabel("DiT layer steered (alone, strength 1)")
    ax.set_ylabel("gain in P(target genre)")
    ax.grid(axis="x", visible=False)
    P.save(fig, OUT / "fig_exp2_layers.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("stage", nargs="?")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    ensure_dirs()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "plan":
        make_plan(a.stage)
    elif a.cmd == "gen":
        generate(a.stage, a.shard, a.nshards)
    elif a.cmd == "score":
        score(a.stage)
    elif a.cmd == "choose-strength":
        choose_strength()
    elif a.cmd == "choose-layers":
        choose_layers()
    elif a.cmd == "choose-alpha":
        choose_alpha()
    elif a.cmd == "stats":
        stats()
    else:
        raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
