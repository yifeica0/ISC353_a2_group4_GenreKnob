"""Experiment 1 - leakage-free genre classification.

Design: representation {handcrafted (HC), MERT-330M, CLAP} x classifier
{logistic regression (LR), XGBoost (XGB)}, plus CLAP zero-shot (no training).
Split: artist-disjoint fault-filtered GTZAN split (train / valid / test).
Hyper-parameters (and the MERT layer) are chosen on valid; final models are
refit on train+valid and evaluated once on the 290 test tracks.
Track prediction = mean of unit probabilities (HC: 10 x 3-s segments,
MERT/CLAP: 3 x 10-s crops). Models are compared with exact McNemar tests
(Holm-corrected) and Cochran's Q.

Run: python -m genreknob.exp1
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import plots as P
from . import stats as S
from .config import CACHE, GENRES, RESULTS, SEED, ensure_dirs
from .data import load_index
from .features import FEATURE_NAMES, GROUPS

warnings.filterwarnings("ignore", category=UserWarning)
OUT = RESULTS / "exp1"
LR_GRID = [{"C": c} for c in (0.001, 0.01, 0.1, 1.0, 10.0)]
XGB_GRID = [{"max_depth": d, "n_estimators": n} for d in (3, 6) for n in (300, 1000)]
NOTEBOOK_XGB = {"n_estimators": 1000, "max_depth": 6}  # first report: XGBClassifier(1000, lr=0.05)

LABELS = {
    "hc_xgb": "Handcrafted + XGBoost", "hc_lr": "Handcrafted + LogReg",
    "mert_xgb": "MERT + XGBoost", "mert_lr": "MERT + LogReg",
    "clap_xgb": "CLAP + XGBoost", "clap_lr": "CLAP + LogReg",
    "clap_zs": "CLAP zero-shot",
}
COMPARISONS = [  # (family label, model A, model B)
    ("H1: MERT+LR vs HC+XGB (main)", "mert_lr", "hc_xgb"),
    ("representation | LR: MERT vs HC", "mert_lr", "hc_lr"),
    ("representation | XGB: MERT vs HC", "mert_xgb", "hc_xgb"),
    ("representation | LR: CLAP vs HC", "clap_lr", "hc_lr"),
    ("representation | XGB: CLAP vs HC", "clap_xgb", "hc_xgb"),
    ("representation | LR: MERT vs CLAP", "mert_lr", "clap_lr"),
    ("classifier | HC: LR vs XGB", "hc_lr", "hc_xgb"),
    ("classifier | MERT: LR vs XGB", "mert_lr", "mert_xgb"),
    ("classifier | CLAP: LR vs XGB", "clap_lr", "clap_xgb"),
    ("supervision | CLAP: LR vs zero-shot", "clap_lr", "clap_zs"),
]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_clf(kind: str, **kw):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=kw["C"], max_iter=5000))
    from xgboost import XGBClassifier
    return XGBClassifier(learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
                         tree_method="hist", n_jobs=-1, random_state=SEED,
                         eval_metric="mlogloss", **kw)


class Units:
    """Unit-level matrix (segments or crops) with track ids and split labels."""

    def __init__(self, X, tracks, idx):
        meta = idx.set_index("track")
        keep = np.isin(tracks, meta.index)
        self.X = np.asarray(X)[keep]
        self.tracks = np.asarray(tracks)[keep]
        self.y = meta.loc[self.tracks, "label"].to_numpy()
        self.split = meta.loc[self.tracks, "split"].to_numpy()
        self.idx = idx

    def order(self, split):
        return self.idx.loc[self.idx.split == split, "track"].to_numpy()

    def track_labels(self, split):
        return self.idx.set_index("track").loc[self.order(split), "label"].to_numpy()

    def fit_predict(self, kind, params, fit_on, pred_on, cols=None):
        X = self.X if cols is None else self.X[:, cols]
        tr = np.isin(self.split, fit_on)
        te = self.split == pred_on
        m = make_clf(kind, **params)
        m.fit(X[tr], self.y[tr])
        p = m.predict_proba(X[te])
        df = pd.DataFrame(p, columns=range(p.shape[1]))
        df["track"] = self.tracks[te]
        tp = df.groupby("track").mean().reindex(self.order(pred_on)).to_numpy()
        return tp, m, p

    def tune(self, kind, grid):
        yv = self.track_labels("valid")
        rows, best = [], None
        for params in grid:
            pv, _, _ = self.fit_predict(kind, params, ["train"], "valid")
            acc = float((pv.argmax(1) == yv).mean())
            rows.append({**params, "valid_acc": acc})
            if best is None or acc > best[1]:
                best = (params, acc)
        return best[0], best[1], rows


def load_hc(idx):
    d = pd.read_csv(CACHE / "hc_segments.csv")
    return Units(d[FEATURE_NAMES].to_numpy(np.float64), d["track"].to_numpy(), idx)


def load_emb(name, idx, layer=None):
    arr = np.load(CACHE / f"{name}_crops.npy", mmap_mode="r")
    tracks = pd.read_csv(CACHE / f"{name}_crops_index.csv")["track"].to_numpy()
    X = np.asarray(arr[:, layer, :] if layer is not None else arr, dtype=np.float32)
    return Units(X, tracks, idx)


def evaluate(name, tp, y, extra=None):
    pred = tp.argmax(1)
    correct = pred == y
    lo, hi = S.bootstrap_ci(correct.astype(float))
    return {"model": name, "label": LABELS[name], "test_acc": float(correct.mean()),
            "ci_lo": lo, "ci_hi": hi,
            "macro_f1": float(f1_score(y, pred, average="macro")),
            **(extra or {})}, correct, pred


def leakage_audit(hc: Units):
    """Same features, three protocols: why the first report's 90% was optimistic."""
    rows = []
    X, y, g = hc.X, hc.y, hc.tracks
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(X, y, g, test_size=0.3, random_state=42)
    m = make_clf("xgb", **NOTEBOOK_XGB).fit(Xtr, ytr)
    rows.append({"protocol": "A random 70/30 split of 3-s segments (first report)",
                 "segment_acc": float((m.predict(Xte) == yte).mean()),
                 "test_tracks_also_in_train": float(np.isin(gte, gtr).mean())})
    accs = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g):
        m = make_clf("xgb", **NOTEBOOK_XGB).fit(X[tr], y[tr])
        accs.append(float((m.predict(X[te]) == y[te]).mean()))
    rows.append({"protocol": "B 5-fold CV grouped by track", "segment_acc": float(np.mean(accs)),
                 "segment_acc_sd": float(np.std(accs)), "test_tracks_also_in_train": 0.0})
    tp, m, p = hc.fit_predict("xgb", NOTEBOOK_XGB, ["train", "valid"], "test")
    te = hc.split == "test"
    rows.append({"protocol": "D artist-disjoint fault-filtered split",
                 "segment_acc": float((p.argmax(1) == hc.y[te]).mean()),
                 "track_acc": float((tp.argmax(1) == hc.track_labels("test")).mean()),
                 "test_tracks_also_in_train": 0.0})
    return pd.DataFrame(rows)


def tree_shap(model, X, y):
    """Exact TreeSHAP values from XGBoost (pred_contribs), shape (n, d, k)."""
    import xgboost as xgb
    c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    c = np.asarray(c)
    if c.ndim == 3:            # (n, k, d+1)
        sv = np.transpose(c[:, :, :-1], (0, 2, 1))
    else:                      # (n, (d+1)*k) flattened
        k = len(GENRES)
        sv = np.transpose(c.reshape(len(X), k, -1)[:, :, :-1], (0, 2, 1))
    return sv


def main():
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    P.setup()
    idx = load_index()
    idx = idx[idx.ok].reset_index(drop=True)
    y_test = idx.loc[idx.split == "test", "label"].to_numpy()
    log("tracks per split:", idx.split.value_counts().to_dict())

    results, correct, preds, probs, tuning = [], {}, {}, {}, []

    # ---- handcrafted -----------------------------------------------------------
    hc = load_hc(idx)
    log("handcrafted units:", hc.X.shape)
    audit = leakage_audit(hc)
    audit.to_csv(OUT / "leakage_audit.csv", index=False)
    log("leakage audit\n", audit.to_string())
    for kind, grid in (("lr", LR_GRID), ("xgb", XGB_GRID)):
        params, vacc, rows = hc.tune(kind, grid)
        tuning += [{"model": f"hc_{kind}", **r} for r in rows]
        tp, model, _ = hc.fit_predict(kind, params, ["train", "valid"], "test")
        r, c, pr = evaluate(f"hc_{kind}", tp, y_test, {"valid_acc": vacc, "params": json.dumps(params)})
        results.append(r); correct[f"hc_{kind}"] = c; preds[f"hc_{kind}"] = pr; probs[f"hc_{kind}"] = tp
        log(r)
        if kind == "xgb":
            hc_xgb_model, hc_xgb_params = model, params

    # ---- MERT: choose the layer on valid ------------------------------------------
    M = np.load(CACHE / "mert_crops.npy", mmap_mode="r")
    n_layers = M.shape[1]
    layer_rows = []
    for layer in range(n_layers):
        u = load_emb("mert", idx, layer)
        params, vacc, _ = u.tune("lr", LR_GRID)
        tp, _, _ = u.fit_predict("lr", params, ["train", "valid"], "test")
        layer_rows.append({"layer": layer, "valid_acc": vacc,
                           "test_acc": float((tp.argmax(1) == y_test).mean()), "C": params["C"]})
        log("MERT layer", layer_rows[-1])
    layers = pd.DataFrame(layer_rows)
    layers.to_csv(OUT / "mert_layer_probe.csv", index=False)
    best_layer = int(layers.sort_values(["valid_acc", "layer"], ascending=[False, True]).iloc[0]["layer"])
    log("best MERT layer (valid):", best_layer)

    for rep, units in (("mert", load_emb("mert", idx, best_layer)), ("clap", load_emb("clap", idx))):
        for kind, grid in (("lr", LR_GRID), ("xgb", XGB_GRID)):
            params, vacc, rows = units.tune(kind, grid)
            tuning += [{"model": f"{rep}_{kind}", **r} for r in rows]
            tp, _, _ = units.fit_predict(kind, params, ["train", "valid"], "test")
            extra = {"valid_acc": vacc, "params": json.dumps(params)}
            if rep == "mert":
                extra["mert_layer"] = best_layer
            r, c, pr = evaluate(f"{rep}_{kind}", tp, y_test, extra)
            results.append(r); correct[f"{rep}_{kind}"] = c; preds[f"{rep}_{kind}"] = pr; probs[f"{rep}_{kind}"] = tp
            log(r)

    # ---- CLAP zero-shot (no training) -------------------------------------------
    zs = np.load(CACHE / "clap_zeroshot_crops.npy")
    zs_tracks = pd.read_csv(CACHE / "clap_crops_index.csv")["track"].to_numpy()
    zdf = pd.DataFrame(zs); zdf["track"] = zs_tracks
    order = idx.loc[idx.split == "test", "track"].to_numpy()
    zt = zdf.groupby("track").mean().reindex(order).to_numpy()
    r, c, pr = evaluate("clap_zs", zt, y_test, {"valid_acc": np.nan, "params": "{}"})
    results.append(r); correct["clap_zs"] = c; preds["clap_zs"] = pr; probs["clap_zs"] = zt
    log(r)

    res = pd.DataFrame(results)
    res.to_csv(OUT / "model_results.csv", index=False)
    pd.DataFrame(tuning).to_csv(OUT / "tuning_valid.csv", index=False)
    pred_df = pd.DataFrame({"track": order, "label": y_test, **{f"pred_{k}": v for k, v in preds.items()}})
    pred_df.to_csv(OUT / "test_predictions.csv", index=False)

    # ---- significance tests -----------------------------------------------------
    rows = []
    for fam, a, b in COMPARISONS:
        t = S.mcnemar_exact(correct[a], correct[b])
        lo, hi = S.paired_bootstrap_diff(correct[a].astype(float), correct[b].astype(float))
        rows.append({"comparison": fam, "A": LABELS[a], "B": LABELS[b], **t, "diff_ci_lo": lo, "diff_ci_hi": hi})
    mc = pd.DataFrame(rows)
    mc["p_holm"] = S.holm(mc["p"].to_numpy())
    mc.to_csv(OUT / "mcnemar.csv", index=False)
    trained = ["hc_lr", "hc_xgb", "mert_lr", "mert_xgb", "clap_lr", "clap_xgb"]
    q = S.cochran_q(np.column_stack([correct[m] for m in trained]))
    log("McNemar\n", mc.to_string())
    log("Cochran Q", q)

    # ---- SHAP + ablation on the handcrafted XGBoost (the first report's model) ---
    te = hc.split == "test"
    Xte, yte = hc.X[te], hc.y[te]
    sv = tree_shap(hc_xgb_model, Xte, yte)                      # (n, 57, 10)
    true_sv = sv[np.arange(len(yte)), :, yte]                    # SHAP toward the true genre
    feat_imp = pd.DataFrame({"feature": FEATURE_NAMES, "mean_abs_shap": np.abs(true_sv).mean(0)})
    feat_imp.sort_values("mean_abs_shap", ascending=False).to_csv(OUT / "shap_features.csv", index=False)
    name_to_col = {f: i for i, f in enumerate(FEATURE_NAMES)}
    grp_rows, heat = [], np.zeros((len(GENRES), len(GROUPS)))
    for j, (grp, feats) in enumerate(GROUPS.items()):
        cols = [name_to_col[f] for f in feats]
        g_sv = true_sv[:, cols].sum(1)                            # additive group attribution
        grp_rows.append({"group": grp, "n_features": len(cols), "mean_abs_shap": float(np.abs(g_sv).mean())})
        for k in range(len(GENRES)):
            heat[k, j] = float(np.abs(g_sv[yte == k]).mean())
    grp = pd.DataFrame(grp_rows)
    grp.to_csv(OUT / "shap_groups.csv", index=False)
    pd.DataFrame(heat, index=GENRES, columns=list(GROUPS)).to_csv(OUT / "shap_group_by_genre.csv")

    abl = []
    for grp_name, feats in GROUPS.items():
        keep = [i for i, f in enumerate(FEATURE_NAMES) if f not in feats]
        tp, _, _ = hc.fit_predict("xgb", hc_xgb_params, ["train", "valid"], "test", cols=keep)
        c = tp.argmax(1) == y_test
        t = S.mcnemar_exact(correct["hc_xgb"], c)
        abl.append({"dropped_group": grp_name, "test_acc": float(c.mean()),
                    "delta_vs_full": float(c.mean() - correct["hc_xgb"].mean()), "p": t["p"]})
    abl = pd.DataFrame(abl)
    abl["p_holm"] = S.holm(abl["p"].to_numpy())
    abl.to_csv(OUT / "ablation_groups.csv", index=False)
    log("ablation\n", abl.to_string())

    # ---- figures -------------------------------------------------------------------
    best_valid = res.dropna(subset=["valid_acc"]).sort_values("valid_acc", ascending=False).iloc[0]["model"]
    fig_models(res)
    fig_layers(layers, best_layer, res)
    fig_shap(grp, heat)
    fig_confusion(y_test, preds[best_valid], LABELS[best_valid])

    summary = {
        "n_tracks": idx.split.value_counts().to_dict(), "best_mert_layer": best_layer,
        "results": res.to_dict("records"), "cochran_q": q,
        "mcnemar": mc.to_dict("records"), "ablation": abl.to_dict("records"),
        "shap_groups": grp.to_dict("records"), "leakage_audit": audit.to_dict("records"),
        "best_model_by_valid": best_valid,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log("done")


# ---------------------------------------------------------------------------------
def fig_models(res):
    import matplotlib.pyplot as plt
    order = ["hc_lr", "hc_xgb", "mert_lr", "mert_xgb", "clap_lr", "clap_xgb", "clap_zs"]
    r = res.set_index("model").loc[order]
    color = {"lr": P.BLUE, "xgb": P.ORANGE, "zs": P.AQUA}
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ys = np.arange(len(order))[::-1]
    for yv, m in zip(ys, order):
        kind = m.split("_")[1]
        row = r.loc[m]
        ax.errorbar(row.test_acc, yv, xerr=[[row.test_acc - row.ci_lo], [row.ci_hi - row.test_acc]],
                    fmt="o", color=color[kind], ms=6, capsize=0, elinewidth=2)
        ax.text(row.ci_hi + 0.012, yv, f"{row.test_acc:.1%}", va="center", fontsize=8, color=P.TEXT)
    ax.set_yticks(ys, [LABELS[m] for m in order])
    ax.set_xlabel("Test accuracy on 290 artist-disjoint tracks (95% bootstrap CI)")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.1, color=P.GRAY, lw=0.8, ls=":")
    ax.text(0.11, ys[0] + 0.45, "chance (10%)", fontsize=7, color=P.TEXT2, va="bottom")
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.9)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=c, label=l) for l, c in
         (("Logistic regression", P.BLUE), ("XGBoost", P.ORANGE), ("Zero-shot", P.AQUA))]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0.13, 0.0), fontsize=7.5)
    ax.grid(axis="y", visible=False)
    P.save(fig, OUT / "fig_models.png")


def fig_layers(layers, best_layer, res):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(layers.layer, layers.valid_acc, color=P.BLUE, label="validation")
    ax.plot(layers.layer, layers.test_acc, color=P.ORANGE, label="test")
    ax.scatter([best_layer], [layers.set_index("layer").loc[best_layer, "valid_acc"]], s=40,
               color=P.BLUE, zorder=3)
    ax.annotate(f"chosen layer {best_layer}", (best_layer, layers.set_index("layer").loc[best_layer, "valid_acc"]),
                xytext=(6, 8), textcoords="offset points", fontsize=7.5, color=P.TEXT)
    hc_acc = float(res.set_index("model").loc["hc_xgb", "test_acc"])
    ax.axhline(hc_acc, color=P.GRAY, lw=1, ls="--")
    ax.text(11.0, hc_acc + 0.008, "handcrafted + XGBoost (test)", ha="left", fontsize=7.5, color=P.TEXT2)
    ax.set_xlabel("MERT layer (0 = CNN features)")
    ax.set_ylabel("Accuracy (LogReg probe)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.12), fontsize=7.5)
    P.save(fig, OUT / "fig_mert_layers.png")


def fig_shap(grp, heat):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.0),
                                 gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.75})
    g = grp.sort_values("mean_abs_shap")
    a1.barh(g.group, g.mean_abs_shap, color=P.BLUE, height=0.6)
    a1.set_xlabel("mean |SHAP| toward true genre")
    a1.grid(axis="y", visible=False)
    norm = heat / heat.sum(1, keepdims=True)
    vmax = float(norm.max())
    a2.imshow(norm, cmap=P.SEQ, aspect="auto", vmin=0, vmax=vmax)
    a2.set_xticks(range(len(GROUPS)), [k.split(" ")[0] for k in GROUPS], rotation=30, ha="right")
    a2.set_yticks(range(len(GENRES)), GENRES)
    a2.grid(False)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            a2.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", fontsize=6.5,
                    color=P.ink_on(P.SEQ, norm[i, j] / vmax))
    a2.set_title("share of attribution per genre", fontsize=8.5)
    P.save(fig, OUT / "fig_shap.png")


def fig_confusion(y, pred, title):
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y, pred, labels=range(len(GENRES)))
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap=P.SEQ)
    ax.set_xticks(range(len(GENRES)), GENRES, rotation=45, ha="right")
    ax.set_yticks(range(len(GENRES)), GENRES)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color=P.ink_on(P.SEQ, cm[i, j] / cm.max()))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    P.save(fig, OUT / "fig_confusion.png")


def replot():
    """Re-draw every Exp 1 figure from the saved tables (no model fitting)."""
    P.setup()
    res = pd.read_csv(OUT / "model_results.csv")
    layers = pd.read_csv(OUT / "mert_layer_probe.csv")
    best_layer = int(json.loads((OUT / "summary.json").read_text())["best_mert_layer"])
    grp = pd.read_csv(OUT / "shap_groups.csv")
    heat = pd.read_csv(OUT / "shap_group_by_genre.csv", index_col=0).loc[GENRES, list(GROUPS)].to_numpy()
    pred = pd.read_csv(OUT / "test_predictions.csv")
    best_valid = res.dropna(subset=["valid_acc"]).sort_values("valid_acc", ascending=False).iloc[0]["model"]
    fig_models(res)
    fig_layers(layers, best_layer, res)
    fig_shap(grp, heat)
    fig_confusion(pred["label"].to_numpy(), pred[f"pred_{best_valid}"].to_numpy(), LABELS[best_valid])


if __name__ == "__main__":
    import sys
    replot() if "--replot" in sys.argv else main()
