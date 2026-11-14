"""Statistical tests used in both experiments."""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact (binomial) McNemar test on paired correctness vectors."""
    a = np.asarray(correct_a, bool)
    b = np.asarray(correct_b, bool)
    n01 = int(np.sum(a & ~b))   # A right, B wrong
    n10 = int(np.sum(~a & b))   # A wrong, B right
    n = n01 + n10
    p = 1.0 if n == 0 else float(st.binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"a_only": n01, "b_only": n10, "discordant": n, "p": p,
            "acc_a": float(a.mean()), "acc_b": float(b.mean()),
            "diff": float(a.mean() - b.mean())}


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def cochran_q(correct: np.ndarray) -> dict:
    """Cochran's Q for k related binary outcomes. `correct`: (n_items, k)."""
    x = np.asarray(correct, float)
    n, k = x.shape
    col = x.sum(0)
    row = x.sum(1)
    total = x.sum()
    denom = k * total - np.sum(row ** 2)
    q = 0.0 if denom == 0 else (k - 1) * (k * np.sum(col ** 2) - total ** 2) / denom
    p = float(st.chi2.sf(q, k - 1))
    return {"Q": float(q), "df": k - 1, "p": p}


def bootstrap_ci(values: np.ndarray, func=np.mean, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI over the first axis (items)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    n = len(v)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([func(v[i]) for i in idx])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                          seed: int = 0) -> tuple[float, float]:
    """CI of mean(a) - mean(b) resampling items jointly (paired)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return bootstrap_ci(d, np.mean, n_boot=n_boot, seed=seed)


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test with matched-pairs rank-biserial r."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    nz = d[d != 0]
    if len(nz) == 0:
        return {"n": len(d), "W": 0.0, "p": 1.0, "r_rb": 0.0,
                "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "median_diff": 0.0}
    res = st.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    ranks = st.rankdata(np.abs(nz))
    r_plus = ranks[nz > 0].sum()
    r_minus = ranks[nz < 0].sum()
    r_rb = float((r_plus - r_minus) / ranks.sum())
    return {"n": int(len(d)), "W": float(res.statistic), "p": float(res.pvalue),
            "r_rb": r_rb, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "median_diff": float(np.median(d))}


def balance_score(hit: float, melody: float) -> float:
    """Harmonic mean of genre-hit rate and normalised melody preservation (both 0..1)."""
    hit, melody = max(hit, 0.0), max(melody, 0.0)
    return 0.0 if hit + melody == 0 else 2 * hit * melody / (hit + melody)
