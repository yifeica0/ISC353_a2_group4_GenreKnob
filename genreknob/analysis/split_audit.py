"""
Split audit for the GTZAN genre classifier (ISC353 group 4).

Why: the original notebook used train_test_split on features_3_sec.csv, so 3-second
segments of the SAME 30-s track appear in both train and test (100% of test tracks
leak). This script compares four protocols:
  A  random 70/30 split on 3-s segments (original notebook protocol -> leaky)
  B  5-fold CV grouped by track (no track leakage)
  C  5-fold CV on 30-s track-level features
  D  artist-disjoint "fault-filtered" split (Kereliuk, Sturm & Larsen 2015;
     443 train / 197 valid / 290 test tracks; train+valid used for fitting)

Usage:
  python split_audit.py --csv3 ../../gtzan-dataset-music-genre-classification/Data/features_3_sec.csv \
                        --csv30 ../../gtzan-dataset-music-genre-classification/Data/features_30_sec.csv \
                        --splits splits_fault_filtered
Uses XGBoost (as in the notebook) if installed, otherwise sklearn HistGradientBoosting.
"""
import argparse, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def booster():
    try:
        from xgboost import XGBClassifier
        return 'XGBoost', XGBClassifier(n_estimators=1000, learning_rate=0.05, tree_method='hist')
    except ImportError:
        return 'GBoost(HGB)', HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, random_state=0)

def models():
    bn, b = booster()
    return {bn: b, 'KNN(k=19)': KNeighborsClassifier(n_neighbors=19),
            'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=10, random_state=0, n_jobs=-1)}

def fit_pred(m, Xtr, ytr, Xte):  # scaler is fit on TRAIN only (inside the pipeline)
    return make_pipeline(MinMaxScaler(), m).fit(Xtr, ytr).predict(Xte)

def vote(groups, y, p):
    df = pd.DataFrame({'g': groups, 'y': y, 'p': p})
    v = df.groupby('g').agg(y=('y', 'first'), p=('p', lambda s: s.value_counts().index[0]))
    return (v.y == v.p).mean()

ap = argparse.ArgumentParser(); ap.add_argument('--csv3'); ap.add_argument('--csv30'); ap.add_argument('--splits')
a = ap.parse_args()
d3 = pd.read_csv(a.csv3); d30 = pd.read_csv(a.csv30)
d3['track'] = d3['filename'].str.rsplit('.', n=2).str[0]
feat = [c for c in d3.columns if c not in ('filename', 'length', 'label', 'track')]
le = LabelEncoder().fit(d3.label)
X3, y3, g3 = d3[feat].values, le.transform(d3.label), d3.track.values
rows = []

Xtr, Xte, ytr, yte, gtr, gte = train_test_split(X3, y3, g3, test_size=0.3, random_state=42)
print('A: fraction of test tracks that also appear in train =', np.isin(gte, gtr).mean())
for n, m in models().items():
    rows.append(('A_random_segment_split', n, accuracy_score(yte, fit_pred(m, Xtr, ytr, Xte)), np.nan, np.nan))

for n in models():
    s, t = [], []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X3, y3, g3):
        p = fit_pred(models()[n], X3[tr], y3[tr], X3[te]); s.append(accuracy_score(y3[te], p)); t.append(vote(g3[te], y3[te], p))
    rows.append(('B_grouped_by_track_5CV', n, np.mean(s), np.std(s), np.mean(t)))

X30, y30 = d30[feat].values, le.transform(d30.label)
for n in models():
    s = [accuracy_score(y30[te], fit_pred(models()[n], X30[tr], y30[tr], X30[te]))
         for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X30, y30)]
    rows.append(('C_track_level_30s_5CV', n, np.mean(s), np.std(s), np.nan))

rd = lambda f: {l.strip().split('/')[-1].replace('.wav', '') for l in open(f'{a.splits}/{f}') if l.strip()}
trn = rd('train_filtered.txt') | rd('valid_filtered.txt'); tst = rd('test_filtered.txt')
mtr, mte = d3.track.isin(trn).values, d3.track.isin(tst).values
for n, m in models().items():
    p = fit_pred(m, X3[mtr], y3[mtr], X3[mte])
    rows.append(('D_artist_disjoint_fault_filtered', n, accuracy_score(y3[mte], p), np.nan, vote(g3[mte], y3[mte], p)))

out = pd.DataFrame(rows, columns=['protocol', 'model', 'segment_acc', 'segment_acc_sd', 'track_vote_acc'])
print(out.round(3).to_string(index=False)); out.to_csv('split_audit_results.csv', index=False)
