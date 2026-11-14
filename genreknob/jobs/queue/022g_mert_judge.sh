#!/usr/bin/env bash
# Job 022g: CLAP-independent judge (MERT layer 5 + LogReg) on the diagnostic outputs and the source clips.
source ~/gk-env.sh
cd ~/gk
$MERT_PY -m genreknob.mert_judge fit
$MERT_PY - <<'PY'
import pandas as pd
s = pd.read_csv("results/exp2/sources.csv")
s = s[s.split == "valid"].rename(columns={"genre": "target"})
s.to_csv("results/diag/sources_valid.csv", index=False)
PY
for k in "" 2 3 4; do
  CUDA_VISIBLE_DEVICES=2 $MERT_PY -m genreknob.mert_judge score results/diag/diag_outputs$k.csv out results/diag/mert_outputs$k.csv
done
CUDA_VISIBLE_DEVICES=2 $MERT_PY -m genreknob.mert_judge score results/diag/sources_valid.csv clip results/diag/mert_sources_valid.csv
$MERT_PY - <<'PY'
import pandas as pd
s = pd.read_csv("results/diag/mert_sources_valid.csv")
print("MERT judge on 30 valid sources:", s.hit_mert.mean())
d = pd.concat([pd.read_csv(f"results/diag/mert_outputs{k}.csv") for k in ["", "2", "3", "4"]]).drop_duplicates(["variant", "track", "target"])
d = d[d.target != d.src_genre]
print(d.groupby("variant")[["hit_clap", "hit_radar", "hit_mert", "melody"]].mean().round(2).to_string())
print(d.groupby("variant").top_mert.value_counts().unstack(fill_value=0).to_string())
PY
