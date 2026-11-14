# GenreKnob: README

**ISC353 Group 4:** CAO Yifei, MATSUMOTO Koh, HANAMI Saki

**Code:** <https://github.com/yifeica0/ISC353_a2_group4_GenreKnob>

GenreKnob raises the success rate of *genre swaps* (re-rendering a song in another genre while keeping its tune) with leakage-free genre judges and training-free steering. No model is trained or fine-tuned.

* **Experiment 1: genre recognition without leakage (H1).** GTZAN on the artist-disjoint *fault-filtered* split (Kereliuk, Sturm & Larsen 2015). Handcrafted features (57 librosa descriptors) vs MERT-v1-330M vs CLAP, each with logistic regression and XGBoost, plus CLAP zero-shot. Exact McNemar tests (Holm-corrected), Cochran's Q, bootstrap CIs, TreeSHAP, feature-group ablation and a MERT layer probe.
* **Experiment 2: genre swap with the frozen ACE-Step 1.5 model (H2, H3).** Three conditions: text only (T), cover = text + source audio (C), and cover + contrastive activation steering (CS). Genre is judged by CLAP zero-shot and checked by two GTZAN-trained judges; melody kept is a centred-chroma similarity. Paired Wilcoxon tests on 100 test songs x 9 target genres.
* **GenreKnob demo:** a Gradio app. Upload or hum a song, pick a genre and turn the knobs (keep-original, genre boost, genre fusion, instrument, mood, tempo, key, repaint one section).

The code is provided as this GitHub repository rather than a Colab notebook, because generation needs a multi-GPU server and two separate Python environments. All paths below are relative to the repository root unless they start with `~`.

## 1. Get the code

Run on the GPU server (Linux):

```bash
git clone https://github.com/yifeica0/ISC353_a2_group4_GenreKnob.git
cd ISC353_a2_group4_GenreKnob
ln -sfn "$PWD/genreknob" ~/gk     # every job script runs from ~/gk
```

## 2. Requirements

* Linux with NVIDIA GPUs and a CUDA 12.x driver. The job scripts assume **4 GPUs** (we used 4 x A100 80GB; they set `CUDA_VISIBLE_DEVICES=0..3`).
* About 30 GB of free disk and internet access to GitHub and Hugging Face (model weights download on first use).
* No sudo. Step 3 installs [uv](https://github.com/astral-sh/uv) and both Python environments:
    * **ACE env:** ACE-Step 1.5 (pinned commit) from its own `uv.lock` (torch 2.10 + CUDA 12.8, transformers 4.57, gradio 6.2) plus librosa and scikit-learn. Used for generation, CLAP, the genre radar and the demo.
    * **MERT env:** Python 3.11, torch 2.4.1 (CUDA 12.1), transformers 4.38.2 (MERT's remote code needs an older transformers), xgboost 2.1.4, scikit-learn 1.5.2, librosa 0.10.2.

## 3. Install

```bash
bash ~/gk/jobs/queue/001_install.sh     # ~15-25 min; writes ~/gk-env.sh, used by every later job
bash ~/gk/jobs/queue/002_smoke.sh       # optional: smoke test of both environments
```

## 4. Download the data (GTZAN, not included)

GTZAN comes from Kaggle ([andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)). The Kaggle CLI (`pip install kaggle`) needs your own API token in `~/.kaggle/kaggle.json`; you can also download the zip in a browser. Run from the repository root:

```bash
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip -d gtzan-dataset-music-genre-classification
mkdir -p ~/genreknob-data
ln -sfn "$PWD/gtzan-dataset-music-genre-classification/Data/genres_original" ~/genreknob-data/genres_original
```

The code reads the audio from `~/genreknob-data/genres_original` (override with the environment variable `GK_GTZAN`). The dataset folder is git-ignored. The Kaggle copy of `jazz.00054.wav` is corrupted and is skipped automatically.

## 5. Reproduce the experiments

Each job is a plain bash script. Run them in this order (all outputs go to `genreknob/results/`, which `~/gk/results/` points to):

```bash
cd ~/gk
bash jobs/queue/004_clap_select.sh      # choose the CLAP zero-shot checkpoint on validation tracks
bash jobs/queue/010_exp1.sh             # Experiment 1: features, 7 classifiers, tests, SHAP, ablation
bash jobs/queue/020_ace_prep.sh         # Experiment 2: source clips, genre radar, steering vectors
for j in 022a_diag 022c_diag2 022d_diag3 022e_diag4 022f_diag5 022g_mert_judge; do
  bash jobs/queue/$j.sh                 # validation diagnostics of generator, judges, melody metric
done
bash jobs/queue/040_exp2v2_strength.sh  # choose keep-original strength (validation songs)
bash jobs/queue/041_exp2v2_layers.sh    # choose steering layers (validation songs)
bash jobs/queue/042_exp2v2_alpha.sh     # choose steering strength alpha (validation songs)
bash jobs/queue/043_exp2v2_test.sh      # test stage: 900 pairs x {T, C, CS}, three judges, statistics
bash jobs/queue/046_exp2_stats_v2b.sh   # final Experiment 2 statistics (chance level, knob reference)
bash jobs/queue/044_demo_cache_v2.sh    # optional: offline demo gallery (results/demo_cache/index.html)
```

* Jobs 000, 003, 022b and 045-049 are environment checks and demo tests. Jobs 021-024 and 030 are a first Experiment 2 run with wrong settings (timestep shift 1, covers from pure noise) that the diagnostics caught; they are kept for transparency and are not needed.
* Instead of running jobs by hand, `tools/gk_runner.sh` (copied to the server) runs every new `jobs/queue/*.sh` once in name order. From a Mac, `bash genreknob/tools/mac_sync.sh USER@GPU_SERVER` copies GTZAN to the server, starts the runner and syncs code and results every 15 s; it expects the clone at `~/ISC353_a2_group4` (use `git clone <url> ~/ISC353_a2_group4`).
* Fixed seeds (`SEED = 0` in `genreknob/genreknob/config.py`; per-pair generation seeds hash song and target). Experiment 1 reproduces exactly; GPU generation is not bit-exact, so individual clips change on a re-run but condition means do not (a repeated setting gave target probability 0.150 vs 0.148, melody 0.397 vs 0.393).

The leakage audit on the Kaggle feature CSVs alone runs on a CPU (needs pandas, scikit-learn, xgboost), from the repository root:

```bash
python genreknob/analysis/split_audit.py \
  --csv3  gtzan-dataset-music-genre-classification/Data/features_3_sec.csv \
  --csv30 gtzan-dataset-music-genre-classification/Data/features_30_sec.csv \
  --splits genreknob/splits/fault_filtered
```

## 6. Run the live demo

From your laptop, one line starts the demo on the GPU server and opens a private SSH tunnel (keep the window open):

```bash
ssh -L 7860:127.0.0.1:7860 USER@GPU_SERVER 'bash ~/gk/tools/start_demo.sh; sleep 36000'
```

When it prints `Ready`, open **http://localhost:7860** in the browser. Loading takes about a minute; each 10-s swap takes about 2 s on an A100. Because the page is served from `localhost`, the browser allows the microphone, so humming works (turn keep-original down to about 0.1 for humming). Nothing is published on the internet. Stop it with `ssh USER@GPU_SERVER 'pkill -f genreknob.app'`. `tools/start_share.sh` instead opens a temporary password-protected public Gradio link.

## 7. Outputs and key results

The jobs write every table, figure and log to `genreknob/results/` (not tracked in git).

| File (under `genreknob/results/`) | What it shows |
|---|---|
| `exp1/leakage_audit.csv` | Random-split vs grouped vs artist-disjoint accuracy with the same features |
| `exp1/model_results.csv`, `mcnemar.csv`, `fig_models.png` | Test accuracy of all classifiers and the pairwise tests (H1) |
| `exp1/shap_groups.csv`, `ablation_groups.csv`, `fig_shap.png` | Feature importance of the handcrafted model |
| `exp1/mert_layer_probe.csv`, `fig_mert_layers.png` | Genre information per MERT layer |
| `diag/diag_summary*.txt` | Validation diagnostics of generator, judges and melody metric |
| `exp2/choices.json` | Settings chosen on validation songs |
| `exp2/test_wilcoxon.csv`, `test_condition_summary.csv`, `test_knob_curve.csv`, `fig_exp2_*.png` | Genre-swap results (H2, H3, knob) |

Our run (290 artist-disjoint test tracks; 100 test songs x 9 targets):

* **Leakage:** same features + XGBoost: 89.9% on a random 3-s segment split, 73.1% grouped by song, 49.1% (segments) / 53.1% (tracks) on the artist-disjoint split.
* **H1:** test accuracy CLAP+LR 84.1%, CLAP+XGB 83.8%, MERT+LR 78.6%, MERT+XGB 78.3%, CLAP zero-shot 71.4%, handcrafted+LR 56.2%, handcrafted+XGB 53.1%. MERT+LR vs handcrafted+XGB: Holm-adjusted p = 2.0e-14.
* **Chosen settings:** keep-original 0.3, steering at DiT layers 0 and 5, alpha = 2.
* **H2:** melody kept C 0.40 vs T 0.06 (Wilcoxon, Holm-adjusted p = 1.6e-17), at a cost in target-genre probability (0.11 vs 0.19).
* **H3:** CLAP target-genre probability CS 0.24 vs C 0.11 (r = 0.87, Holm-adjusted p = 1.6e-13), melody 0.37 vs 0.40.

## 8. Repository layout

| Path | Contents |
|---|---|
| `README.md` | This file (identical copies in `genreknob/README.md` and `genreknob/README.pdf`) |
| `genreknob/genreknob/` | Python package (modules below) |
| `genreknob/jobs/queue/` | The job scripts that produce every result |
| `genreknob/splits/fault_filtered/` | Artist-disjoint GTZAN split lists (train / valid / test) |
| `genreknob/analysis/split_audit.py` | Leakage audit on the Kaggle feature CSVs |
| `genreknob/tools/` | Job runner, Mac sync script, demo launchers and demo self-tests |

| Module | Role |
|---|---|
| `config.py` | Paths, genres, constants (all overridable by environment variables) |
| `data.py` | GTZAN index and split, audio loading, segmentation |
| `features.py` | 57 handcrafted descriptors (same set as the Kaggle CSV) and feature groups |
| `embed.py` | MERT (all 25 layers) and CLAP embeddings; CLAP zero-shot |
| `exp1.py` | Experiment 1: tuning on validation data, test evaluation, tests, SHAP, ablation, figures (`--replot` redraws figures) |
| `stats.py` | Exact McNemar, Holm, Cochran's Q, bootstrap CIs, Wilcoxon with rank-biserial r |
| `ace.py` | ACE-Step 1.5 wrapper (text2music, cover, repaint; timestep shift 3) and steering hooks |
| `steer.py` | Contrastive steering vectors (genres, instruments, moods) |
| `judge.py` | CLAP zero-shot judge, genre radar (CLAP + logistic regression), CLAP checkpoint selection |
| `mert_judge.py` | Second, CLAP-independent judge: MERT layer 5 + logistic regression |
| `melody.py` | Melody-kept metric (centred CQT chroma, lag-searched cosine) |
| `diag.py` | Validation diagnostics of generator, judges and melody metric |
| `exp2.py` | Experiment 2: sources, plans, multi-GPU generation, scoring, selection, statistics, figures |
| `plots.py` | Shared plotting helpers |
| `app.py` | Gradio demo and offline gallery renderer |

## 9. Not included; licences

Not in the repository: the GTZAN audio (step 4), model weights (ACE-Step 1.5, MERT, CLAP; downloaded by the jobs) and all outputs (re-run step 5). ACE-Step 1.5 is MIT-licensed; MERT and CLAP weights are used under their own Hugging Face licences. GTZAN contains excerpts of commercial recordings; it is used for research and teaching only, and audio generated from it should not be published.
