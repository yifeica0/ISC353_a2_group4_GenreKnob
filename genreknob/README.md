# GenreKnob

**Leakage-free genre recognition and a training-free genre-swap toy**
ISC353 group project, Group 4.

This folder contains everything needed to reproduce both experiments and to run the live demo.

* **Experiment 1: genre recognition without leakage.** The earlier baseline's 90% XGBoost accuracy came from a leaky split (3-s pieces of the same song in training and test data). We re-evaluate on the artist-disjoint *fault-filtered* GTZAN split and compare handcrafted features (57 librosa descriptors) with two pretrained representations (MERT-v1-330M, CLAP), each with logistic regression and XGBoost, plus CLAP zero-shot. Exact McNemar tests (Holm-corrected), Cochran's Q, bootstrap intervals, TreeSHAP, feature-group ablation and a MERT layer probe.
* **Experiment 2: genre swap with ACE-Step 1.5.** A frozen text-to-music model re-renders a 10-s song clip in another genre. Three conditions: text only (T), cover = text + source audio (C), and cover + activation steering with a contrastive genre direction added to the model's cross-attention outputs (CS). Genre is judged by CLAP zero-shot (never trained on GTZAN) and checked by two GTZAN-trained judges (CLAP + logistic regression, MERT + logistic regression); melody kept is a centred-chroma similarity. Paired Wilcoxon tests on 100 test songs x 9 target genres.
* **GenreKnob demo:** a Gradio app. Upload or hum a song, pick a genre and turn the knobs: keep-original, genre boost, genre fusion, instrument, mood, tempo, key, repaint one section. A live genre radar shows what the Experiment 1 classifier hears.

## Folder layout

| Path | Contents |
|---|---|
| `genreknob/` | Python package (see table below) |
| `jobs/queue/` | The exact job scripts that produced every result, run in name order |
| `results/` | Created when the jobs run: tables (CSV/JSON), figures (PNG), logs, demo gallery. Not tracked in git |
| `splits/fault_filtered/` | Artist-disjoint GTZAN split lists (Kereliuk, Sturm & Larsen 2015) |
| `analysis/split_audit.py` | Leakage audit on the Kaggle feature CSVs |
| `tools/` | GPU job runner, Mac sync script, demo launcher and demo self-test |

| Module | Role |
|---|---|
| `config.py` | Paths, genres, constants (all overridable by environment variables) |
| `data.py` | GTZAN index and split, audio loading, segmentation |
| `features.py` | 57 handcrafted descriptors (same set as the Kaggle CSV) and feature groups |
| `embed.py` | MERT (all 25 layers) and CLAP embeddings; CLAP zero-shot |
| `exp1.py` | Experiment 1: tuning on validation data, test evaluation, McNemar and Cochran, SHAP, ablation, figures (`--replot` redraws figures from the saved tables) |
| `stats.py` | Exact McNemar, Holm, Cochran's Q, bootstrap CIs, Wilcoxon with rank-biserial r |
| `ace.py` | ACE-Step 1.5 wrapper (text2music, cover, repaint; timestep shift 3) and steering hooks |
| `steer.py` | Contrastive steering vectors (genres, instruments, moods) |
| `judge.py` | CLAP zero-shot judge, genre radar (CLAP embedding + logistic regression), CLAP checkpoint selection |
| `mert_judge.py` | Second, CLAP-independent judge: MERT layer 5 + logistic regression |
| `melody.py` | Melody-kept metric (centred CQT chroma, lag-searched cosine) |
| `diag.py` | Validation diagnostics of generator, judges and melody metric (`results/diag/`) |
| `exp2.py` | Experiment 2: sources, plans, multi-GPU generation, scoring, selection, statistics, figures |
| `app.py` | Gradio demo and offline gallery renderer |

## Getting the code

```bash
git clone https://github.com/yifeica0/ISC353_a2_group4_GenreKnob.git
cd ISC353_a2_group4_GenreKnob/genreknob
ln -sfn "$PWD" ~/gk      # the job scripts expect the code at ~/gk
```

All commands below are run from this `genreknob/` folder on the GPU server (`tools/mac_sync.sh` is an alternative that syncs it from a Mac to `~/gk`).

## Requirements

* Linux with an NVIDIA GPU (used: 4 x A100 80GB, CUDA 12.x driver), about 30 GB of disk, internet access to Hugging Face.
* [uv](https://github.com/astral-sh/uv) (installed by `jobs/queue/001_install.sh`, no sudo needed).
* Two Python environments, both created by `001_install.sh`:
    * **ACE env:** ACE-Step 1.5 from its own `uv.lock` (torch 2.10 + CUDA 12.8, transformers 4.57, gradio 6.2) plus librosa and scikit-learn. Used for generation, CLAP, the radar and the demo.
    * **MERT env:** Python 3.11, torch 2.4.1 (CUDA 12.1), transformers 4.38.2 (MERT's remote code needs an older transformers), xgboost 2.1.4, scikit-learn 1.5.2, librosa 0.10.2.
* Data: GTZAN from Kaggle (`andradaolteanu/gtzan-dataset-music-genre-classification`), folder `genres_original/` copied to `~/genreknob-data/genres_original`. The Kaggle copy of `jazz.00054.wav` is corrupted and is skipped automatically.

## Reproducing the results

Each job in `jobs/queue/` is a plain bash script; run them in name order:

```bash
bash jobs/queue/001_install.sh          # environments + ACE-Step (no sudo)
bash jobs/queue/004_clap_select.sh      # choose the CLAP zero-shot judge on validation tracks
bash jobs/queue/010_exp1.sh             # Experiment 1 (features, classifiers, tests, SHAP)
bash jobs/queue/020_ace_prep.sh         # Exp 2 source clips and genre radar
bash jobs/queue/022a_diag.sh ... 022g_mert_judge.sh   # validation diagnostics (results/diag)
bash jobs/queue/040_exp2v2_strength.sh  # steering vectors + choose keep-original strength (validation)
bash jobs/queue/041_exp2v2_layers.sh    # choose steering layers (validation)
bash jobs/queue/042_exp2v2_alpha.sh     # choose steering strength (validation)
bash jobs/queue/043_exp2v2_test.sh      # test generation, three judges, statistics
bash jobs/queue/044_demo_cache_v2.sh    # offline demo gallery
bash jobs/queue/045_demo_selftest.sh    # start the demo locally and drive it through its API
```

Jobs 021-024 and 030 belong to a first Experiment 2 run that used ACE's default timestep shift (1) and covers started from pure noise; the validation diagnostics (`results/diag/`) showed both were wrong, so that run was stopped before the test stage.

All random choices use fixed seeds (`SEED = 0` in `config.py`; per-pair generation seeds are a hash of song and target genre). Experiment 1 reproduces exactly; GPU generation in Experiment 2 is not bit-exact, so a re-run changes individual clips but not the condition means (a repeated setting gave target probability 0.150 vs 0.148 and melody 0.397 vs 0.393). Figures of Experiment 1 can be redrawn from the tables with `python -m genreknob.exp1 --replot`.

## Running the live demo

From the Mac, one line starts the demo on the GPU server and opens a private SSH tunnel (keep the terminal window open during the talk):

```bash
ssh -L 7860:127.0.0.1:7860 USER@GPU_SERVER 'bash ~/gk/tools/start_demo.sh; sleep 36000'
```

When it prints `Ready`, open **http://localhost:7860** in your browser. For humming, turn keep-original down to about 0.1 so the genre can take over. Loading takes about a minute; each 10-s swap takes about 2 s on an A100. Replace `USER@GPU_SERVER` with your own server login. Because the page is served from `localhost`, the browser allows the microphone, so humming works. The demo is not published on the internet (Gradio's public `--share` link is off by default). Stop it with `ssh USER@GPU_SERVER 'pkill -f genreknob.app'`.

`jobs/queue/044_demo_cache_v2.sh` pre-renders an offline gallery into `results/demo_cache/` (open `index.html`) that works without the GPU.

## Key outputs (generated under `results/` by the jobs)

| File | What it shows |
|---|---|
| `results/exp1/leakage_audit.csv` | Random-split vs grouped vs artist-disjoint accuracy with the same features |
| `results/exp1/model_results.csv`, `mcnemar.csv`, `fig_models.png` | Test accuracy of all classifiers and the planned pairwise tests |
| `results/exp1/shap_groups.csv`, `shap_group_by_genre.csv`, `ablation_groups.csv`, `fig_shap.png` | Feature importance of the handcrafted model |
| `results/exp1/mert_layer_probe.csv`, `fig_mert_layers.png` | Genre information per MERT layer |
| `results/diag/diag_summary*.txt`, `mert_outputs*.csv` | Validation diagnostics of generator, judges and melody metric |
| `results/exp2/choices.json` | Settings chosen on validation songs |
| `results/exp2/test_wilcoxon.csv`, `test_mcnemar_pairs.csv`, `test_condition_summary.csv`, `fig_exp2_*.png` | Genre-swap results |
| `results/demo_cache/`, `results/demo_selftest/` | Offline gallery and the demo self-test outputs |

## Licences and data use

ACE-Step 1.5 (code and weights) is MIT-licensed. MERT and CLAP weights are downloaded from Hugging Face under their own licences. GTZAN contains excerpts of commercial recordings; it is used here for research and teaching only, and swapped versions of GTZAN songs should not be published.
