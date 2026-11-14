# ISC353 Group 4: GenreKnob

**Raising the success rate of genre swaps with leakage-free genre judges and training-free steering.**

A genre swap re-renders a song in another genre while keeping its tune. The project has two experiments:

1. **Genre recognition without leakage.** Our earlier Kaggle-tutorial baseline reached 90% on GTZAN by following a Kaggle tutorial, but its random split of 3-s segments leaked songs into the test set (53.1% on an artist-disjoint split). Pretrained representations (MERT, CLAP) raise accuracy to 78.6–84.1% (exact McNemar, Holm-corrected). These classifiers become the genre judges of Experiment 2.
2. **GenreKnob.** A training-free genre-swap tool on the frozen ACE-Step 1.5 text-to-music model: audio-conditioned covers keep the melody, activation steering raises the judged target genre, and a keep-original knob lets the user trade melody for genre (paired Wilcoxon tests on 100 songs x 9 target genres).

## Quick start

```bash
git clone https://github.com/yifeica0/ISC353_a2_group4_GenreKnob.git
cd ISC353_a2_group4_GenreKnob/genreknob      # all code and job scripts live here
ln -sfn "$PWD" ~/gk                          # the job scripts expect the code at ~/gk
bash jobs/queue/001_install.sh               # create the two Python environments (GPU server)
```

Then download GTZAN (next section) and run the jobs in the order given in [`genreknob/README.md`](genreknob/README.md).

## Repository layout

| Path | Contents |
|---|---|
| [`genreknob/`](genreknob/) | All code of the project: Python package, the job scripts that produce every result, the artist-disjoint split and helper tools. **Start with [`genreknob/README.md`](genreknob/README.md)** (requirements, run order, live demo). |
| `gtzan-dataset-music-genre-classification/` | **Not in the repository.** Put the GTZAN dataset here (see below). |

## Data (not included)

GTZAN is downloaded from Kaggle ([andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)):

```bash
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip -d gtzan-dataset-music-genre-classification
# -> gtzan-dataset-music-genre-classification/Data/{genres_original, images_original, features_3_sec.csv, features_30_sec.csv}
```

The Kaggle CLI needs your own API token (`~/.kaggle/kaggle.json` or a local `.env`, which is git-ignored). The final-project code reads the audio from `~/genreknob-data/genres_original` by default (override with `GK_GTZAN`); `genreknob/tools/mac_sync.sh` copies it there on the GPU server. The Kaggle copy of `jazz.00054.wav` is corrupted and is skipped automatically.

## Reproducing

Generation needs an NVIDIA GPU (we used 4 x A100) and two Python environments created by `genreknob/jobs/queue/001_install.sh`, so the code is provided as this repository rather than a Colab notebook. Follow [`genreknob/README.md`](genreknob/README.md) for the run order. All outputs (tables, figures, logs, generated audio) are written to `genreknob/results/`, which is not tracked in git.

## What is not in the repository

* The GTZAN audio (download it as above).
* Model weights (ACE-Step 1.5, MERT, CLAP) and caches; the job scripts download them.
* Experiment outputs (`genreknob/results/`): re-run the jobs to regenerate them. GTZAN contains excerpts of commercial recordings, so audio generated from it should not be published.

## Team

CAO Yifei, MATSUMOTO Koh, HANAMI Saki
