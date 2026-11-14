"""GenreKnob live demo (Gradio).

Run on the GPU server (ACE env):
    $ACE_PY -m genreknob.app                  # serves on port 7860 (no public link by default)
Controls: input song / microphone hum, target genre, genre fusion, "keep original"
(how much of the source the cover starts from: ACE cover_noise_strength), genre
boost / instrument / mood sliders (activation steering),
tempo and key, and "repaint only a section". Output: the new audio, a genre radar
(before vs after) and how much of the melody survived.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from .config import CACHE, GENRE_TEXT, GENRES, RESULTS, WORK

KEYS = ["auto"] + [f"{n} {m}" for n in ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
                   for m in ("major", "minor")]
INSTRUMENTS = ["none", "piano", "electric guitar", "violin", "trumpet", "saxophone", "synthesizer"]
MOODS = ["none", "happy", "sad", "energetic", "calm"]


def _choices():
    p = RESULTS / "exp2" / "choices.json"
    return json.loads(p.read_text()) if p.exists() else {}


class Engine:
    def __init__(self):
        from .ace import Ace, GenreVectors
        from .judge import Judges
        self.ace = Ace()
        self.judges = Judges(eval_mode=False)
        ch = _choices()
        self.layers = ch.get("layers") or self.ace.steering.layers[len(self.ace.steering.layers) // 3:][:2]
        self.default_keep = float(ch.get("strength", 0.3))
        self.default_boost = float(ch.get("alpha", 1.0))
        self.vec = {}
        for fam in ("genre", "instrument", "mood"):
            p = CACHE / f"{fam}_vectors.pt"
            if p.exists():
                self.vec[fam] = GenreVectors(p)
        self.tmp = Path(tempfile.mkdtemp(prefix="gk_app_"))

    # ---- helpers ----------------------------------------------------------------
    def _clip(self, path: str, start: float, length: float) -> str:
        import librosa
        import soundfile as sf
        y, sr = librosa.load(path, sr=48000, mono=True, offset=float(start), duration=float(length))
        if len(y) < sr * 3:
            raise ValueError("Input is shorter than 3 seconds - record or choose a longer clip.")
        out = self.tmp / f"in_{time.time_ns()}.wav"
        sf.write(str(out), y, sr, subtype="PCM_16")
        return str(out)

    def _deltas(self, target, mix_genre, mix, boost, inst, inst_amt, mood, mood_amt):
        deltas = {}

        def add(d):
            for l, v in d.items():
                deltas[l] = deltas.get(l, 0) + v
        if "genre" in self.vec and boost > 0:
            w = {target: 1.0 - mix}
            if mix_genre and mix_genre != "none" and mix > 0:
                w[mix_genre] = w.get(mix_genre, 0.0) + mix
            add(self.vec["genre"].deltas(w, self.layers, boost))
        if "instrument" in self.vec and inst != "none" and inst_amt > 0:
            add(self.vec["instrument"].deltas({inst: 1.0}, self.layers, inst_amt))
        if "mood" in self.vec and mood != "none" and mood_amt > 0:
            add(self.vec["mood"].deltas({mood: 1.0}, self.layers, mood_amt))
        return deltas or None

    def _caption(self, target, mix_genre, mix, inst, mood):
        g = GENRE_TEXT[target]
        if mix_genre and mix_genre != "none" and mix >= 0.25:
            g = f"{g} and {GENRE_TEXT[mix_genre]} fusion"
        parts = [f"{g} music"]
        if inst != "none":
            parts.append(f"featuring {inst}")
        if mood != "none":
            parts.append(f"{mood} mood")
        parts.append("instrumental")
        return ", ".join(parts)

    def radar_probs(self, paths):
        r = self.judges.score_files(paths)
        return r.get("radar", r["clap"]), r["clap"]

    # ---- main entry -----------------------------------------------------------------
    def run(self, audio, start, length, target, mix_genre, mix, keep, boost, inst, inst_amt,
            mood, mood_amt, bpm, key, repaint_on, rp_start, rp_end, seed):
        from .melody import chroma_similarity, load_chroma
        if not audio:
            raise ValueError("Choose an example, upload a song or record something first.")
        t0 = time.time()
        src = self._clip(audio, start, length)
        cap = self._caption(target, mix_genre, mix, inst, mood)
        deltas = self._deltas(target, mix_genre, mix, boost, inst, inst_amt, mood, mood_amt)
        out = self.tmp / f"out_{time.time_ns()}.wav"
        if repaint_on and rp_end > rp_start:
            self.ace.generate(out, cap, task="repaint", src=src, duration=length, seed=int(seed),
                              repaint=(rp_start, rp_end), deltas=deltas,
                              bpm=int(bpm) if bpm else None, keyscale="" if key == "auto" else key)
        else:
            self.ace.generate(out, cap, task="cover", src=src, duration=length, seed=int(seed),
                              cover_strength=1.0, deltas=deltas, extra={"cover_noise_strength": float(keep)},
                              bpm=int(bpm) if bpm else None, keyscale="" if key == "auto" else key)
        gen_s = time.time() - t0
        radar, clap = self.radar_probs([src, str(out)])
        mel = chroma_similarity(load_chroma(src), load_chroma(str(out)))
        fig = radar_figure(radar[0], radar[1], target)
        top_b, top_a = GENRES[int(radar[0].argmax())], GENRES[int(radar[1].argmax())]
        md = (f"**Prompt:** {cap}  \n"
              f"**Genre radar:** before = **{top_b}** ({radar[0].max():.0%}), "
              f"after = **{top_a}** ({radar[1].max():.0%}); target *{target}* "
              f"{radar[0][GENRES.index(target)]:.0%} -> {radar[1][GENRES.index(target)]:.0%}  \n"
              f"**CLAP judge says:** {GENRES[int(clap[1].argmax())]}  \n"
              f"**Melody kept (chroma similarity):** {mel:.2f}  \n"
              f"**Generated in** {gen_s:.1f} s")
        return str(out), fig, md


def radar_figure(before, after, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import plots as P
    ang = np.linspace(0, 2 * np.pi, len(GENRES), endpoint=False)
    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_subplot(111, polar=True)
    for vals, color, label in ((before, P.GRAY, "before"), (after, P.BLUE, "after")):
        v = np.concatenate([vals, vals[:1]])
        a = np.concatenate([ang, ang[:1]])
        ax.plot(a, v, color=color, lw=2, label=label)
        ax.fill(a, v, color=color, alpha=0.15)
    ax.set_xticks(ang, [f"{g}*" if g == target else g for g in GENRES], fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75], ["", "", ""])
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


# Best steered swaps of the Experiment 2 test set: all three judges named the target and the
# melody was kept (chroma similarity 0.57-0.72). Picking one also sets the target genre and the
# seed that test run used, with the default keep-original and boost.
SHOWCASE = [("reggae.00083", "jazz"), ("blues.00024", "hiphop"), ("country.00057", "blues"),
            ("rock.00039", "classical"), ("classical.00011", "jazz"), ("rock.00013", "hiphop")]


def seed_of(track: str, target: str) -> int:
    """Same per-pair seed as Experiment 2 (exp2.seed_of)."""
    import zlib
    return zlib.crc32(f"{track}|{target}".encode()) % (2 ** 31 - 1)


def showcase_label(track: str, target: str) -> str:
    return f"\u2605 {track.split('.')[0]} -> {target} ({track}, test-set best)"


def examples():
    src = WORK / "exp2" / "sources"
    pref = ["classical", "jazz", "blues", "country", "reggae", "hiphop", "metal", "rock", "pop", "disco"]
    out = {}
    for track, target in SHOWCASE:
        f = src / f"{track}.wav"
        if f.exists():
            out[showcase_label(track, target)] = str(f)
    if src.exists():
        for g in pref:
            files = sorted(src.glob(f"{g}.*.wav"))
            if files:
                out[f"{g} ({files[0].stem})"] = str(files[0])
    return out


def _local_copies(ex):
    """Copy the example clips into a temp folder: Gradio only serves files from its allowed
    folders (temp dir, working dir), and the clips live in GK_WORK outside both."""
    import shutil
    d = Path(tempfile.mkdtemp(prefix="gk_examples_"))
    out = {}
    for k, path in ex.items():
        dst = d / Path(path).name
        shutil.copy(path, dst)
        out[k] = str(dst)
    return out


def build_ui(engine: Engine | None):
    import gradio as gr
    ex = _local_copies(examples())

    show = {showcase_label(t, g): (g, seed_of(t, g)) for t, g in SHOWCASE}

    def load_example(k, cur_target, cur_seed, cur_keep, cur_boost):
        if k in show:   # a showcase pair: also set its target, seed and the default knobs
            g, sd = show[k]
            return (ex.get(k), g, sd, engine.default_keep if engine else 0.3,
                    engine.default_boost if engine else 1.0)
        return ex.get(k), cur_target, cur_seed, cur_keep, cur_boost
    gallery = sorted((RESULTS / "demo_cache").glob("*.mp3")) if (RESULTS / "demo_cache").exists() else []

    def run(*args):
        if engine is None:
            raise gr.Error("Engine not loaded")
        try:
            return engine.run(*args)
        except Exception as e:  # show the message in the UI instead of a stack trace
            raise gr.Error(str(e))

    with gr.Blocks(title="GenreKnob") as demo:
        gr.Markdown("# GenreKnob\nTurn any melody into any genre. Hum, upload or pick a song, "
                    "choose a genre and turn the knobs.")
        with gr.Tab("Live"):
            with gr.Row():
                with gr.Column(scale=1):
                    pick = gr.Dropdown(["(none)"] + list(ex), value="(none)", label="Example songs (GTZAN test set)")
                    audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input: upload or hum")
                    with gr.Row():
                        start = gr.Slider(0, 60, value=0, step=0.5, label="Start (s)")
                        length = gr.Slider(10, 30, value=10, step=1, label="Length (s)")
                    target = gr.Radio(GENRES, value="rock", label="Target genre")
                    with gr.Row():
                        mix_genre = gr.Dropdown(["none"] + GENRES, value="none", label="Fuse with")
                        mix = gr.Slider(0, 1, value=0.0, step=0.05, label="Fusion amount")
                    keep = gr.Slider(0.0, 0.8, value=engine.default_keep if engine else 0.3, step=0.05,
                                     label="Keep original (how much of the song the remix starts from)")
                    boost = gr.Slider(0, 4, value=engine.default_boost if engine else 1.0, step=0.25,
                                      label="Genre boost (activation steering)")
                    with gr.Accordion("More knobs", open=False):
                        with gr.Row():
                            inst = gr.Dropdown(INSTRUMENTS, value="none", label="Add instrument")
                            inst_amt = gr.Slider(0, 4, value=1.0, step=0.25, label="Instrument amount")
                        with gr.Row():
                            mood = gr.Dropdown(MOODS, value="none", label="Mood")
                            mood_amt = gr.Slider(0, 4, value=1.0, step=0.25, label="Mood amount")
                        with gr.Row():
                            bpm = gr.Slider(0, 200, value=0, step=1, label="Tempo BPM (0 = keep)")
                            key = gr.Dropdown(KEYS, value="auto", label="Key")
                        repaint_on = gr.Checkbox(False, label="Only change a section (repaint)")
                        with gr.Row():
                            rp_start = gr.Slider(0, 30, value=4, step=0.5, label="Section start (s)")
                            rp_end = gr.Slider(0, 30, value=8, step=0.5, label="Section end (s)")
                        seed = gr.Number(value=7, precision=0, label="Seed")
                    go = gr.Button("Swap genre", variant="primary")
                with gr.Column(scale=1):
                    out_audio = gr.Audio(label="Result", type="filepath")
                    radar = gr.Plot(label="Genre radar (GTZAN-trained classifier)")
                    info = gr.Markdown()
            pick.change(load_example, inputs=[pick, target, seed, keep, boost],
                        outputs=[audio, target, seed, keep, boost], api_name="load_example")
            go.click(run, inputs=[audio, start, length, target, mix_genre, mix, keep, boost, inst, inst_amt,
                                  mood, mood_amt, bpm, key, repaint_on, rp_start, rp_end, seed],
                     outputs=[out_audio, radar, info])
        with gr.Tab("Offline gallery"):
            gr.Markdown("Pre-rendered results (work without the GPU).")
            for f in gallery[:60]:
                gr.Audio(value=str(f), label=f.stem.replace("__", " -> "), type="filepath")
    return demo


CACHE_SONGS = ["classical", "jazz", "country", "reggae"]
CACHE_GENRES = GENRES


def render_cache(engine: Engine):
    """Pre-render a gallery (MP3 + offline HTML page) so the demo survives a network failure."""
    import soundfile as sf
    out_dir = RESULTS / "demo_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    ex = examples()
    rows = []
    for song in CACHE_SONGS:
        key = next((k for k in ex if k.startswith(song)), None)
        if key is None:
            continue
        src = ex[key]
        y, sr = sf.read(src)
        sf.write(str(out_dir / f"{song}__original.mp3"), y, sr, format="MP3")
        for g in CACHE_GENRES:
            if g == song:
                continue
            path, fig, md = engine.run(src, 0, 10, g, "none", 0.0, engine.default_keep, engine.default_boost,
                                       "none", 0.0, "none", 0.0, 0, "auto", False, 0, 0, 7)
            a, asr = sf.read(path)
            mp3 = out_dir / f"{song}__{g}.mp3"
            sf.write(str(mp3), a, asr, format="MP3")
            fig.savefig(out_dir / f"{song}__{g}_radar.png", dpi=110)
            rows.append({"song": song, "target": g, "file": mp3.name, "info": md})
            print("cached", mp3.name, flush=True)
    # two fusions for the talk
    for song, a_, b_ in (("classical", "metal", "reggae"), ("jazz", "hiphop", "disco")):
        key = next((k for k in ex if k.startswith(song)), None)
        if key is None:
            continue
        path, fig, md = engine.run(ex[key], 0, 10, a_, b_, 0.5, engine.default_keep, engine.default_boost,
                                   "none", 0.0, "none", 0.0, 0, "auto", False, 0, 0, 7)
        a, asr = sf.read(path)
        mp3 = out_dir / f"{song}__{a_}+{b_}.mp3"
        sf.write(str(mp3), a, asr, format="MP3")
        rows.append({"song": song, "target": f"{a_} + {b_}", "file": mp3.name, "info": md})
    (out_dir / "gallery.json").write_text(json.dumps(rows, indent=2))
    html = ["<!doctype html><meta charset='utf-8'><title>GenreKnob gallery</title>",
            "<style>body{font-family:system-ui;margin:24px;max-width:980px}"
            "td{padding:6px 10px;border-bottom:1px solid #ddd}audio{width:260px}</style>",
            "<h1>GenreKnob offline gallery</h1><table><tr><th>song</th><th>target</th><th>audio</th></tr>"]
    for song in CACHE_SONGS:
        if (out_dir / f"{song}__original.mp3").exists():
            html.append(f"<tr><td>{song}</td><td><b>original</b></td><td><audio controls src='{song}__original.mp3'></audio></td></tr>")
        for r in [r for r in rows if r["song"] == song]:
            html.append(f"<tr><td>{song}</td><td>{r['target']}</td><td><audio controls src='{r['file']}'></audio></td></tr>")
    html.append("</table>")
    (out_dir / "index.html").write_text("\n".join(html))
    print("gallery written:", len(rows), "clips")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="also create a public gradio.live link (off by default)")
    ap.add_argument("--host", default=os.environ.get("GK_HOST", "0.0.0.0"),
                    help="interface to listen on; 127.0.0.1 = this machine only")
    ap.add_argument("--port", type=int, default=int(os.environ.get("GK_PORT", 7860)))
    ap.add_argument("--auth", default=os.environ.get("GK_AUTH"),
                    help="user:password asked before the page opens (use it with --share)")
    ap.add_argument("--build-only", action="store_true", help="construct the UI and exit (smoke test)")
    ap.add_argument("--cache", action="store_true", help="render the offline gallery and exit")
    a = ap.parse_args()
    engine = None if a.build_only else Engine()
    if a.cache:
        render_cache(engine)
        return
    demo = build_ui(engine)
    if a.build_only:
        print("UI built OK")
        return
    import inspect
    kw = dict(share=a.share, server_name=a.host, server_port=a.port)
    if "allowed_paths" in inspect.signature(demo.launch).parameters:   # offline gallery files
        kw["allowed_paths"] = [str(RESULTS / "demo_cache")]
    if a.auth:
        user, _, pw = a.auth.partition(":")
        kw["auth"] = (user, pw)
    demo.queue().launch(**kw)


if __name__ == "__main__":
    main()
