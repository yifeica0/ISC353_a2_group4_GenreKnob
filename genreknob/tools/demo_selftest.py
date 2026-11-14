"""Drive the running GenreKnob demo through its public API, exactly as the browser would.

Usage (with the app already listening on port 7860):
    $ACE_PY tools/demo_selftest.py http://127.0.0.1:7860 <example.wav> <out_dir>
Writes a hummed-style test melody, runs five requests (plain swap, fusion,
instrument + mood, repaint, hum input) and prints timing and the info text.
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def hum(path, sr=22050):
    """A 10-s sung-like melody: vibrato sine with harmonics (stands in for a recorded hum)."""
    notes = [60, 62, 64, 65, 67, 65, 64, 62, 60, 64, 67, 72, 71, 67, 64, 60]
    t_note = 10.0 / len(notes)
    y = []
    for n in notes:
        f0 = 440 * 2 ** ((n - 69) / 12)
        t = np.arange(int(sr * t_note)) / sr
        f = f0 * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * t))
        ph = 2 * np.pi * np.cumsum(f) / sr
        env = np.minimum(1, t / 0.04) * np.minimum(1, (t_note - t) / 0.06)
        y.append(env * (0.6 * np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)))
    sf.write(path, 0.4 * np.concatenate(y), sr)


def main(url, example, out_dir):
    from gradio_client import Client, handle_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hum_path = str(out / "test_hum.wav")
    hum(hum_path)
    c = Client(url, verbose=False)
    # audio, start, length, target, mix_genre, mix, keep, boost, inst, inst_amt, mood, mood_amt,
    # bpm, key, repaint_on, rp_start, rp_end, seed
    cases = {
        "swap_rock": [example, 0, 10, "rock", "none", 0.0, 0.3, 1.0, "none", 0.0, "none", 0.0, 0, "auto", False, 4, 8, 7],
        "fusion_hiphop_disco": [example, 0, 10, "hiphop", "disco", 0.5, 0.3, 1.0, "none", 0.0, "none", 0.0, 0, "auto", False, 4, 8, 7],
        "sax_happy": [example, 0, 10, "jazz", "none", 0.0, 0.3, 1.0, "saxophone", 1.5, "happy", 1.5, 0, "auto", False, 4, 8, 7],
        "repaint_4_8": [example, 0, 10, "metal", "none", 0.0, 0.3, 1.0, "none", 0.0, "none", 0.0, 0, "auto", True, 4, 8, 7],
        "hum_to_reggae": [hum_path, 0, 10, "reggae", "none", 0.0, 0.3, 1.0, "none", 0.0, "none", 0.0, 0, "auto", False, 4, 8, 7],
    }
    ok = 0
    for name, args in cases.items():
        args = list(args)
        args[0] = handle_file(args[0])
        t0 = time.time()
        try:
            res = c.predict(*args, api_name="/run")
            wav = res[0]
            y, sr = sf.read(wav)
            dst = out / f"{name}.wav"
            sf.write(str(dst), y, sr)
            info = res[2] if len(res) > 2 else ""
            print(f"[selftest] {name}: OK {time.time() - t0:.1f}s, {len(y) / sr:.1f}s audio -> {dst.name}")
            print("           " + str(info).replace("\n", " ")[:400])
            ok += 1
        except Exception as e:
            print(f"[selftest] {name}: FAILED {e!r}"[:600])
    print(f"[selftest] {ok}/{len(cases)} requests succeeded")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
