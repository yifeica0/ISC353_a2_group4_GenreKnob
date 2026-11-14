"""Check the example dropdown of a running GenreKnob demo through its API.

Usage: $ACE_PY tools/demo_examples_test.py http://127.0.0.1:7861 <out_dir>
Selects every example song (as the dropdown does), then swaps the first one to rock.
"""
import sys
import time
from pathlib import Path

import soundfile as sf


def main(url, out_dir):
    from gradio_client import Client, handle_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = Client(url, verbose=False)
    keys = [k for k in c.view_api(return_format="dict", print_info=False)["named_endpoints"]
            ["/load_example"]["parameters"][0]["type"].get("enum", []) if k != "(none)"]
    print(f"[examples] {len(keys)} examples: {keys}")
    ok, first = 0, None
    for k in keys:
        try:
            res = c.predict(k, "rock", 7, 0.3, 2.0, api_name="/load_example")
            f = res[0] if isinstance(res, (list, tuple)) else res
            y, sr = sf.read(f)
            extra = f", target {res[1]}, seed {res[2]}" if isinstance(res, (list, tuple)) else ""
            print(f"[examples] {k}: OK, {len(y) / sr:.1f} s{extra}")
            first = first or f
            ok += 1
        except Exception as e:
            print(f"[examples] {k}: FAILED {e!r}"[:400])
    if first:
        t0 = time.time()
        res = c.predict(handle_file(first), 0, 10, "rock", "none", 0.0, 0.3, 2.0, "none", 0.0, "none", 0.0,
                        0, "auto", False, 4, 8, 7, api_name="/run")
        y, sr = sf.read(res[0])
        sf.write(str(out / "example_to_rock.wav"), y, sr)
        print(f"[examples] swap of first example: OK {time.time() - t0:.1f}s")
        print("           " + str(res[2]).replace("\n", " ")[:300])
    print(f"[examples] {ok}/{len(keys)} examples loaded")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
