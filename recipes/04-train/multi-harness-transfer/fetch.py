"""Copy scored cells from the Modal volume to out/<model>/cells (works around `modal volume get` on Windows).

    python fetch.py base oc-rl mh-rl
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

HERE = Path(__file__).parent
vol = modal.Volume.from_name("multi-harness-transfer")


def fetch(model: str, tag: str = "phase0") -> int:
    dest = HERE / "out" / model / "cells"
    dest.mkdir(parents=True, exist_ok=True)
    entries = [e for e in vol.listdir(f"{tag}/{model}/cells") if e.path.endswith(".json")]

    def one(entry):
        target = dest / Path(entry.path).name
        target.write_bytes(b"".join(vol.read_file(entry.path)))

    with ThreadPoolExecutor(32) as pool:
        list(pool.map(one, entries))
    return len(entries)


if __name__ == "__main__":
    for m in sys.argv[1:]:
        print(m, fetch(m), flush=True)
