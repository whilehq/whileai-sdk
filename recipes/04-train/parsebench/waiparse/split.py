"""Document-level dev/test split of ParseBench.

ParseBench ships no train split. We never train on it. The harness is tuned on
`dev` (one source report in five, chosen by hash) and the headline number is
`test` (the other four in five) plus `full` for leaderboard parity. Pages cut from
the same source report (`<stem>_p<N>.pdf`) always land on the same side, so a
report's layout and fonts cannot leak from dev into test.
"""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

DEV_EVERY = 5  # 1 in 5 source reports -> dev


def source_doc(pdf: str) -> str:
    stem = Path(pdf).stem
    return re.sub(r"_p(age)?\d+$", "", stem)


def side(pdf: str) -> str:
    h = int(hashlib.sha1(source_doc(pdf).encode()).hexdigest(), 16)
    return "dev" if h % DEV_EVERY == 0 else "test"


def materialize(full_dir: str, out_dir: str, split: str) -> None:
    """Write a ParseBench-shaped dir holding only `split`'s rows and documents."""
    full, out = Path(full_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for jsonl in full.glob("*.jsonl"):
        kept = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip() and side(json.loads(line)["pdf"]) == split:
                kept.append(line)
                src = full / json.loads(line)["pdf"]
                dst = out / json.loads(line)["pdf"]
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
        (out / jsonl.name).write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"{split} {jsonl.name}: {len(kept)} rules")
