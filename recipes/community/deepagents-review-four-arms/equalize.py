"""Cut each experiment's two training sets to the same size (PREREGISTRATION.md).

N is the smaller kept set; the larger is cut to N by `random.Random(0).sample`
over its trace ids sorted ascending. The cut happens before smithtune's
80/10/10 split. Writes keep/<arm>.json, the trace ids `export_sft.py --keep`
reads, and keep/summary.json with both sizes before and after.

    python equalize.py a     # council-v2 against while_pick.json
    python equalize.py b     # council-v2 + council-b-random against while_pick_b.json
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
ST = HERE / ".cache" / "st"
KEEP = HERE / "keep"
SEED = 0


def council_kept(pull: str) -> set[str]:
    """Trace ids of the trajectories a smithtune council kept."""
    d = ST / pull
    kept = {
        r["trajectory_id"]
        for r in map(json.loads, (d / "labels.jsonl").read_text(encoding="utf-8").splitlines())
        if r["keep"] == 1
    }
    out = set()
    for f in (d / "conversations").glob("*.json"):
        ex = json.loads(f.read_text(encoding="utf-8"))["example"]
        if ex["id"] in kept:
            out.add(ex["metadata"]["source_trace_id"])
    if len(out) != len(kept):
        raise SystemExit(f"{pull}: {len(kept)} kept labels but {len(out)} conversations")
    return out


def while_kept(pick: str) -> set[str]:
    return set(json.loads((HERE / pick).read_text(encoding="utf-8"))["run_ids"])


def cut(ids: set[str], n: int) -> list[str]:
    ordered = sorted(ids)
    return ordered if len(ordered) == n else sorted(random.Random(SEED).sample(ordered, n))


def main() -> int:
    exp = sys.argv[1] if len(sys.argv) > 1 else "a"
    if exp == "a":
        without, with_ = council_kept("council-v2"), while_kept("while_pick.json")
    elif exp == "b":
        without = council_kept("council-v2") | council_kept("council-b-random")
        with_ = while_kept("while_pick_b.json")
    else:
        raise SystemExit("usage: python equalize.py a|b")
    n = min(len(without), len(with_))
    KEEP.mkdir(exist_ok=True)
    sizes = {}
    for arm, ids in ((f"{exp}-without", without), (f"{exp}-with", with_)):
        kept = cut(ids, n)
        (KEEP / f"{arm}.json").write_text(json.dumps(kept, indent=1) + "\n")
        sizes[arm] = {"kept": len(ids), "trained": len(kept)}
    summary_path = KEEP / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary[exp] = {"n": n, "seed": SEED, **sizes}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary[exp], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
