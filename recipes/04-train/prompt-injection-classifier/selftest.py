"""The pure parts, on known inputs: the metrics, the planting, the obfuscations, the frozen hash.

python selftest.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data
import metrics

HERE = Path(__file__).resolve().parent


def main() -> None:
    # AUROC on a hand-computed case: pairs (0.1,0.35) (0.1,0.8) (0.4,0.35) (0.4,0.8) -> 3 of 4 correct
    assert abs(metrics.auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) - 0.75) < 1e-9
    # a tie between a positive and a negative counts half
    assert abs(metrics.auroc([0, 1], [0.5, 0.5]) - 0.5) < 1e-9
    # recall at 1% FPR with 100 negatives allows one false positive
    y = [0] * 100 + [1] * 10
    s = [i / 198 for i in range(100)] + [0.49, 0.495, 0.51, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    assert abs(metrics.recall_at_fpr(y, s) - 0.9) < 1e-9, metrics.recall_at_fpr(y, s)
    thr = metrics.choose_threshold(y, s)
    assert abs(metrics.at_threshold(y, s, thr)["fpr"] - 0.01) < 1e-9
    # a metric that cannot fail must fail here: an all-negative slice has no AUROC
    rng = random.Random(0)
    r = metrics.report([0] * 20, [rng.random() for _ in range(20)], 0.5, n_boot=10)
    assert "auroc" not in r and "fpr" in r
    # the interval contains the point estimate
    r = metrics.report(y, s, thr, n_boot=200)
    assert r["auroc_ci95"][0] <= r["auroc"] <= r["auroc_ci95"][1]
    # planting never goes before the first segment and always keeps every segment
    rng = random.Random(0)
    for name, fn in data.CARRIERS.items():
        segs = fn(rng)
        out = data.plant(segs, "PAYLOAD", rng)
        lines = out.split("\n")
        assert lines[0] == segs[0].split("\n")[0], name
        assert "PAYLOAD" in out and len(lines) == sum(s.count("\n") + 1 for s in segs) + 1, name
    # obfuscations keep the label by construction: each kind is a total function of the payload
    kinds = {data.obfuscate("please do the thing", random.Random(i))[1] for i in range(40)}
    assert kinds == {"base64", "spacing", "homoglyph", "leet"}, kinds
    # the frozen test matches its pinned hash
    rows = data.read_jsonl(HERE / "test.jsonl")
    assert data.sha256_rows(rows) == (HERE / "test.sha256").read_text().strip()
    assert {r["slice"] for r in rows} >= {
        "deepset",
        "notinject",
        "indirect_in_dist",
        "indirect_heldout_family",
        "indirect_heldout_carrier",
    }
    # held-out families and carriers are the ones the training draw excludes
    assert not set(data.HELDOUT_FAMILIES) & {
        "injecagent",
        "bipia_task",
        "bipia_obfusc",
        "bipia_harm",
    }
    assert set(data.HELDOUT_CARRIERS) <= set(data.CARRIERS)
    train_carriers = {r["carrier"] for r in rows if r["slice"] == "indirect_in_dist"}
    assert not train_carriers & set(data.HELDOUT_CARRIERS)
    print(
        f"selftest ok: metrics, planting over {len(data.CARRIERS)} carriers, "
        f"4 obfuscations, frozen test {len(rows)} rows"
    )


if __name__ == "__main__":
    main()
