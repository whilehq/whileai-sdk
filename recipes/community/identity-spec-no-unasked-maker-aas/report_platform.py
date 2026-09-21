"""Post this run to the platform the way skills/manage-experiments asks.

One question, one frozen test per behaviour, one run per arm with the five
note lines, one figure, and a readback that has to come back empty.

    python report_platform.py          # needs WHILEAI_API_KEY
    python report_platform.py --dry    # prints what it would post, posts nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

HERE = Path(__file__).parent
OUT = HERE / "out"
AGENT = "identity-spec-agent"
BASE_MODEL = "Qwen/Qwen3-1.7B"
PAPER = "arXiv:2607.07023"

SETTING = re.compile(r"(lr\d|\de-0\d|-s\d+\b|_s\d+$|_seed\d+|^h-[0-9a-f]{12}$|^v\d+$)")


def readback(tracked) -> list[str]:
    """What a teammate opening the page could not read. Empty means clean."""
    out = []
    if tracked.experiment() is None:
        out.append("no question posted: tracked.experiment(question=...)")
    for b in tracked.behaviors():
        if SETTING.search(b.name):
            out.append(f"behavior {b.name!r} names a seed or setting; put it in Optimizer(seed=)")
        if not (b.test_version or "").startswith("t-"):
            out.append(f"behavior {b.name!r}: test_version is not the asks' hash")
    pictured = {f.run for f in tracked.figures()}
    for r in tracked.runs():
        v, note = r["version"], r.get("notes") or ""
        trained = r.get("method") not in (None, "none", "eval")
        if SETTING.search(v):
            out.append(f"version {v!r} encodes settings; say the arm in words, numbers in record")
        if not ((r.get("record") or {}).get("data") or {}):
            out.append(f"run {v!r}: no data posted; RunRecord(data=Data(...))")
        for word in ("Changed", "Moved", "Why", "Learned", "Reproduce"):
            if trained and f"{word}:" not in note:
                out.append(f"run {v!r}: note has no '{word}:' line; run.note(...)")
        if trained and r["id"] not in pictured:
            out.append(f"run {v!r}: no picture; tracked.figure(name, fig, run=run)")
        for e in r.get("evals") or []:
            if e["score"] <= 1:
                out.append(f"{v} {e['behavior']}: {e['score']} reads as a fraction; post points")
    return out


def _pts(x: float) -> float:
    """Points out of 100, never a fraction: the platform counts points."""
    return round(100.0 * float(x), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    from whileai.platform import (
        Behavior,
        Data,
        Judge,
        Optimizer,
        Provenance,
        RunRecord,
        track,
    )

    results = json.loads((HERE / "results.json").read_text())
    prep = results["prep"]
    ada = results["ada"]["arms"]
    leak_holdout = [json.loads(x) for x in (OUT / "holdout_leak.jsonl").read_text().splitlines()]
    TEST = (
        "t-" + hashlib.sha256("\n".join(r["prompt"] for r in leak_holdout).encode()).hexdigest()[:8]
    )

    arms = results["arms"]
    mvb = results["method_vs_baseline"]["leak"]

    def target(arm: str, split: str = "leak") -> tuple[float, float]:
        """The arm's own score and half-width, read off the pass_at_1 metric."""
        m = arms[arm][split]["metrics"]["pass_at_1"]
        lo, hi = m["ci95"]
        return float(m["mean_b"]), float(hi - lo) / 2.0

    if args.dry:
        print(json.dumps({"test": TEST, "ada": ada, "method_vs_baseline": mvb}, indent=2)[:2000])
        return

    tracked = track(AGENT, model=BASE_MODEL)
    tracked.experiment(
        question=(
            "Does capping the identity share of an SFT selector's token budget stop the "
            "agent volunteering its maker, without costing it the identity answer?"
        ),
        hypothesis=(
            "Loss-based selection over-picks identity rows because they are the rows the "
            "base is most surprised by, and the agent then introduces itself unprompted."
        ),
        method=(
            f"{PAPER}. Three selectors, one token budget, one base, one holdout. "
            "Every grade is a program; no judge."
        ),
        measure=(
            "no_unasked_maker on 500 held-out ordinary requests, points out of 100, "
            "95% interval, against a noise floor from three base passes."
        ),
        decide=(
            "Promote the arm whose interval clears both the baseline arm and the floor, "
            "provided the identity answer does not regress."
        ),
    )

    floor = results["noise_floor"]["leak"]["target"]["run_std"]
    tracked.behavior(
        Behavior(
            name="no_unasked_maker",
            test_version=TEST,
            n=len(leak_holdout),
            judge=Judge(
                name="identity-claim detector as a program",
                agreement=1.0,
                human_n=prep["detector_checked_on"],
            ),
            noise_floor=_pts(floor),
            contamination=prep["dropped_by_decontaminate"],
            reward_is_judge=False,
            description=(
                "The reply does not volunteer the agent's name or maker when the user "
                "did not ask. Graded by a program, not a model."
            ),
        )
    )

    labels = {
        "random": "random selection",
        "loss": "highest-loss rows first",
        # Run names cap at 40 characters, so the cap itself is said in the note.
        "aas": "highest-loss, identity share capped",
    }
    posted = {}
    for arm in ("random", "loss", "aas"):
        score, hw = target(arm)
        ident, ident_hw = target(arm, "identity")
        run = tracked.run(
            labels[arm],
            method="SFT",
            base=BASE_MODEL,
            targets=["no_unasked_maker"],
            trained_on=["while-ai/identity-behavior, spec-corrected"],
            record=RunRecord(
                data=Data(
                    train="while-ai/identity-behavior train, spec-corrected",
                    n_train=ada[arm]["rows"],
                    holdout=TEST,
                    n_holdout=len(leak_holdout),
                    decontaminated_dropped=prep["dropped_by_decontaminate"],
                ),
                optimizer=Optimizer(lr=1e-4, seed=0, lora_rank=16, temperature=0.7, top_p=0.9),
                provenance=Provenance(
                    pins={"trl": "0.19.1", "transformers": "4.54.0", "peft": "0.16.0"},
                    paper=PAPER,
                    recipe="recipes/community/identity-spec-no-unasked-maker-aas",
                ),
            ),
        )
        run.score("no_unasked_maker", _pts(score), ci=_pts(hw), n=len(leak_holdout))
        run.note(
            f"Changed: selector only. {labels[arm]}, same {ada[arm]['tokens']:,} training "
            f"tokens as every other arm.\n"
            f"Moved: no_unasked_maker {_pts(score)} points (+/-{_pts(hw)}); the identity "
            f"answer sits at {_pts(ident)} points (+/-{_pts(ident_hw)}).\n"
            f"Why: this selector spent {ada[arm]['identity_token_share']:.1%} of its budget "
            f"on identity rows against {ada[arm]['pool_identity_token_share']:.1%} in the pool.\n"
            f"Learned: the selector's attribute mixture, readable before any GPU, is what "
            f"moves the behaviour ({PAPER}).\n"
            f"Reproduce: python run.py prep && modal run --detach train_modal.py && "
            f"modal run --detach eval_modal.py && python run.py analyse"
        )
        posted[arm] = (run, _pts(score), _pts(hw))

    # A picture per trained run: readback wants every run to carry one, and a
    # single figure hung off one arm leaves the other two unreadable.
    for arm, (run, _, _) in posted.items():
        tracked.figure(
            f"selector-{arm}",
            {
                "data": [
                    {
                        "type": "bar",
                        "x": [labels[a] for a in posted],
                        "y": [posted[a][1] for a in posted],
                        "error_y": {"type": "data", "array": [posted[a][2] for a in posted]},
                    }
                ],
                "layout": {
                    "title": "no_unasked_maker, points out of 100",
                    "yaxis": {"range": [0, 100]},
                },
            },
            caption=(
                f"{labels[arm]}: {ada[arm]['identity_token_share']:.1%} of an equal token "
                f"budget on identity rows (pool {ada[arm]['pool_identity_token_share']:.1%}). "
                "Every arm is at ceiling on this behaviour."
            ),
            run=run,
        )

    tracked.figure(
        "selectors",
        {
            "data": [
                {
                    "type": "bar",
                    "x": [labels[a] for a in posted],
                    "y": [posted[a][1] for a in posted],
                    "error_y": {
                        "type": "data",
                        "array": [posted[a][2] for a in posted],
                    },
                }
            ],
            "layout": {
                "title": "no_unasked_maker, points out of 100",
                "yaxis": {"range": [0, 100]},
            },
        },
        caption=(
            "Same base, same token budget, same holdout. Only the selector differs. "
            f"Method minus baseline {_pts(mvb['metrics']['pass_at_1']['delta'])} points."
        ),
        run=posted["aas"][0],
    )
    for run, _, _ in posted.values():
        run.finish(say=False)

    problems = readback(tracked)
    print("\nreadback:", "clean" if not problems else f"{len(problems)} problems")
    for p in problems:
        print(" -", p)
    print(tracked.dashboard())


if __name__ == "__main__":
    main()
