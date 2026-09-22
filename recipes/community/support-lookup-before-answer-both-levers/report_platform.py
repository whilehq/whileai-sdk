"""Put the four arms on the platform so a teammate reads them in one look.

    python report_platform.py            # needs WHILEAI_API_KEY
    python report_platform.py --dry-run  # offline, no key: prints what it would post

One question, one frozen test named by its content, four runs each with the
five lines (Changed, Moved, Why, Learned, Reproduce), one figure, and a
``readback`` that has to come back empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from whileai.config import provenance
from whileai.platform import (
    Behavior,
    Data,
    Harness,
    Judge,
    Optimizer,
    Provenance,
    RunRecord,
    track,
)

HERE = Path(__file__).resolve().parent
BASE = "Qwen/Qwen3-4B"

ARMS = {
    "deployed prompt, untrained": ("neither", "00_deployed", "eval"),
    "skills text, untrained": ("harness", None, "eval"),
    "trained under the deployed prompt": ("weights", "00_deployed", "SFT"),
    "trained under the skills text": ("both", None, "SFT"),
}

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
            if 0 < e["score"] < 1:
                out.append(f"{v} {e['behavior']}: {e['score']} reads as a fraction; post points")
    return out


def main() -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    res = json.loads((HERE / "results.json").read_text())
    cells = res["cells"]
    searched = res["searched_harness"]
    n_hold = cells["neither"]["n"]
    holdout = (
        json.loads((HERE / "out" / "holdout.json").read_text())
        if (HERE / "out" / "holdout.json").exists()
        else []
    )
    asks = [t["prompt"] for t in holdout] or [f"task-{i}" for i in range(n_hold)]
    TEST = "t-" + hashlib.sha256("\n".join(asks).encode()).hexdigest()[:8]

    if a.dry_run:
        print(f"would post agent 'support-lookup', test {TEST}, {len(ARMS)} runs")
        for words, (cell, _h, method) in ARMS.items():
            c = cells[cell]
            print(
                f"  {words:36s} {method:5s} "
                f"right_first_action {round(c['right_first_action'] * 100)} points"
            )
        return 0

    tracked = track("support-lookup", model=BASE)
    tracked.experiment(
        question="Does the support agent look the customer up before answering, and is that "
        "the harness or the weights?",
        hypothesis="A skills text saying when to call a tool moves it further than SFT on the "
        "good traces does, and training under that text adds little on top.",
        method="Same 320 held-out decision points, same program grader. Harness: the deployed "
        "policy against a skills text. Training: LoRA SFT on 508 reference first "
        "actions, one arm under each harness.",
        measure="right_first_action on the frozen holdout, points out of 100, 95% interval, "
        "against a noise floor from three base re-runs.",
        decide="Take the arm whose interval clears the served one and the floor, and only if "
        "stays_in_scope does not regress.",
    )
    tracked.behavior(
        Behavior(
            name="right_first_action",
            test_version=TEST,
            n=n_hold,
            judge=Judge(name="tool-call match as a program", agreement=1.0, human_n=0),
            noise_floor=res["noise_band"],
            contamination=res["decontamination"]["n_contaminated"],
            reward_is_judge=False,
            rubric="On an ask about the customer's own account, call the tool the reference "
            "agent called. On an ask the reference refused, call nothing.",
            description="Looks it up before answering, and still stays in scope.",
        )
    )

    posted = {}
    for words, (cell, harness_label, method) in ARMS.items():
        c = cells[cell]
        label = harness_label or searched
        run = tracked.run(
            words,
            method=method,
            base=BASE if method != "eval" else None,
            harness=Harness(label=f"{label}@{BASE}", model=BASE),
            targets=["right_first_action"],
            trained_on=["tau2-simulated reference first actions"] if method != "eval" else None,
            record=RunRecord(
                data=Data(
                    train="tau2-simulated train split" if method != "eval" else None,
                    n_train=res["n_train"] if method != "eval" else None,
                    holdout=TEST,
                    n_holdout=n_hold,
                ),
                optimizer=Optimizer(lr=1e-4, seed=17) if method != "eval" else Optimizer(seed=101),
                provenance=Provenance(pins=res["pins"]),
            ),
        )
        run.score(
            "right_first_action",
            round(c["right_first_action"] * 100),
            ci=round(c.get("ci_halfwidth", 0.0) * 100),
            n=n_hold,
        )
        posted[cell] = run

    d = res["deltas"]
    base_pts = round(cells["neither"]["right_first_action"] * 100)
    for cell, run in posted.items():
        pts = round(cells[cell]["right_first_action"] * 100)
        twin = round(cells[cell]["stays_in_scope"] * 100)
        run.note(
            f"Changed: {'the prompt only' if cell == 'harness' else 'LoRA SFT on 508 reference first actions' if cell != 'neither' else 'nothing; this is the served agent'}"
            f", harness {searched if cell in ('harness', 'both') else '00_deployed'}.\n"
            f"Moved: {base_pts} to {pts} points on {n_hold} held-out decision points; "
            f"stays_in_scope {twin}.\n"
            f"Why: {res['why'].get(cell, 'see the recipe README')}\n"
            f"Learned: {res['learned_one_line']}\n"
            f"Reproduce: modal run --detach support_modal.py, then python run.py --analyse"
        )

    tracked.figure(
        "levers",
        {
            "data": [
                {
                    "type": "bar",
                    "x": [w for w in ARMS],
                    "y": [round(cells[c]["right_first_action"] * 100) for c, *_ in ARMS.values()],
                }
            ],
            "layout": {
                "title": "right_first_action, points out of 100",
                "yaxis": {"range": [0, 100]},
            },
        },
        caption=f"The searched harness is {searched}; "
        f"method vs baseline {d['both vs weights  (method vs baseline)']['delta']:+.3f}.",
        run=posted["both"],
    )
    for r in posted.values():
        r.finish(say=False)

    problems = readback(tracked)
    for problem in problems:
        print("fix:", problem)
    print("readback clean" if not problems else f"{len(problems)} problems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
