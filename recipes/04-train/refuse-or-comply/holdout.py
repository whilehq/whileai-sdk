"""The frozen holdout: 120 attacked and 120 clean scenarios, pinned by content.

The world and the grader are the sibling recipe's
(`recipes/04-train/resist-planted-instruction`); this file only draws a new
holdout from them with an order-id block no training row can name, halves it
so the clean half is as large as the attacked half (the over-refusal control
is sized like the treatment), and writes the sha256 that every scorer checks
before it reads a number (Lambert 2025, chapter *Evaluation*: a held-out set
is kept apart and named).

    python holdout.py            # prints the composition and the pin
    python holdout.py --write    # writes holdout.sha256 next to this file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIBLING = HERE.parent / "resist-planted-instruction"
sys.path.insert(0, str(SIBLING))

from world import build_world

# Disjoint from the sibling's train (40000), holdout (58000) and probe (71000)
# blocks, and from the gate's probe world below. Change any block and the
# pin moves, which is the point.
HOLDOUT_WORLD = {"n": 300, "seed": 2626, "prefix": 64000, "attack_share": 0.5}
PER_HALF = 120

#: Where the gate's window and threshold were chosen. Never scored.
PROBE_WORLD = {"n": 300, "seed": 4242, "prefix": 77000, "attack_share": 0.5}

PIN_PATH = HERE / "holdout.sha256"


def build_holdout() -> dict:
    """{"db", "scenarios"} restricted to the 240 pinned scenarios, in order."""
    world = build_world(**HOLDOUT_WORLD)
    attacked = [sid for sid, sc in world["scenarios"].items() if sc["attack"]][:PER_HALF]
    clean = [sid for sid, sc in world["scenarios"].items() if not sc["attack"]][:PER_HALF]
    if len(attacked) < PER_HALF or len(clean) < PER_HALF:
        raise SystemExit(
            f"the world drew {len(attacked)} attacked and {len(clean)} clean scenarios; "
            f"raise HOLDOUT_WORLD['n'] above {HOLDOUT_WORLD['n']}"
        )
    keep = sorted(attacked + clean)
    scenarios = {sid: world["scenarios"][sid] for sid in keep}
    return {"db": world["db"], "scenarios": scenarios}


def tasks_for(world: dict) -> list[dict]:
    """One pinned task per scenario. The opener names the order and the
    grader reads it back out of the prompt."""
    return [
        {
            "prompt": sc["opener"],
            "scenario_id": sid,
            "half": "attack" if sc["attack"] else "clean",
        }
        for sid, sc in world["scenarios"].items()
    ]


def pin_rows(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


def check_pin(rows: list[dict], pin_path: Path = PIN_PATH) -> str:
    """The rows' digest, or a SystemExit naming the mismatch."""
    digest = pin_rows(rows)
    pinned = pin_path.read_text(encoding="utf-8").strip()
    if digest != pinned:
        raise SystemExit(
            f"holdout: sha256 {digest[:12]} does not match the pin {pinned[:12]} in "
            f"{pin_path.name}; the eval world changed and every number scored on it is void"
        )
    return digest


def composition(world: dict) -> dict:
    from collections import Counter

    scs = list(world["scenarios"].values())
    return {
        "n": len(scs),
        "attack": sum(1 for s in scs if s["attack"]),
        "clean": sum(1 for s in scs if not s["attack"]),
        "task_kind": dict(Counter(s["task"]["kind"] for s in scs)),
        "attack_kind": dict(Counter(s["attack"]["kind"] for s in scs if s["attack"])),
        "attack_frame": dict(Counter(s["attack"]["frame"] for s in scs if s["attack"])),
        "attack_channel": dict(Counter(s["attack"]["channel"] for s in scs if s["attack"])),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--write", action="store_true", help="write holdout.sha256")
    args = p.parse_args(argv)
    world = build_holdout()
    tasks = tasks_for(world)
    digest = pin_rows(tasks)
    print(json.dumps(composition(world), indent=1))
    print(f"sha256 over {len(tasks)} task rows: {digest}  (test_version t-{digest[:8]})")
    if args.write:
        PIN_PATH.write_text(digest + "\n", encoding="utf-8")
        print("wrote", PIN_PATH)
    elif PIN_PATH.exists():
        check_pin(tasks)
        print("matches the pin in", PIN_PATH.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
