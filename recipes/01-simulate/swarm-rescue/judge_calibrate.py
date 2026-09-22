"""Pre-flight for a judge as the swarm's fitness: does its score predict the pass?

A verifier's partial credit turned out to be a cliff (calibrate.py). A model
judge can grade closeness on a scale by construction; this asks whether that
scale means anything. It shows a judge the problem and a saved program and
asks for the probability, 0 to 100, that the program passes every hidden
test. Then P(hidden pass | judge bucket) over the same samples. The judge
never sees a test result. The saved programs come from a run of run.py.

Run: python judge_calibrate.py --out out-27b --limit 900     # base samples of the all-fail tasks
     python judge_calibrate.py --out out-27b --reuse         # the curve again
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R

JUDGE_SYSTEM = (
    "You are a strict competitive-programming judge. You will see a problem and a "
    "candidate Python program. Estimate the probability that the program passes every "
    "hidden test, including edge cases and time limits. Consider the algorithm, its "
    "complexity against the constraints, input parsing, output format and edge cases. "
    "Reply with one line: `PROBABILITY: <integer 0-100>` and nothing else."
)
PROB = re.compile(r"PROBABILITY:\s*(\d{1,3})")


def judge_messages(task: R.Task, code: str) -> list[dict]:
    user = f"{task.name}\n\n{task.statement}\n\nCandidate program:\n```python\n{code}\n```"
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="out-27b")
    p.add_argument("--limit", type=int, default=900, help="samples to judge")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL") or R.HOSTED_URL)
    p.add_argument("--model", default=os.environ.get("JUDGE_MODEL") or R.HOSTED_MODEL)
    p.add_argument("--reuse", action="store_true")
    args = p.parse_args(argv)
    out_dir = R.HERE / args.out
    path = out_dir / "judge.jsonl"
    if not (args.reuse and path.exists()):
        base = json.loads((out_dir / "base.json").read_text(encoding="utf-8"))
        tasks = {t.id: t for t in R.load_tasks(None, 0)}
        all_fail = [t for t, b in base.items() if not any(s["grade"]["correct"] for s in b)]
        items = []
        for t in all_fail:
            for s in base[t]:
                items.append((t, s))
            for arm in R.ARMS:
                arm_path = out_dir / f"arm-{arm}.json"
                if arm_path.exists():
                    res = json.loads(arm_path.read_text(encoding="utf-8"))
                    for s in res.get(t, {}).get("samples", []):
                        items.append((t, s))
        random.Random(args.seed).shuffle(items)
        items = items[: args.limit]
        # Every sample carries its hidden-test verdict already: the judge
        # is scored against that, never shown it.
        model = R.Model(
            args.base_url,
            args.model,
            R.resolve_api_key() if args.base_url == R.HOSTED_URL else None,
            max_tokens=24,
        )
        print(
            f"{len(items)} samples from {len(all_fail)} all-fail tasks, judge {model.model}",
            file=sys.stderr,
        )

        def one(item: tuple[str, dict]) -> dict:
            t, s = item
            code = R.extract_code(s["text"])[:6000]
            reply = model.chat(judge_messages(tasks[t], code), seed=args.seed)
            m = PROB.search(reply)
            return {
                "task_id": t,
                "judge": int(m.group(1)) if m else None,
                "fitness": s["grade"]["fitness"],
                "correct": bool(s["grade"]["correct"]),
                "reply": reply[:80],
            }

        rows = []
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for n, row in enumerate(ex.map(one, items), 1):
                rows.append(row)
                if n % 100 == 0:
                    print(f"  {n}/{len(items)}", file=sys.stderr)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    parsed = [r for r in rows if r["judge"] is not None]
    print(
        f"\n{len(rows)} judged, {len(parsed)} parsed; pass rate {sum(r['correct'] for r in rows) / len(rows):.3f}"
    )
    b: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in parsed:
        j = r["judge"]
        k = (
            "0-9"
            if j < 10
            else "10-29"
            if j < 30
            else "30-49"
            if j < 50
            else "50-69"
            if j < 70
            else "70-89"
            if j < 90
            else "90-100"
        )
        b[k][1] += 1
        b[k][0] += int(r["correct"])
    order = ["0-9", "10-29", "30-49", "50-69", "70-89", "90-100"]
    print("P(hidden pass | judge probability bucket)")
    for k in order:
        if k in b:
            c, n = b[k]
            print(f"  judge {k:>7}: {c:4d}/{n:<5d} = {c / n:.3f}")
    # rank quality: AUC of the judge score for the pass
    pos = [r["judge"] for r in parsed if r["correct"]]
    neg = [r["judge"] for r in parsed if not r["correct"]]
    if pos and neg:
        wins = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg)
        print(
            f"AUC(judge -> hidden pass) = {wins / (len(pos) * len(neg)):.3f} over {len(pos)} passes, {len(neg)} fails"
        )
        vpos = [r["fitness"] for r in parsed if r["correct"]]
        vneg = [r["fitness"] for r in parsed if not r["correct"]]
        wins = sum((x > y) + 0.5 * (x == y) for x in vpos for y in vneg)
        print(
            f"AUC(visible-test fitness -> hidden pass) = {wins / (len(vpos) * len(vneg)):.3f}, for comparison"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
