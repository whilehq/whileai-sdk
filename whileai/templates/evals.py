"""The files ``wai init-evals`` writes.

Each constant is the text of one generated file. The tokens spelled
``__WAI_...__`` are replaced by :mod:`whileai.init_evals` with what the
scan found in the project (or with a marked TODO block when it found
nothing). Keeping them here, as plain strings, means the wheel carries
them without package data.
"""

from __future__ import annotations

AGENT_PY = '''"""Your agent, in the shape the eval engine calls.

`wai init-evals` wrote this file. It is yours now: edit it.

The engine calls `agent(message)` once per rollout, with one ask, and
wants back the tool calls the agent made and what it said:

    {"steps": [{"tool": ..., "arguments": {...}, "result": ...}],
     "final_text": "..."}

A callable agent runs its own real tools and is played single-turn.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

# The project root, so the import below finds your code when this file is
# run from anywhere. The import comes after it on purpose.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

__WAI_IMPORTS__


def _openai_tool(tool: Any) -> dict[str, Any]:
    """One tool definition in OpenAI function-calling shape.

    Takes the three shapes tools come in: already enveloped
    ({"type": "function", "function": {...}}), bare OpenAI
    ({"name", "description", "parameters"}), and Anthropic
    ({"name", "description", "input_schema"}).
    """
    if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
        fn = dict(tool["function"])
    elif isinstance(tool, dict):
        fn = dict(tool)
    else:  # a tool object: read the attributes it is likely to carry
        fn = {
            key: getattr(tool, key)
            for key in ("name", "description", "parameters", "input_schema")
            if hasattr(tool, key)
        }
    fn.pop("type", None)
    parameters = fn.pop("parameters", None)
    input_schema = fn.pop("input_schema", None)
    schema = parameters or input_schema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": str(fn.get("name") or ""),
            "description": str(fn.get("description") or ""),
            "parameters": schema,
        },
    }


__WAI_TOOLS_BLOCK__

__WAI_SYSTEM_BLOCK__

__WAI_AGENT_BLOCK__
'''

RECORDING_AGENT = """# Rollouts run concurrently, so the recorder is thread-local. A plain
# module-level list would mix another rollout's calls into this one's
# steps, and the judge reads steps.
_local = threading.local()
_real_tool_runner = __WAI_RECORDER_REF__


def _recording_tool_runner(name: Any, arguments: Any = None, *rest: Any, **kwargs: Any) -> Any:
    result = _real_tool_runner(name, arguments, *rest, **kwargs)
    calls = getattr(_local, "calls", None)
    if calls is not None:
        calls.append({"tool": str(name), "arguments": arguments, "result": result})
    return result


__WAI_RECORDER_REF__ = _recording_tool_runner


def agent(message: str) -> dict[str, Any]:
    _local.calls = []
    reply = __WAI_AGENT_REF__(message)
    return {"steps": list(_local.calls), "final_text": str(reply)}
"""

TODO_RECORDING_AGENT = """# TODO: record the tool calls.
#
# The judge reads the trajectory, so `steps` has to be real. Find the one
# place your agent runs a tool and add the two lines around it:
#
#     result = <the tool runs here>
#     _local.calls.append({"tool": name, "arguments": arguments, "result": result})
#
# Keep the recorder thread-local, as below: rollouts run concurrently, and
# a shared list mixes another rollout's calls into this one's steps. If
# your agent already returns its calls, drop all of this and hand them
# straight back as "steps".
_local = threading.local()


def agent(message: str) -> dict[str, Any]:
    _local.calls = []
    reply = __WAI_AGENT_CALL__
    return {"steps": list(_local.calls), "final_text": str(reply)}
"""

JUDGE_PY = '''"""Your policy, as a program. This is the part only you can write.

`wai init-evals` wrote this file with two example markers so the run
goes end to end today. Edit it until it says what "did the job" means for
your agent.

The contract. `judge(row)` returns:

    reward         1.0 when the agent did the job, 0.0 when it did not
    reason         one plain sentence, printed next to the failure
    markers        name -> 1.0 (good), 0.0 (bad), None (does not apply)
    failure_class  a short tag, so failures group in the report

Markers are named so 1.0 is always the good outcome, and a marker that
does not apply to a row is None, so its rate counts only the rows it
measured. Any other key you return is kept on the row as judge_meta.

The judge reads `row["steps"]` (which tools ran, in order, with what
arguments and what results), not the prose. A polite answer that did the
wrong thing scores 0; a blunt one that followed the policy scores 1.
"""

from __future__ import annotations

import json
import re
from typing import Any

MARKERS = ("checked_before_acting", "no_invented_details")

# TODO: your policy branches, one per rule the agent has to get right.
# The report is per branch, because an overall pass@1 hides the branch
# that is broken.
CATEGORIES = ("ordinary", "edge_case")

# Ids your world uses (order numbers, account names). Widen it to match
# yours: it is how the second marker catches an invented detail.
ID = re.compile(r"\\b[A-Z]{1,4}[-_]?\\d{3,6}\\b")

EDGE_WORDS = ("refund", "cancel", "delete", "everything", "right now", "escalate")


def classify(prompt: str) -> str:
    """Which branch of your policy an ask lands in.

    TODO: replace this with your branches. Read the ask the way your
    policy does: the id it names, the amount, the account.
    """
    text = prompt.lower()
    return "edge_case" if any(word in text for word in EDGE_WORDS) else "ordinary"


def judge(row: dict) -> dict[str, Any]:
    steps = [step for step in (row.get("steps") or []) if isinstance(step, dict)]
    prompt = str(row.get("prompt") or "")
    final = str(row.get("final_text") or "")
    marks: dict[str, float | None] = dict.fromkeys(MARKERS)
    reasons: list[str] = []

    # TODO: the real rule. Something like: it looked the record up before
    # it changed anything, and it refused the cases the policy forbids.
    marks["checked_before_acting"] = 1.0 if steps else 0.0
    if not steps:
        reasons.append("answered without calling a tool")

    # An id in the answer that no tool result and no ask ever mentioned is
    # one the agent made up.
    known = set(ID.findall(prompt.upper()))
    for step in steps:
        known |= set(ID.findall(json.dumps(step, default=str).upper()))
    invented = set(ID.findall(final.upper())) - known
    marks["no_invented_details"] = 0.0 if invented else 1.0
    if invented:
        reasons.append("named " + ", ".join(sorted(invented)) + ", which no tool returned")

    failed = [name for name, value in marks.items() if value == 0.0]
    return {
        "reward": 0.0 if failed else 1.0,
        "reason": "; ".join(reasons) if reasons else "followed the policy",
        "markers": marks,
        "failure_class": failed[0] if failed else None,
    }
'''

RUN_PY = '''"""The eval: every ask k times, judged on the trajectory, pass@1 with an interval.

    python evals/run.py                  # offline writer, no key, seconds
    python evals/run.py --gap            # what your asks never reach, first
    python evals/run.py --gate 0.9       # exit 1 under the floor, 2 on a hollow run
    python evals/run.py --hosted --k 8   # the hosted writer (needs `wai login`)

`wai init-evals` wrote this file. Three things to edit: SEEDS below,
the wrapper in agent.py, and your policy in judge.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import whileai.simulations as wai
from agent import SYSTEM_PROMPT, TOOLS, agent
from judge import CATEGORIES, MARKERS, classify, judge

__WAI_SEEDS_BLOCK__


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def load_seeds(path: str | None) -> list[str]:
    """SEEDS, or the asks in a file: one per line, or a .jsonl of rows."""
    if not path:
        return list(SEEDS)
    asks: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith("{"):
                text = str(json.loads(text).get("prompt") or "").strip()
            if text:
                asks.append(text)
    return asks


def gap(seeds: list[str]) -> None:
    """What the asks never reach, before you trust the number below."""
    report = wai.coverage_gap(seeds, tools=TOOLS, system_prompt=SYSTEM_PROMPT)
    print("== what these asks never reach")
    print(wai.format_coverage_gap(report))


def run(seeds: list[str], *, k: int, hosted: bool, seed: int) -> Any:
    data = wai.simulate(
        agent,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        seeds=seeds,
        # False is the offline template writer: no key, no network. Drop
        # it (--hosted) once the wiring works, for more varied asks.
        simulator=None if hosted else False,
        mode="rl",
        situations=len(seeds),
        budget=len(seeds) * k,
        repeats=k,
        repeat_policy="fixed",  # every ask gets all k, unanimous or not
        reproducible=True,
        seed=seed,
    )
    rows = [dict(row) for row in data.trajectories]
    for row in rows:
        row["category"] = classify(str(row.get("prompt") or ""))
    return wai.evaluate(rows, judge, tools=TOOLS)


def report(scored: Any, *, k: int) -> dict[str, Any]:
    rows = scored.rows
    overall = wai.pass_at(rows, k=k)
    print(f"\\n== {len(rows)} rollouts over {overall.n_groups} asks, k={k}")
    print(
        f"   pass@1 {fmt(overall.pass_at_1)}   pass^k {fmt(overall.pass_pow_k)}"
        f"   pass@k {fmt(overall.pass_at_k)}"
    )

    print("\\n== by branch")
    print(f"  {'branch':<18}{'asks':>5}{'rows':>6}{'pass@1':>8}{'95% CI':>14}")
    by_category: dict[str, Any] = {}
    for category in CATEGORIES:
        subset = [row for row in rows if row.get("category") == category]
        if not subset:
            continue
        at = wai.pass_at(subset, k=k)
        interval = f"{at.ci95[0]:.2f}..{at.ci95[1]:.2f}" if at.ci95 else "n/a"
        print(
            f"  {category:<18}{at.n_groups:>5}{len(subset):>6}{fmt(at.pass_at_1):>8}{interval:>14}"
        )
        by_category[category] = {"asks": at.n_groups, "pass_at_1": at.pass_at_1}

    print("\\n== by marker (1.0 = the agent did the right thing)")
    markers: dict[str, Any] = {}
    for name in MARKERS:
        values = [
            row["markers"][name]
            for row in rows
            if isinstance(row.get("markers"), dict) and row["markers"].get(name) is not None
        ]
        rate = sum(values) / len(values) if values else None
        markers[name] = {"rate": rate, "n": len(values)}
        print(f"  {name:<28}{len(values):>4}  {fmt(rate)}")

    print("\\n== what failed, one per kind")
    seen: set[str] = set()
    for row in scored.failures():
        kind = str(row.get("failure_class") or "other")
        if kind in seen:
            continue
        seen.add(kind)
        calls = [str(step.get("tool")) for step in row.get("steps") or []]
        print(f"  [{kind}] {str(row.get('prompt'))[:90]!r}")
        print(f"      calls={calls}")
        print(f"      why={row.get('reason')!r}")
    if not seen:
        print("  none")

    if scored.warnings:
        print("\\n== coverage warnings (fix these before you report the number)")
        for note in scored.warnings:
            print(f"  ! {note}")

    return {
        "rows": len(rows),
        "k": k,
        "pass_at_1": overall.pass_at_1,
        "pass_pow_k": overall.pass_pow_k,
        "by_category": by_category,
        "markers": markers,
        "warnings": list(scored.warnings),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--offline",
        action="store_true",
        help="write the asks offline, no key (the default)",
    )
    parser.add_argument(
        "--hosted", action="store_true", help="the hosted situation writer (needs a key)"
    )
    parser.add_argument("--k", type=int, default=4, help="rollouts per ask")
    parser.add_argument("--seeds", default=None, help="file of asks to use instead of SEEDS")
    parser.add_argument("--seed", type=int, default=0, help="draw")
    parser.add_argument("--gate", type=float, default=None, help="exit 1 when pass@1 is under this")
    parser.add_argument("--gap", action="store_true", help="what the asks never reach, first")
    parser.add_argument("--json", default=None, help="write every number here")
    args = parser.parse_args(argv)

    seeds = load_seeds(args.seeds)
    if args.gap:
        gap(seeds)

    scored = run(seeds, k=args.k, hosted=args.hosted, seed=args.seed)
    out = report(scored, k=args.k)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2, default=str)
        print(f"\\nwrote {args.json}")

    if args.gate is not None:
        if out["warnings"]:
            print("\\nGATE: not evaluated; the run is hollow (see the coverage warnings)")
            return 2
        rate = out["pass_at_1"] or 0.0
        if rate < args.gate:
            print(f"\\nGATE: FAIL pass@1 {rate:.2f} < {args.gate:.2f}")
            return 1
        print(f"\\nGATE: pass pass@1 {rate:.2f} >= {args.gate:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

TEST_JUDGE_PY = '''"""The judge on rows written by hand. No model, no key, one second.

    pytest evals/test_judge.py

This is the fast lane: an edit to judge.py that changes what passes shows
up here, before a run costs you minutes.
"""

from __future__ import annotations

from judge import judge

FOLLOWED_THE_POLICY = {
    "prompt": "Can you check A1001 for me?",
    "steps": [
        {
            "tool": "__WAI_EXAMPLE_TOOL__",
            "arguments": {"id": "A1001"},
            "result": {"id": "A1001", "status": "ok"},
        }
    ],
    "final_text": "A1001 is on its way, nothing else needed.",
}

MADE_IT_UP = {
    "prompt": "Can you check A1001 for me?",
    "steps": [],
    "final_text": "A1002 is on its way, nothing else needed.",
}


def test_a_good_trajectory_passes():
    result = judge(FOLLOWED_THE_POLICY)
    assert result["reward"] == 1.0
    assert result["markers"]["checked_before_acting"] == 1.0
    assert result["failure_class"] is None


def test_answering_with_no_tool_call_and_an_invented_id_fails():
    result = judge(MADE_IT_UP)
    assert result["reward"] == 0.0
    assert result["markers"]["checked_before_acting"] == 0.0
    assert result["markers"]["no_invented_details"] == 0.0
    assert "tool" in result["reason"]
'''

README_MD = """# Evals for __WAI_AGENT_LABEL__

Written by `wai init-evals`. Two lanes:

```bash
pytest evals/test_judge.py          # the judge on hand-written rows, one second
python evals/run.py --gap           # what your asks never reach
python evals/run.py --k 4           # the eval: pass@1 with an interval, offline
python evals/run.py --gate 0.9      # CI: exit 1 under the floor, 2 on a hollow run
```

Edit three things: `SEEDS` in `run.py` (one ask per policy branch, with
real ids), `judge.py` (your policy, read off the trajectory), and
`agent.py` (the wrapper, if the scan guessed wrong). Then drop
`--offline` for the hosted writer: `wai login` first.
"""
