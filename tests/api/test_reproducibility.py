"""A seeded serial run is the same run in every process.

Python salts str hashes per process, so anything that leaks hash() or
set order into a row shows up here as a diff between two interpreters
started with different PYTHONHASHSEED values. Timing fields are
stripped; everything else must match.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_SCRIPT = r"""
import json, re, sys
sys.path.insert(0, %(repo)r)
from tests.helpers import simulate_offline
TIMING = re.compile(r"(seconds|elapsed|rate|_s$|_at$|per_second)")
def scrub(o):
    if isinstance(o, dict):
        return {k: scrub(v) for k, v in o.items() if not TIMING.search(str(k))}
    if isinstance(o, (list, tuple)):
        return [scrub(x) for x in o]
    return o
traces = [{"prompt": f"refund order 8{i}", "reward": 0, "final_text": "Refunded.",
           "steps": [{"tool": "create_refund", "arguments": {"order_id": f"8{i}"},
                      "result": {"status": "timeout"}}]} for i in range(5)]
d = simulate_offline(traces=traces, budget=24, per_round=40, concurrency=1)
print(json.dumps(scrub({"rows": d.trajectories, "search": d.search,
                        "coverage": d.coverage, "stopped": d.stopped_because}),
                 sort_keys=True, default=str))
"""


_GRADER_SCRIPT = _SCRIPT.replace(
    "d = simulate_offline(traces=traces, budget=24, per_round=40, concurrency=1)",
    "def grader(row):\n"
    "    return 1 if len(row.get('steps') or ()) >= 2 else 0\n"
    "d = simulate_offline(traces=traces, budget=24, per_round=40, concurrency=1, grader=grader)",
).replace(
    'TIMING = re.compile(r"(seconds|elapsed|rate|_s$|_at$|per_second)")',
    # lineage.scoring_run_id is per-invocation identity, documented as
    # outside the bit-for-bit guarantee, like the timing fields.
    'TIMING = re.compile(r"(seconds|elapsed|rate|_s$|_at$|per_second|scoring_run_id)")',
)
assert _GRADER_SCRIPT != _SCRIPT


def _run(hash_seed: str, script: str = _SCRIPT) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    for key in ("OPENAI_API_KEY", "WHILEAI_API_KEY", "VLLM_API_KEY"):
        env.pop(key, None)
    out = subprocess.run(
        [sys.executable, "-c", script % {"repo": str(REPO)}],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_serial_seeded_run_is_identical_across_processes():
    first = _run("1")
    second = _run("2")
    assert first["rows"], "the offline run produced no rows"
    assert first == second


def test_serial_seeded_graded_run_is_identical_except_scoring_run_id():
    """grader= is the one in-simulate path that mints per-invocation identity."""
    first = _run("1", _GRADER_SCRIPT)
    second = _run("2", _GRADER_SCRIPT)
    assert first["rows"], "the offline run produced no rows"
    assert all(r.get("reward") in (0, 1) for r in first["rows"])
    assert all("scoring_run_id" not in (r.get("lineage") or {}) for r in first["rows"])
    assert first == second


def test_parallel_run_is_identical_with_reproducible_flag():
    """Eight workers, jittered agent latency, two runs: same rows."""
    import random
    import re

    import whileai.simulations as wai
    from tests.helpers import POLICY, TOOLS, scripted_agent

    timing = re.compile(r"(seconds|elapsed|rate|_s$|_at$|per_second)")

    def scrub(o):
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items() if not timing.search(str(k))}
        if isinstance(o, (list, tuple)):
            return [scrub(x) for x in o]
        return o

    def run(jitter_seed: int):
        rng = random.Random(jitter_seed)

        def jittery(message: str) -> dict:
            import time

            time.sleep(rng.random() * 0.03)
            return scripted_agent(message)

        d = wai.simulate(
            jittery,
            tools=TOOLS,
            policy=POLICY,
            budget=40,
            seed=0,
            concurrency=8,
            simulator=False,
            grade=False,
            time_budget=None,
            reproducible=True,
            advanced={"per_round": 40, "mutate_failures": False},
        )
        return scrub(
            {
                "rows": d.trajectories,
                "search": d.search,
                "coverage": d.coverage,
                "stopped": d.stopped_because,
            }
        )

    first, second = run(11), run(97)
    assert len(first["rows"]) == 40
    assert first == second


def test_serial_graded_reruns_in_one_process_are_identical():
    """Two same-seed ``concurrency=1`` runs in one process: same rows.

    The pass-at-k example's shape (rl mode, eight asks by eight repeats,
    graded in the loop) drew different situations on a rerun. The 0.35 s
    collect window took whatever had finished when the first rollout
    landed: an instant agent had several of a batch done by then, a
    slower one exactly one, and the idle rounds in between drifted the
    counter that seeds selection. One run here answers at once and the
    other sleeps on every rollout, the two collect patterns the window
    told apart, so the test fails without round-synchronous scheduling
    instead of passing on a quiet machine.
    """
    import re
    import time

    import whileai.simulations as wai
    from tests.helpers import POLICY, TOOLS, scripted_agent
    from whileai.simulations.generate.agents import current_rollout

    timing = re.compile(r"(seconds|elapsed|rate|_s$|_at$|per_second)")

    def scrub(o):
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items() if not timing.search(str(k))}
        if isinstance(o, (list, tuple)):
            return [scrub(x) for x in o]
        return o

    def run(latency_s: float):
        def careless(message: str) -> dict:
            # Latency is not part of the agent's answer. A split group
            # (every third repeat refunds without looking) makes the
            # successive allocator's verify path run too.
            if latency_s:
                time.sleep(latency_s)
            if getattr(current_rollout, "rollout_index", 0) % 3 == 1:
                return {
                    "steps": [
                        {
                            "tool": "create_refund",
                            "arguments": {"order_id": "acct_1", "amount": 150},
                            "result": {"status": "created", "id": "re_150"},
                        }
                    ],
                    "final_text": "Refunded $150.",
                }
            return scripted_agent(message)

        d = wai.simulate(
            careless,
            tools=TOOLS,
            policy=POLICY,
            situations=6,
            repeats=8,
            budget=48,
            seed=2,
            grade="conduct",
            concurrency=1,
            simulator=False,
            time_budget=None,
            mode="rl",
        )
        return scrub(
            {
                "rows": list(d.rows()),
                "search": d.search,
                "coverage": d.coverage,
                "stopped": d.stopped_because,
            }
        )

    first, second = run(0.0), run(0.02)
    assert len(first["rows"]) == 48
    assert len({r["prompt"] for r in first["rows"]}) == 6
    assert first == second
