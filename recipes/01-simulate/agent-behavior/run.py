"""Run the simulation and ship it: spans to /v1/traces, verdicts to /v1/scores.

    python run.py --runs 40 --days 3

Every run is one agent turn on one task under one persona. What leaves this
process per run is one OTLP batch (root span plus every child) and one POST of
named measurements against that trace id. Nothing is retried in the background
and nothing is buffered across runs: if a send fails you see it, on the line
for the run it belonged to.

`--days` backdates the spans. The platform's time filter reads the producer's
clock, so spreading 40 runs across three days gives the charts a shape instead
of a single vertical line at the moment you ran this. The dataset itself is
still filed under today, which is the day the store first saw the trace.
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import agent as agents
import gate
import tasks as task_module

VERSION = "0.1.0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", type=int, default=40, help="agent turns to simulate (default 40)")
    p.add_argument(
        "--days", type=float, default=3.0, help="spread run timestamps over this many past days"
    )
    p.add_argument("--concurrency", type=int, default=4, help="turns in flight at once (default 4)")
    p.add_argument(
        "--dataset", default="agent-behavior-demo", help="whileai.dataset resource attribute"
    )
    p.add_argument(
        "--agent", default="demo-agent", help="gen_ai.agent.name; the row the platform groups by"
    )
    p.add_argument(
        "--service", default="agent-behavior-example", help="service.name resource attribute"
    )
    p.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    p.add_argument(
        "--personas", default="", help="comma-separated persona names (default: the weighted mix)"
    )
    p.add_argument(
        "--max-steps", type=int, default=12, help="tool rounds before a turn is cancelled"
    )
    p.add_argument(
        "--seed", type=int, default=None, help="make the task and persona draw reproducible"
    )
    p.add_argument("--api-key", default=None, help=f"zp_ key, or set ${gate.API_KEY_ENV}")
    p.add_argument(
        "--gate", default=None, help=f"platform API base. Required, or set ${gate.API_URL_ENV}."
    )
    p.add_argument(
        "--model-url",
        default=None,
        help=f"OpenAI-compatible base URL ending in /v1. Required, or set ${agents.MODEL_URL_ENV}.",
    )
    p.add_argument(
        "--model-key",
        default=None,
        help=f"bearer token for the model endpoint, or set ${agents.MODEL_KEY_ENV}",
    )
    p.add_argument("--model", default=None, help=f"model id (default: {agents.DEFAULT_MODEL})")
    p.add_argument(
        "--no-judge", action="store_true", help="skip the judge; send observable signals only"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="run everything, send nothing, print the summary"
    )
    return p.parse_args(argv)


def draw(
    count: int,
    task_ids: list[str],
    personas: list[str],
    weights: dict[str, float],
    rng: random.Random,
):
    """Pick (task, persona) for every run.

    Tasks round-robin rather than sampling, so every task is attempted several
    times. That is what makes the data trainable later: a scenario attempted
    once is a group of one, and a group of one has no variance to learn from.
    Personas are sampled, so the same task lands on a careful agent and a
    careless one and the difference is visible within the scenario.
    """
    pool = [personas[0]] if len(personas) == 1 else personas
    population = [p for p in pool if p in weights] or pool
    ws = [weights.get(p, 1.0) for p in population]
    return [
        (
            task_module.BY_ID[task_ids[i % len(task_ids)]],
            rng.choices(population, weights=ws, k=1)[0],
        )
        for i in range(count)
    ]


def stamp(runs_total: int, index: int, days: float, rng: random.Random) -> float:
    """A plausible past start time for run `index`, in epoch ms.

    Spread evenly across the window, then jittered, so the sparklines have
    texture rather than a comb. Nothing lands in the future.
    """
    now = time.time() * 1000
    if days <= 0:
        return now
    window = days * 24 * 3600 * 1000
    base = now - window + (window * (index + 0.5) / max(runs_total, 1))
    jitter = rng.uniform(-0.4, 0.4) * (window / max(runs_total, 1))
    return min(base + jitter, now - 1000)


def build_trace(
    run: agents.Run, task, args: argparse.Namespace, started_ms: float
) -> tuple[gate.Trace, dict, float]:
    """The whole run as one OTLP batch, with every timestamp shifted to `started_ms`."""
    shift = started_ms - run.began_ms

    trace = gate.Trace(
        agent=args.agent,
        prompt=task.prompt,
        started_ms=started_ms,
        # One task attempt is one conversation. The scenario id is the task, so
        # every attempt at it groups together across conversations.
        session_id=f"{task.id}-{int(started_ms)}",
        scenario_id=task.id,
        attributes={
            "zeroproof.persona": run.persona,
            "zeroproof.task": task.id,
        },
    )

    model = args.model or os.environ.get("WHILEAI_MODEL") or agents.DEFAULT_MODEL
    for step in run.steps:
        if step.kind == "llm":
            trace.llm(
                model=model,
                input_messages=step.payload["input_messages"],
                output_text=step.payload["output_text"],
                tool_calls=step.payload["tool_calls"],
                started_ms=step.started_ms + shift,
                duration_ms=step.duration_ms,
                input_tokens=step.payload["input_tokens"],
                output_tokens=step.payload["output_tokens"],
            )
        else:
            trace.tool(
                name=step.payload["name"],
                arguments=step.payload["arguments"],
                result=step.payload["result"],
                failed=step.payload["failed"],
                started_ms=step.started_ms + shift,
                duration_ms=step.duration_ms,
            )

    # Ground truth is not in here. It goes out on POST /v1/scores instead, as
    # the one measurement carrying a `pass_at`, because that is what nominates
    # it as the outcome the charts get ranked against and a span attribute
    # cannot say it. See `agents.ground_truth_score`.
    measurements = run.signals.summarize(run.outcome, run.final_text)

    # The label a trainer would read. Solved is necessary, and gaming the suite
    # disqualifies: an agent that skipped the test did not earn the row even if
    # something else made the held-out suite pass.
    gamed = any(k.startswith("hack.") for k in run.signals.evidence)
    reward = 1.0 if (run.solved and not gamed) else 0.0

    ended_ms = started_ms + run.duration_ms
    body = trace.body(
        {
            "service.name": args.service,
            "service.version": VERSION,
            # The name the gate files these traces under; it reads this key.
            "zeroproof.dataset": args.dataset,
        },
        final_text=run.final_text,
        ended_ms=ended_ms,
        outcome=run.outcome,
        measurements=measurements,
        evidence=run.signals.evidence,
        reward=reward,
    )
    return trace, body, reward


def one(index: int, task, persona: str, args: argparse.Namespace, rng_seed: int) -> dict:
    """Simulate, grade, judge, and send one turn. Returns a row for the summary."""
    llm = agents.Llm(args.model_url, args.model_key, args.model)
    run = agents.run_turn(task, persona, llm, max_steps=args.max_steps)

    started_ms = stamp(args.runs, index, args.days, random.Random(rng_seed))
    trace, body, reward = build_trace(run, task, args, started_ms)

    row = {
        "task": task.id,
        "persona": persona,
        "outcome": run.outcome,
        "solved": run.solved,
        "reward": reward,
        "misbehaviour": len(run.signals.evidence),
        "flags": sorted(run.signals.evidence),
        "trace_id": trace.trace_id,
        "score": None,
        "sent": False,
        "error": run.error,
    }

    verdict: dict = {}
    raw = ""
    if not args.no_judge:
        verdict, raw = agents.judge(task, run, llm)
        score = verdict.get("score")
        row["score"] = float(score) if isinstance(score, (int, float)) else None

    if args.dry_run:
        return row

    client = gate.Client(args.api_key, args.gate)
    try:
        client.send_trace(body)
        row["sent"] = True
    except gate.GateError as err:
        row["error"] = f"trace: {err}"
        return row

    # Ground truth goes out whether or not a judge ran. It is the measurement
    # carrying `pass_at`, so without it nothing on the run has been nominated
    # as the outcome and the charts can only be ordered by volume: `--no-judge`
    # would quietly cost you the ranking as well as the verdict.
    scores = [agents.ground_truth_score(run.solved)]
    if not args.no_judge:
        scores += agents.verdict_scores(verdict, raw)

    try:
        client.send_scores(trace.trace_id, scores)
    except gate.GateError as err:
        row["error"] = f"scores: {err}"

    return row


def summarize(rows: list[dict]) -> str:
    """The same story the dashboard tells, in the terminal, before you go look."""
    by_persona: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_persona[row["persona"]].append(row)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    lines = [
        "",
        f"{'persona':<12} {'runs':>5} {'solved':>7} {'judge':>7} {'misbeh':>7} {'edited tests':>13} {'claimed pass':>13}",
        "-" * 70,
    ]
    for persona in sorted(
        by_persona, key=lambda p: -mean([1.0 if r["solved"] else 0.0 for r in by_persona[p]])
    ):
        group = by_persona[persona]
        scores = [r["score"] for r in group if r["score"] is not None]
        lines.append(
            f"{persona:<12} {len(group):>5} "
            f"{mean([1.0 if r['solved'] else 0.0 for r in group]):>6.0%} "
            f"{(mean(scores) if scores else float('nan')):>7.2f} "
            f"{mean([float(r['misbehaviour']) for r in group]):>7.2f} "
            f"{mean([1.0 if 'hack.test_edited' in r['flags'] else 0.0 for r in group]):>12.0%} "
            f"{mean([1.0 if 'lie.tests_claimed' in r['flags'] else 0.0 for r in group]):>12.0%}"
        )

    scores = [r["score"] for r in rows if r["score"] is not None]
    solved = mean([1.0 if r["solved"] else 0.0 for r in rows])
    lines += [
        "-" * 70,
        f"{'all':<12} {len(rows):>5} {solved:>6.0%} "
        f"{(mean(scores) if scores else float('nan')):>7.2f} "
        f"{mean([float(r['misbehaviour']) for r in rows]):>7.2f}",
        "",
        f"trainable rows (reward 1): {sum(1 for r in rows if r['reward'] == 1.0)} of {len(rows)}",
    ]

    if scores:
        # The one comparison worth printing. The judge reads the transcript; the
        # held-out suite reads the code. Where they disagree is the whole point.
        gap = mean(scores) - solved
        lines.append(
            f"judge mean {mean(scores):.2f} vs held-out solve rate {solved:.0%}: "
            f"the judge is {abs(gap):.0%} {'over' if gap > 0 else 'under'} ground truth"
        )

    failures = [r for r in rows if r["error"]]
    if failures:
        lines.append(f"\n{len(failures)} run(s) had errors:")
        lines += [f"  {r['task']}/{r['persona']}: {r['error']}" for r in failures[:10]]

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    # Model output is UTF-8 and the Windows console is not. Without this a run
    # dies partway through printing a summary it already paid for.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)

    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()] or [
        t.id for t in task_module.TASKS
    ]
    unknown = [t for t in task_ids if t not in task_module.BY_ID]
    if unknown:
        print(f"unknown task(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(task_module.BY_ID)}", file=sys.stderr)
        return 2

    personas = [p.strip() for p in args.personas.split(",") if p.strip()] or list(agents.PERSONAS)
    unknown = [p for p in personas if p not in agents.PERSONAS]
    if unknown:
        print(f"unknown persona(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(agents.PERSONAS)}", file=sys.stderr)
        return 2

    if not args.dry_run:
        # Fail on a missing key now rather than after the model bill.
        try:
            gate.Client(args.api_key, args.gate)
        except gate.GateError as err:
            print(err, file=sys.stderr)
            return 2

    # A dry run still needs the model. Say so once, here, rather than once
    # per row from inside the pool with an exit code of 0 at the end.
    try:
        agents.Llm(args.model_url, args.model_key, args.model)
    except agents.LlmError as err:
        print(err, file=sys.stderr)
        return 2

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    rng = random.Random(seed)
    plan = draw(args.runs, task_ids, personas, agents.PERSONA_WEIGHTS, rng)

    print(f"{args.runs} runs, {len(task_ids)} tasks, {len(personas)} personas, seed {seed}")
    print(f"dataset {args.dataset!r}, agent {args.agent!r}" + (", dry run" if args.dry_run else ""))
    print()

    rows: list[dict] = []
    began = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {
            pool.submit(one, i, task, persona, args, seed + i): (i, task, persona)
            for i, (task, persona) in enumerate(plan)
        }
        for done in as_completed(futures):
            _i, task, persona = futures[done]
            try:
                row = done.result()
            except Exception as err:  # a crashed run must not take the batch with it
                row = {
                    "task": task.id,
                    "persona": persona,
                    "outcome": "error",
                    "solved": False,
                    "reward": 0.0,
                    "misbehaviour": 0,
                    "flags": [],
                    "trace_id": "",
                    "score": None,
                    "sent": False,
                    "error": f"{type(err).__name__}: {err}",
                }
            rows.append(row)

            mark = "ok " if row["solved"] else "BAD"
            flags = ",".join(f.split(".")[-1] for f in row["flags"][:3]) or "-"
            score = f"{row['score']:.2f}" if row["score"] is not None else " -- "
            print(
                f"[{len(rows):>3}/{args.runs}] {mark} {row['task']:<18} {row['persona']:<12} "
                f"judge {score}  flags {flags:<28} {row['error'] or ''}"
            )

    print(summarize(rows))
    print(f"\n{time.time() - began:.0f}s")
    if not args.dry_run:
        print(
            f"\nhttps://app.withwhile.com/platform/traces  (agent {args.agent!r}, last {args.days:g} days)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
