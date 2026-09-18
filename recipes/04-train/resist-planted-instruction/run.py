"""Rows for a behaviour a program can grade: resist a planted instruction.

The teacher is the base model, the reward is code, and only rows the reward
passed are kept (rlhf-book ch. 9). `--dry-run` grades ten bundled rows and
runs both selection rules on them with no key, no network and no GPU.

Run: python recipes/04-train/resist-planted-instruction/run.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    import whileai.simulations as wai
except ImportError:  # a fresh clone, before the package is installed
    raise SystemExit(
        "This recipe needs the SDK importable. From the repository root run "
        "`uv sync --extra dev` (or `pip install -e .`), then re-run this script."
    ) from None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rubric import criterion_failures, make_grader, rubric
from world import (
    ORDER_RE,
    POLICY,
    TEACHER_SCAFFOLD,
    TOOLS,
    Store,
    build_world,
    make_execute,
    run_tool,
)

HERE = Path(__file__).resolve().parent

# The two worlds never share an order id, so decontamination has a structural
# answer as well as the SDK's n-gram one. Change the seeds and the id blocks
# together or the holdout stops being a holdout.
TRAIN_WORLD = {"n": 700, "seed": 1717, "prefix": 40000, "attack_share": 0.8}
HOLDOUT_WORLD = {"n": 120, "seed": 9091, "prefix": 58000, "attack_share": 0.8}
PROBE_WORLD = {"n": 100, "seed": 3131, "prefix": 71000, "attack_share": 0.8}

SFT_TARGET = 376  # what the reward filter yielded; the control matches it


def tasks_for(world: dict) -> list[dict]:
    """One pinned task per scenario. The opener names the order, the grader
    reads the order id back out of the prompt, so no privileged plumbing."""
    return [{"prompt": sc["opener"], "scenario_id": sid} for sid, sc in world["scenarios"].items()]


def structurally_clean(rows: list[dict], holdout: dict) -> list[dict]:
    """Drop any row that names a holdout order or repeats a holdout opener.

    The SDK coverage rule flags template-shared opener frames on this set
    (0 exact hits, hundreds of near ones); both numbers are printed, this
    one is what training uses.
    """
    ids = set(holdout["scenarios"])
    openers = {sc["opener"] for sc in holdout["scenarios"].values()}
    keep = []
    for r in rows:
        prompt = str(r.get("prompt") or "")
        if prompt in openers or any(o in ids for o in ORDER_RE.findall(prompt)):
            continue
        keep.append(r)
    return keep


def select_and_unroll(rows: list[dict], out: Path | None) -> dict:
    """Both ch. 9 selection rules, then the masked SFT samples.

    `top_per_prompt` with `min_reward=1.0` is the treatment: the best
    completion per prompt, only if it passed. `random_per_prompt` with
    `min_reward=0.0` is the matched control: the same count, the same
    one-per-prompt rule and signature round-robin, no reward filter.
    """
    selected, report = wai.select_for_sft(
        rows, min_reward=1.0, target=SFT_TARGET, select="top_per_prompt", seed=0
    )
    control, creport = wai.select_for_sft(
        rows, min_reward=0.0, target=SFT_TARGET, select="random_per_prompt", seed=0
    )
    sft = wai.training_rows(
        selected,
        system_prompt=POLICY,
        tools=TOOLS,
        unroll=True,
        strip_think=True,
        max_tool_output_chars=4000,
    )
    sft_control = wai.training_rows(
        control,
        system_prompt=POLICY,
        tools=TOOLS,
        unroll=True,
        strip_think=True,
        max_tool_output_chars=4000,
    )
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        for name, data in (
            ("selected.jsonl", selected),
            ("selected_random.jsonl", control),
            ("sft_rows.jsonl", sft),
            ("sft_rows_random.jsonl", sft_control),
        ):
            (out / name).write_text("\n".join(json.dumps(r, default=str) for r in data))
    mix = Counter(r.get("reward") for r in control)
    return {
        "selected": len(selected),
        "selected_report": {k: report.get(k) for k in ("n_eligible", "unique_behaviors")},
        "control": len(control),
        "control_reward_mix": {str(k): v for k, v in mix.items()},
        "control_report": {k: creport.get(k) for k in ("n_eligible", "unique_behaviors")},
        "sft_samples": len(sft),
        "sft_samples_control": len(sft_control),
        "loss_mask_example": sft[0]["loss_mask"] if sft else None,
    }


def dry_run(limit: int) -> int:
    world = build_world(**TRAIN_WORLD)
    scenarios = world["scenarios"]

    # 1. The world refuses what it should. A world that confirmed these would
    #    let the grader score a fabrication as a completed task.
    store = Store(world["db"])
    oid = next(iter(world["db"]["orders"]))
    total = world["db"]["orders"][oid]["total_cents"]
    checks = {
        "unknown order": run_tool(store, "get_order", {"order_id": "SO-00000"}).get("error"),
        "refund over total": run_tool(
            store, "issue_refund", {"order_id": oid, "amount_cents": total + 1, "reason": "x"}
        ).get("error"),
        "email off the record": run_tool(
            store, "send_email", {"to": "nobody@nowhere.example", "subject": "s", "body": "b"}
        ).get("error"),
    }
    print("world refusals:", checks)
    assert all(checks.values()), "the world confirmed something it should have refused"

    # 2. The grader reproduces the labels the published rows carry.
    rows = [
        json.loads(line)
        for line in (HERE / "fixtures.jsonl").read_text().splitlines()
        if line.strip()
    ][:limit]
    recorded = [r.get("reward") for r in rows]
    scored = wai.run_judge(rows, make_grader(scenarios), version=f"code@{rubric().version}")
    regraded = [r.get("reward") for r in scored.rows]
    assert regraded == recorded, f"grader drifted: {recorded} -> {regraded}"
    print(f"regraded {len(rows)} bundled rows; every label reproduced")
    print(wai.pass_at(scored.rows))
    print("per-criterion failures:", criterion_failures(scored.rows)["failures"])

    # 3. Both selection rules and the masked samples, on the same rows.
    print("selection:", json.dumps(select_and_unroll(scored.rows, None), indent=1))
    print(
        "\nNext: python run.py --backend vllm:Qwen/Qwen3-4B@http://127.0.0.1:8000/v1 "
        "--repeats 2, then modal run modal_train_eval.py::run_train"
    )
    return 0


def select_pool(args: argparse.Namespace) -> int:
    """Decontaminate and select from pools that `run_generate` on Modal
    already graded: `out/pool_seed<N>.jsonl`, one file per wave."""
    world = build_world(**TRAIN_WORLD)
    holdout = build_world(**HOLDOUT_WORLD)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for path in args.pool:
        wave = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        print(f"{path}: {len(wave)} rows")
        rows += wave
    # The same grader, on the same scenarios, so a pool graded by an older
    # rubric version cannot slip through with stale labels.
    scored = wai.run_judge(
        rows, make_grader(world["scenarios"]), version=f"code@{rubric().version}"
    )
    rows = scored.rows
    (out / "pool.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in rows))
    print(f"pool {len(rows)} rows")
    print(wai.pass_at(rows))
    print("per-criterion failures:", criterion_failures(rows)["failures"])
    return finish(rows, holdout, out)


def finish(rows: list[dict], holdout: dict, out: Path) -> int:
    sdk_clean, report = wai.decontaminate(rows, tasks_for(holdout))
    clean = structurally_clean(rows, holdout)
    print(
        f"decontaminate: SDK rule kept {len(sdk_clean)} (exact {report.get('n_exact')}, "
        f"near {report.get('n_near')}); structural rule kept {len(clean)}"
    )
    selection = select_and_unroll(clean, out)
    print("selection:", json.dumps(selection, indent=1))
    (out / "selection.json").write_text(json.dumps(selection, indent=1))
    (out / "holdout_tasks.json").write_text(json.dumps(tasks_for(holdout), indent=1))
    print(f"\nNext: modal run modal_train_eval.py::run_train --rows-file {out / 'sft_rows.jsonl'}")
    return 0


def live(args: argparse.Namespace) -> int:
    """Generate from the base, grade with code, decontaminate, select.

    Every knob is written out. `budget` matters: the SDK default of 1000
    rows stopped the first wave at 500 of 700 scenarios.
    """
    world = build_world(**TRAIN_WORLD)
    holdout = build_world(**HOLDOUT_WORLD)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = wai.simulate(
        tools=TOOLS,
        system_prompt=POLICY,
        backend=args.backend,
        # The customer and the situation writer are pinned to the same model
        # as the policy here on purpose: this is generation, and the eval
        # pins them again so both arms face the same customer.
        user_model=args.user_model or args.backend,
        simulator=args.user_model or args.backend,
        execute=make_execute(world),
        # Generation-only. Appended to the teacher's system prompt, never to
        # profile.policy, so exports and both eval arms run on POLICY alone.
        scaffold=TEACHER_SCAFFOLD,
        tasks=tasks_for(world),
        repeats=args.repeats,
        budget=args.budget,
        avg_turns=6.0,
        max_turns=30,
        concurrency=32,
        seed=args.seed,
        time_budget=args.time_budget,
        fault_rate=None,  # difficulty comes from the planted text, not faults
        grade=False,
    )
    scored = wai.run_judge(
        data.trajectories, make_grader(world["scenarios"]), version=f"code@{rubric().version}"
    )
    rows = scored.rows
    (out / "pool.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in rows))
    print(f"rollouts {len(rows)}, stopped because {data.stopped_because}")
    print(wai.pass_at(rows))
    print("per-criterion failures:", criterion_failures(rows)["failures"])
    return finish(rows, holdout, out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="no model calls, no key")
    p.add_argument("--limit", type=int, default=10, help="bundled rows to grade")
    p.add_argument("--backend", default="", help="vllm:<model>@<url>; turns on generation")
    p.add_argument(
        "--pool",
        nargs="+",
        default=[],
        help="graded pool files from `modal run modal_train_eval.py::run_generate`; "
        "skips generation and selects from them",
    )
    p.add_argument("--user-model", default="", help="backend spec for the customer")
    p.add_argument("--repeats", type=int, default=2, help="rollouts per scenario (k)")
    p.add_argument("--budget", type=int, default=1500, help="rollout cap for this wave")
    p.add_argument("--seed", type=int, default=11, help="rollout seed; vary it per wave")
    p.add_argument("--time-budget", type=float, default=2700.0, help="seconds")
    p.add_argument("--out", default=str(HERE / "out"), help="where rows land")
    args = p.parse_args(argv)
    if args.pool and not args.dry_run:
        return select_pool(args)
    if args.dry_run or not args.backend:
        return dry_run(args.limit)
    return live(args)


if __name__ == "__main__":
    raise SystemExit(main())
