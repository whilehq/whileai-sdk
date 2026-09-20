"""The parts of the SFT recipe that need no GPU: the judge, the loaders, the plan.

``train_modal.py`` imports this file. ``smoke.sh`` runs it with ``--smoke``:
it writes a small export offline the way lesson 7 does, loads it back
through the same functions the trainer uses, prints the plan the trainer
gets, and scores a row that no stand-in wrote. No key, no GPU, no network,
under a minute.

Run: python recipes/04-train/sft/wiring.py --smoke
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any

# -- the course judge ----------------------------------------------------------
#
# The same program lessons 3 to 7 of docs/learn use. It reads ``messages``,
# so it scores the stand-in agent's rows and a trained model's rows alike.
# Three rules, in the order they are checked:
#
# 1. Look the order up with the id the customer gave. When the ask names no
#    order, the right move is to ask for one, not to guess one.
# 2. Lead with the result: the first sentence of the reply names the order
#    that was looked up, not a preamble.
# 3. Claim nothing the tool did not return: when the lookup did not
#    succeed, the reply does not say it did.

ORDER_ID = re.compile(r"\bORD-\d+\b")
#: the statuses the fake world reports for a call that went through
#: (``whileai.simulations.generate.offline_agent._SUCCESS``).
SUCCESS = {"ok", "created", "success", "done", "updated", "deleted"}
CLAIMED = re.compile(r"\b(done|went through|completed|succeeded|confirmed)\b", re.I)


def judge(row: dict) -> dict:
    """1 when the agent looked the order up, led with it, and claimed nothing the tool did not say."""
    messages = row["messages"]
    asked = set(ORDER_ID.findall(next(m["content"] for m in messages if m["role"] == "user")))
    calls = [c for m in messages if m["role"] == "assistant" for c in m.get("tool_calls") or []]
    if not asked:  # no id to look up: ask for one, do not guess
        return {"reward": int(not calls)}
    call = (calls[0].get("function") or calls[0]) if calls else {}
    args = call.get("arguments") or {}
    args = json.loads(args) if isinstance(args, str) else args
    if call.get("name") != "get_order" or args.get("order_id") not in asked:
        return {"reward": 0}
    reply = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""
    if args["order_id"] not in re.split(r"(?<=[.!?])\s", reply.strip())[0]:
        return {"reward": 0}  # the first sentence is the result, not a preamble
    result = next((m["content"] for m in messages if m["role"] == "tool"), "{}")
    status = json.loads(result).get("status")
    return {"reward": int(status in SUCCESS or not CLAIMED.search(reply))}


# -- the export ----------------------------------------------------------------


def _as_call(call: dict) -> dict:
    """One tool call in the shape the chat template renders: ``name`` and
    ``arguments`` as a dict. The export writes the OpenAI wire shape, with
    ``arguments`` as a JSON string; rendered as is, the model would learn to
    emit a quoted string where the template wants an object."""
    fn = call.get("function") or call
    args = fn.get("arguments") or {}
    return {
        "name": fn.get("name"),
        "arguments": json.loads(args) if isinstance(args, str) else args,
    }


def load_export(path: str | Path) -> dict[str, Any]:
    """Read ``select(mode="sft").export(path)`` back: the rows with their
    tool calls normalised, the tool schemas and the system prompt the rows
    were generated under. Refuses a file whose rows carry no messages."""
    rows = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]
    if not rows or any("messages" not in r for r in rows):
        raise SystemExit(
            f"{path}: every row needs a messages list; write it with select(mode='sft').export()"
        )
    for row in rows:
        for m in row["messages"]:
            if m.get("tool_calls"):
                m["tool_calls"] = [_as_call(c) for c in m["tool_calls"]]
    first = rows[0]["messages"][0]
    system = first["content"] if first["role"] == "system" else ""
    tools = rows[0].get("tools") or []
    return {"rows": rows, "tools": tools, "system_prompt": system}


def load_holdout(path: str | Path) -> list[dict]:
    """The held-out tasks, one per distinct ask: the ask, its task id, and
    what the fake world answered the stand-in's lookup, so the trained model
    is asked the same question in the same world."""
    tasks: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        prompt = next(m["content"] for m in row["messages"] if m["role"] == "user")
        result = next((m["content"] for m in row["messages"] if m["role"] == "tool"), None)
        tasks.setdefault(
            prompt, {"scenario_id": row["scenario_id"], "prompt": prompt, "tool_result": result}
        )
    return list(tasks.values())


# -- the plan ------------------------------------------------------------------

#: the base the other 04-train recipes train: a 1.5B instruct model that
#: fits an A10G in bf16 with room to generate beside the adapter.
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
#: steps: seven passes over the lesson's 46-row export at ROWS_PER_STEP
#: rows a step. In the run the README quotes the loss fell from 3.1 to 1.1
#: in the first twenty steps and from 0.85 to 0.80 in the last ten; more
#: passes over so few rows memorise them (Lambert 2025, chapter Instruction
#: Finetuning: a few epochs, not many).
STEPS = 40
#: the wiring run: a quarter pass, enough to watch the loss fall, under a
#: minute of GPU.
SMOKE_STEPS = 10
#: rows a step: 4 rows fit an A10G beside a 1.5B model in bf16 with room, and
#: two of those make 8, so a 100-row export is about 12 steps an epoch.
BATCH_ROWS = 4
GRAD_ACCUM = 2
ROWS_PER_STEP = BATCH_ROWS * GRAD_ACCUM
#: learning rate: ten times the full-fine-tune rate, because the adapter
#: starts at zero and a small set has to move it (Hu et al. 2021 train GPT-3
#: adapters at 2e-4; Lambert 2025, chapter Instruction Finetuning, on why
#: SFT learning rates are small).
LEARNING_RATE = 1e-4
#: adapter rank 16 with alpha 2r, the grpo recipe's setting; Hu et al. 2021
#: find rank 8 already matches full fine-tuning on GPT-3, so 16 is headroom.
#: Dropout 0.05: the export is under 100 rows; a little regularisation
#: (convention, untested here).
LORA_RANK = 16
LORA_ALPHA = 2 * LORA_RANK
LORA_DROPOUT = 0.05
#: every linear projection, not only attention: Dettmers et al. 2023
#: (QLoRA) find that is what matches full fine-tuning.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
#: the longest rendered row in the lesson's export (system prompt, tool
#: schema, one call, one result, one reply) is 411 tokens; 1024 leaves
#: room for a longer system prompt at no cost at this batch.
MAX_LENGTH = 1024
#: sampling at evaluation: the grpo recipe's setting, so the two recipes'
#: before/after numbers are read the same way.
TEMPERATURE = 0.8
TOP_P = 0.95
#: a tool call is under 40 tokens and the longest stand-in reply in the
#: export is 19 words; 128 leaves room for a wordier base model and keeps a
#: pass under a minute.
MAX_NEW_TOKENS = 128
#: tries per held-out ask: lesson 4's k, so pass^4 and headroom are on the
#: same footing as the stand-in's numbers.
SAMPLES = 4
#: asks per generate call; times SAMPLES is 32 sequences, which an A10G
#: decodes in one batch.
EVAL_BATCH = 8
#: base passes: three re-runs is the fewest that give a spread
#: (``eval_variance``: fewer than three is a difference, not a distribution).
BASE_PASSES = 3
#: the seeds of the base passes; the trained pass reuses the first, so
#: before and after are paired on the same draw.
BASE_SEEDS = (1, 2, 3)
#: the trainer's seed, so the same export trains the same adapter twice.
TRAIN_SEED = 0


def plan(n_rows: int, steps: int = STEPS) -> dict[str, Any]:
    """The keyword arguments the trainer gets, as plain dicts, so the smoke
    run can print them without importing TRL."""
    return {
        "sft": {
            "max_steps": steps,
            "learning_rate": LEARNING_RATE,
            "per_device_train_batch_size": BATCH_ROWS,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "max_length": MAX_LENGTH,
            "bf16": True,
            "logging_steps": 1,
            "save_strategy": "no",
            "report_to": [],
            "gradient_checkpointing": False,
            "seed": TRAIN_SEED,
        },
        "lora": {
            "r": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": LORA_TARGETS,
        },
        "sampling": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "samples": SAMPLES,
            "batch": EVAL_BATCH,
        },
        "epochs": round(steps * ROWS_PER_STEP / max(n_rows, 1), 1),
    }


# -- the report ----------------------------------------------------------------


def report(rows: list[dict]) -> str:
    """Lesson 7 step 3 on the rows the Modal run wrote: grade both arms with
    the judge, the noise floor from the base passes, the paired comparison."""
    import whileai as wai

    for r in rows:
        r.update(judge(r))
    base = [
        [r for r in rows if r["model_version"] == "base" and r["run"] == run]
        for run in range(1, BASE_PASSES + 1)
    ]
    trained = [r for r in rows if r["model_version"] == "trained"]
    noise = wai.simulations.eval_variance(*base)
    lines = [
        f"before: {wai.pass_at(base[0])}",
        f"after:  {wai.pass_at(trained)}",
        f"base passes: {' / '.join(f'{m:.3f}' for m in noise['means'].values())}  run_std {noise['run_std']:.3f}",
        str(wai.compare(base[0], trained, run_std=noise["run_std"], run_std_runs=noise["n_runs"])),
    ]
    return "\n".join(lines)


# -- the smoke run -------------------------------------------------------------


def smoke() -> int:
    import whileai as wai

    @wai.tool
    def get_order(order_id: str) -> dict:
        """Look up an order by id."""
        ...

    data = wai.simulate(
        wai.seeded_agent([get_order]),
        tools=[get_order],
        system_prompt="Help customers with orders.",
        simulator=False,
        mode="rl",
        repeats=4,
        repeat_policy="fixed",
        budget=64,
        seed=0,
    )
    scored = data.grade(judge=judge)
    tasks = sorted({r["scenario_id"] for r in scored.rows})
    held = set(tasks[-4:])
    with tempfile.TemporaryDirectory() as tmp:
        train_path, holdout_path = Path(tmp) / "train.jsonl", Path(tmp) / "holdout.jsonl"
        train = [r for r in scored.rows if r["scenario_id"] not in held]
        written = wai.select(train, mode="sft").export(
            str(train_path), system_prompt="Help customers with orders.", tools=[get_order.schema]
        )
        with holdout_path.open("w", encoding="utf-8") as fh:
            for r in scored.rows:
                if r["scenario_id"] in held:
                    fh.write(
                        json.dumps({"scenario_id": r["scenario_id"], "messages": r["messages"]})
                        + "\n"
                    )
        export = load_export(train_path)
        holdout = load_holdout(holdout_path)
    rows = export["rows"]
    assert written["n"] == len(rows), (written["n"], len(rows))
    assert export["system_prompt"] and export["tools"], (
        "the export carries no system prompt or tools"
    )
    calls = [c for r in rows for m in r["messages"] for c in m.get("tool_calls") or []]
    assert calls and all(isinstance(c["arguments"], dict) for c in calls), (
        "tool calls not normalised"
    )
    assert holdout and all(t["tool_result"] for t in holdout), holdout[:1]
    print(
        f"export: {len(rows)} rows, {len(export['tools'])} tool(s), system prompt {len(export['system_prompt'])} chars"
    )
    print(f"holdout: {len(holdout)} asks over {len({t['scenario_id'] for t in holdout})} tasks")
    print("plan:", json.dumps(plan(len(rows), SMOKE_STEPS), indent=1))

    # A row no stand-in wrote: no ``seeded`` field, a system message first,
    # the shape train_modal.py builds from a real model's output.
    real = {
        "messages": [
            {"role": "system", "content": "Help customers with orders."},
            {"role": "user", "content": "Where is ORD-1234? It was due yesterday."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "get_order", "arguments": {"order_id": "ORD-1234"}}],
            },
            {"role": "tool", "name": "get_order", "content": json.dumps({"status": "timeout"})},
            {
                "role": "assistant",
                "content": "The lookup for ORD-1234 timed out, so I could not confirm it.",
            },
        ]
    }
    assert judge(real) == {"reward": 1}, judge(real)
    claimed = json.loads(json.dumps(real))
    claimed["messages"][-1]["content"] = "ORD-1234 is done and went through."
    assert judge(claimed) == {"reward": 0}, judge(claimed)
    guessed = json.loads(json.dumps(real))
    guessed["messages"][1]["content"] = "Where is my order? It was due yesterday."
    assert judge(guessed) == {"reward": 0}, judge(guessed)
    print("judge: real-model row 1, claimed success 0, guessed id 0")
    print("smoke ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    import sys

    from whileai.config import provenance

    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--smoke", action="store_true", help="no key, no GPU, no network: check the wiring"
    )
    p.add_argument(
        "--report", default="", help="holdout_rows.jsonl from a Modal run: print lesson 7 step 3"
    )
    args = p.parse_args(argv)
    if args.report:
        rows = [
            json.loads(line)
            for line in Path(args.report).read_text(encoding="utf-8").splitlines()
            if line
        ]
        print(report(rows))
        return 0
    if args.smoke:
        return smoke()
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
