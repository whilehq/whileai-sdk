"""Did my prompt rewrite really help? Run both prompts on one model, then
``wai.compare`` the two arms.

    import whileai as wai

    report = wai.harness.compare(
        "You are a helpful assistant.",                     # before: the prompt you ship
        "Answer with the final number only.",               # after: the rewrite
        tasks=[{"prompt": "What is 7 * 6?", "reference": "42"}, ...],
        reward=wai.verify.Numeric(),
        model=wai.Ollama("qwen3:4b-instruct"),
    )
    print(report)          # seed, both pass@1 lines, then the compare block and its verdict

The same check from a terminal, free and offline with the scripted demo
model, is ``wai compare --demo``; with your own model it is
``wai compare --model ollama:qwen3:4b-instruct --before old.txt --after new.txt
--tasks tasks.jsonl --reward Numeric``.

Mechanism. Each task is sent ``k`` times to each arm, the arm's prompt as
the system message and the task as the user message, and the whole eval
runs ``runs`` times. Draw ``j`` of task ``t`` in run ``r`` carries the
same sampling seed on both arms (common random numbers: the sampler's own
luck is shared, so the paired difference is the prompt's and not the
dice's). The replies become rows through ``wai.rows`` with the reward you
name, each stamped with its ``lineage.eval_run`` the way
``simulate(runs=N)`` stamps it, and the two row sets go to
``wai.compare`` unchanged: the paired task differences, the bootstrap
interval, the re-run noise floor and the verdict are that call's, not
this module's. Nothing here computes a statistic.

Reference: Lambert 2025 (rlhfbook.com), chapter Evaluation, on comparing
two configurations on one fixed task set; Miller 2024 (arXiv:2411.00640),
"Adding Error Bars to Evals", for paired differences over tasks and for
resampling each task more than once. The external case study this
packages ran the same shape on qwen3:4b-instruct through Ollama on one
laptop (gentlyventures.com/casestudies/whileai, whileai 0.126).
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .harness import Harness
from .report import Report

#: K = 4: replies per task per arm. Four is the fewest the package reads
#: pass^k and pass@k from (``min_k`` in ``score.passat``), and more than
#: one reply per task is what lets the interval see the model's own
#: sampling spread (Miller 2024, arXiv:2411.00640, section 3.2). On a
#: laptop the cost is linear in k; raise it when the interval is too wide.
K = 4
#: SEED = 0: the base seed every per-draw sampling seed is derived from,
#: the same default ``wai.compare`` resamples its bootstrap with. Printed
#: on every report so a rerun with the same value reproduces it.
SEED = 0
#: RUNS = 3: times the whole eval runs per arm, ``MIN_RERUNS``: the fewest
#: re-runs a sample sd has two degrees of freedom from, so ``wai.compare``
#: can measure the eval's own noise floor and call a gain past it PASS. One
#: run is one draw of the eval, and ``wai.compare`` reads a gain on one run
#: as INCONCLUSIVE (Lambert 2025, chapter Evaluation).
RUNS = 3
#: TEMPERATURE = 0.7: the sampling temperature an arm runs at unless its
#: harness says otherwise (``Disclosure(sampling={"temperature": t})``).
#: Qwen3's model card recommends 0.7 for its non-thinking (instruct) mode,
#: and it is the temperature ``complete()`` uses when a caller names none.
#: Temperature 0 would make the k replies identical and k pointless.
TEMPERATURE = 0.7
#: MAX_TOKENS = 1024: the reply budget per draw, ``COMPLETE_MAX_TOKENS``,
#: enough for a short worked answer from an instruct model (convention,
#: untested).
MAX_TOKENS = 1024
#: SEED_SPACE = 2**31: per-draw seeds are drawn below this, the range
#: every OpenAI-compatible server (Ollama, vLLM) accepts as a signed int32.
SEED_SPACE = 2**31
#: TARGET = "pass_at_1": the metric whose verdict heads the report. A
#: prompt rewrite is meant to move the pass rate, so that is the headline
#: ``wai.compare`` prints with its gain and interval.
TARGET = "pass_at_1"

Chat = Callable[[list[dict], int], str]


def _chat(model: Any, temperature: float) -> Chat:
    """A ``(messages, seed) -> text`` call from a backend object, a
    ``provider:model`` spec string, or such a callable already."""
    from .swarm import _chat_for

    if isinstance(model, str):
        from .simulations.generate.agents import complete, parse_backend_spec

        base_url, name = parse_backend_spec(model)

        def chat(messages: list[dict], seed: int) -> str:
            reply = complete(
                base_url,
                name,
                messages,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                extra={"seed": seed},
            )
            return str(reply.get("content") or "")

        return chat
    return _chat_for(model, temperature, MAX_TOKENS)


def _identity(model: Any) -> Any:
    """What a harness fingerprints a model by. A plain callable has no
    spec, and its repr carries a memory address that changes every run,
    so it is named by its qualified name instead."""
    if model is None or isinstance(model, str) or getattr(model, "spec", None) is not None:
        return model
    if callable(model) and not hasattr(model, "model"):
        return f"callable:{getattr(model, '__qualname__', type(model).__qualname__)}"
    return model


def _arm(side: Any, label: str, model: Any) -> tuple[Harness, Any]:
    """One arm as a ``Harness`` and the model its prompts run on. A prompt
    string becomes a prompted harness on ``model``; a harness is kept,
    borrowing ``model`` if it has none."""
    if isinstance(side, Harness):
        if side.kind == "prompted" and side.model is None:
            named = Harness(
                _identity(model),
                instructions=side.instructions,
                label=side.label or label,
                disclosure=side.disclosure,
            )
            return named, model
        return side, side.model
    if side is None or isinstance(side, str):
        return Harness(_identity(model), instructions=side, label=label), model
    raise TypeError(
        f"{label}= must be a system prompt (str) or a wai.Harness; got {type(side).__name__}"
    )


def _model_name(model: Any) -> str:
    if model is None:
        return "-"
    if isinstance(model, str):
        return model.removeprefix("callable:")
    spec = getattr(model, "spec", None)
    if spec:
        return str(spec)
    return str(getattr(model, "model", None) or type(model).__name__)


def _tasks(tasks: Any) -> tuple[list[str], list[Any] | None, list[str] | None]:
    """Prompts, references and ids from a list of strings, a list of
    ``{"prompt", "reference", "id"}`` dicts, or a JSONL path of those."""
    if isinstance(tasks, (str, Path)):
        path = Path(tasks)
        lines = path.read_text(encoding="utf-8").splitlines()
        tasks = [json.loads(line) for line in lines if line.strip()]
    items = list(tasks)
    if not items:
        raise ValueError("tasks= is empty; pass at least one prompt")
    prompts: list[str] = []
    refs: list[Any] = []
    ids: list[str] = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            prompts.append(item)
            refs.append(None)
            ids.append("")
            continue
        if not isinstance(item, dict):
            raise TypeError(f"tasks[{i}] must be a str or a dict; got {type(item).__name__}")
        prompt = item.get("prompt", item.get("question"))
        if prompt is None:
            raise ValueError(f"tasks[{i}] has no 'prompt' (or 'question') key: {sorted(item)}")
        prompts.append(str(prompt))
        refs.append(item.get("reference", item.get("answer")))
        ids.append(str(item.get("id", item.get("task_id", "")) or ""))
    references = refs if any(r is not None for r in refs) else None
    task_ids = ids if all(ids) else None
    return prompts, references, task_ids


def _draw_seeds(prompts: Sequence[str], k: int, seed: int, run: int) -> list[list[int]]:
    """One sampling seed per (task, draw) in one run, the same on both arms."""
    return [
        [random.Random(f"{seed}:{run}:{prompt}:{j}").randrange(SEED_SPACE) for j in range(k)]
        for prompt in prompts
    ]


def _temperature(harness: Harness) -> float:
    """The arm's sampling temperature: its harness's, else ``TEMPERATURE``."""
    sampling = harness.disclosure.sampling or {}
    return float(sampling.get("temperature", TEMPERATURE))


def _replies(
    harness: Harness,
    model: Any,
    prompts: Sequence[str],
    seeds: list[list[int]],
    temperature: float,
) -> list[list[str]]:
    if harness.kind != "prompted":
        # a callable or command harness owns its own sampling; it is called
        # k times per task and its seeding is its own
        return [
            [str(harness(p).get("final_text") or "") for _ in row] for p, row in zip(prompts, seeds)
        ]
    if model is None:
        raise ValueError(
            "no model to run the prompts on: pass model=wai.Ollama('qwen3:4b-instruct') "
            "(or any backend, spec string, or (messages, seed) -> str callable)"
        )
    chat = _chat(model, temperature)
    system = [{"role": "system", "content": harness.instructions}] if harness.instructions else []
    return [
        [chat([*system, {"role": "user", "content": p}], s) for s in row]
        for p, row in zip(prompts, seeds)
    ]


class BeforeAfter(Report):
    """What ``wai.harness.compare`` ran and what ``wai.compare`` said.

    A dict: ``seed``, ``k``, ``runs``, ``temperature``, ``model``,
    ``n_tasks``, ``before`` and ``after`` (each ``{label, hash, model,
    temperature, pass_at_1, ci95}``, pass@1 over every run),
    ``verdict`` (the word the compare block prints: ``PASS``, ``NO
    DIFFERENCE``, ``FAIL``), ``headline_verdict``, ``ok``, and
    ``compare``, the ``DeltaReport`` itself. ``before_rows`` and
    ``after_rows`` are the graded rows, as attributes, for
    ``wai.pass_at``, ``wai.select`` or ``wai.harness.attribute``.
    ``print(report)`` writes the setup (with the seed to rerun it), one
    pass@1 line per arm, then the compare block exactly as ``wai.compare``
    prints it.
    """

    _summary_keys = ("verdict", "seed")
    before_rows: list[dict]
    after_rows: list[dict]

    def __str__(self) -> str:
        lines = [
            f"before-and-after check: {self['n_tasks']} tasks x {self['k']} replies x "
            f"{self['runs']} runs per arm, model {self['model']}, temperature "
            f"{self['temperature']}, seed {self['seed']} (same seed, same output)"
        ]
        for side in ("before", "after"):
            arm = self[side]
            named = f"  ({arm['label']})" if arm["label"] != side else ""
            lines.append(f"  {side:<6} {arm['hash']}  {arm['pass_at']}{named}")
        lines.append("")
        lines.append(str(self["compare"]))
        return "\n".join(lines)


def compare(
    before: str | Harness | None,
    after: str | Harness | None,
    tasks: Any,
    reward: Any,
    *,
    model: Any = None,
    k: int = K,
    runs: int = RUNS,
    seed: int = SEED,
) -> BeforeAfter:
    """Run an old and a new prompt (or two harnesses) on the same tasks
    and say whether the new one is really better.

    Reach for it before you ship a prompt rewrite. It runs locally on
    whatever model you point it at (``wai.Ollama("qwen3:4b-instruct")``
    needs no key and no credits), and returns a ``BeforeAfter`` that
    prints the setup, both pass rates, and the verdict ``wai.compare``
    gives: the gain, its 95% interval over tasks, and ``PASS`` only when
    the interval supports it.

    * ``before`` / ``after``: the system prompt each arm runs with (a
      string, or ``None`` for no system prompt), or a ``wai.Harness`` to
      compare two whole configurations (two models, two instruction sets,
      two temperatures via ``Disclosure(sampling={"temperature": t})``;
      ``TEMPERATURE`` = 0.7 otherwise, Qwen3's recommended value for its
      instruct mode). A prompted harness with no model borrows ``model=``.
    * ``tasks``: the test set, the same for both arms. A list of prompts,
      a list of ``{"prompt", "reference", "id"}`` dicts (``question`` and
      ``answer`` are read too), or a path to a JSONL file of those.
    * ``reward``: the scorer, as ``wai.rows`` takes it: a verifier
      (``wai.verify.Numeric()``), a ``(prompt, completion[, reference])
      -> number`` callable, or a judge ``(row) -> verdict``.
    * ``model``: where the prompts run: a backend object
      (``wai.Ollama(...)``, ``wai.Endpoint(...)``), a spec string
      (``"ollama:qwen3:4b-instruct"``), or a ``(messages, seed) -> str``
      callable, which is how the tests script a fake model. No default:
      a check that silently ran on a hosted model would spend credits.
    * ``k`` (``K`` = 4): replies per task per arm in each run.
    * ``runs`` (``RUNS`` = 3): times the whole eval runs per arm, so
      ``wai.compare`` measures the eval's own noise and can say PASS. With
      ``runs=1`` a gain reads INCONCLUSIVE, because one run is one draw.
      Cost is ``2 * runs * k * len(tasks)`` model calls.
    * ``seed`` (``SEED`` = 0): every per-draw sampling seed and the
      bootstrap are derived from it. Draw ``j`` of a task in run ``r``
      gets the same seed on both arms. Same seed, same model, same
      output; the report prints it.

    Nothing here does statistics: the rows go to ``wai.compare(before,
    after, target="pass_at_1", seed=seed)`` unchanged, and its
    ``DeltaReport`` is ``report["compare"]``.

    Reference: Lambert 2025, chapter Evaluation; Miller 2024,
    arXiv:2411.00640.
    """
    from .simulations.schema import rows
    from .simulations.score.delta import delta_report, headline_word
    from .simulations.score.passat import pass_at

    if k < 1 or runs < 1:
        raise ValueError(f"k and runs must be at least 1, got k={k}, runs={runs}")
    prompts, references, task_ids = _tasks(tasks)
    arms = {"before": _arm(before, "before", model), "after": _arm(after, "after", model)}
    graded: dict[str, list[dict]] = {}
    summary: dict[str, dict[str, Any]] = {}
    for side, (harness, runs_on) in arms.items():
        out: list[dict] = []
        for run in range(runs):
            seeds = _draw_seeds(prompts, k, seed, run)
            replies = _replies(harness, runs_on, prompts, seeds, _temperature(harness))
            one = list(rows(prompts, replies, reward, references=references, task_ids=task_ids))
            for row in one:
                row.setdefault("lineage", {})["eval_run"] = f"run-{run}"
            out.extend(one)
        harness.stamp_rows(out)
        graded[side] = out
        passed = pass_at(out)
        summary[side] = {
            "label": harness.version,
            "hash": harness.fingerprint,
            "model": harness.model_name,
            "temperature": _temperature(harness),
            "pass_at_1": passed.pass_at_1,
            "ci95": passed.ci95,
            "pass_at": str(passed).split(" | ")[0],
        }
    delta = delta_report(graded["before"], graded["after"], target=TARGET, seed=seed)
    models = {_model_name(h.model) for h, _ in arms.values()}
    temps = sorted({summary[side]["temperature"] for side in arms})
    report = BeforeAfter(
        seed=seed,
        k=k,
        runs=runs,
        temperature=temps[0] if len(temps) == 1 else " vs ".join(map(str, temps)),
        model=" vs ".join(sorted(models)) if len(models) > 1 else models.pop(),
        n_tasks=len(prompts),
        before=summary["before"],
        after=summary["after"],
        verdict=headline_word(delta),
        headline_verdict=delta.get("headline_verdict"),
        ok=delta.get("ok"),
        compare=delta,
    )
    report.before_rows = graded["before"]
    report.after_rows = graded["after"]
    return report


# --------------------------------------------------------------------------
# The demo: a scripted model, so the first command runs with nothing installed
# --------------------------------------------------------------------------

#: DEMO_BEFORE / DEMO_AFTER: the prompt pair ``wai compare --demo`` runs.
DEMO_BEFORE = "You are a helpful assistant."
DEMO_AFTER = "You are a careful calculator. Work it out, then end with the final number only."
#: DEMO_SKILL: the scripted model's chance of the right number under each
#: prompt, on a task of middle difficulty. Round numbers for a demo, not
#: measurements: the rewrite is planted to help, so the check has a real
#: gain to find.
DEMO_SKILL = {DEMO_BEFORE: 0.5, DEMO_AFTER: 0.8}
#: DEMO_TASKS = 24: arithmetic questions in the demo set, enough tasks for
#: the planted 30-point gain to clear its interval at k=4.
DEMO_TASKS = 24


def demo_tasks(n: int = DEMO_TASKS) -> list[dict[str, str]]:
    """``n`` arithmetic questions with their answers, the same every call."""
    rng = random.Random("whileai-demo-tasks")
    out = []
    for i in range(n):
        a, b = rng.randrange(12, 99), rng.randrange(3, 19)
        out.append(
            {"id": f"arith-{i:02d}", "prompt": f"What is {a} * {b}?", "reference": str(a * b)}
        )
    return out


def demo_model(messages: list[dict], seed: int) -> str:
    """A scripted stand-in for a small local model: right at the rate
    ``DEMO_SKILL`` gives the system prompt, adjusted by a per-question
    difficulty, and otherwise a near miss. Deterministic in the question
    and the seed, so the same seed gives the same replies."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    a, b = (int(x) for x in question.rstrip("?").split("is ")[-1].split(" * "))
    hard = random.Random(f"difficulty:{question}").random()
    p = min(1.0, max(0.0, DEMO_SKILL.get(system, 0.5) + 0.4 * (0.5 - hard)))
    rng = random.Random(f"{question}:{seed}")
    right = rng.random() < p
    number = a * b if right else a * b + rng.choice([-10, -1, 1, 2, 10])
    return f"{a} times {b}: multiply the tens, then the ones.\nThe answer is {number}"


__all__ = [
    "RUNS",
    "SEED",
    "TEMPERATURE",
    "BeforeAfter",
    "K",
    "compare",
    "demo_model",
    "demo_tasks",
]
