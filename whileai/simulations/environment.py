"""Export a simulation as an RL environment a trainer can install and drive.

A dataset is rollouts; an environment is what produces them. On-policy RL
(GRPO, RLOO, PPO) samples its own rollouts from the policy under training, so
what it needs from us is not rows but the three things a row came from: the
task set, the world that answers tool calls, and the reward that grades the
finished trajectory (Lambert 2025, chapter Reinforcement Learning on on-policy
sampling and chapter Reasoning on multi-turn tool use with a single
end-of-trajectory reward). ``export_environment`` writes those three as an
installable ``verifiers`` package, the shape Prime Intellect and TRL consume::

    import whileai.simulations as wai
    data = wai.simulate(tools=my_tools, system_prompt=my_policy, mode="rl", repeats=8)
    scored = data.grade()
    wai.export_environment(scored, "envs/my-agent", reward=my_verifier)

    # then, with verifiers installed:
    #   pip install -e envs/my-agent
    #   vf-eval my_agent -a '{"split": "holdout"}' -m <policy> ...

What goes in the package:

* ``spec.json``: the system prompt, the tool schemas verbatim, the turn
  cap, dotted references to the reward and the world, and, when
  ``harnesses=`` was given, the harnesses a rollout may run under (label,
  hash, instructions, tool schemas, disclosure).
* ``data/train.jsonl`` / ``data/holdout.jsonl``: one task per prompt in
  the verifiers shape (``prompt``, ``info``, ``example_id``). ``info``
  carries the task's fault plan, world state, privileged reference and
  calibration. It is read by the world and the reward on the server and
  never enters the prompt, so a training file cannot leak the answer key.
* ``README.md``: the gate. Task counts, the difficulty band applied when
  the rows were graded, and the train-against-holdout decontamination.

The environment class itself lives here, not in the package, so it is
tested once: a ``StatefulToolEnv`` whose world is the SDK's mock world
seeded per task (or the caller's ``execute=``), whose tools are the spec's
schemas, and whose rubric is the reward through the SDK judge contract.
A ``Verifier`` (``CodeExec``, ``MathEqual``, ...), a judge callable, or
``conduct_grade`` all work unchanged. ``verifiers`` is imported lazily;
export needs nothing but the SDK.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .defaults import (
    DIFFICULTY_BAND,
    ENV_DECONTAMINATION_EXAMPLES,
    ENV_DECONTAMINATION_NGRAM,
    ENV_EVAL_EXAMPLES,
    ENV_EVAL_ROLLOUTS,
    ENV_HARNESS_MIX,
    ENV_HARNESS_SEED,
    ENV_HOLDOUT_FRACTION,
    ENV_MAX_TURNS_FALLBACK,
    ENV_RUNTIME,
    ENV_RUNTIMES,
)
from .export import _resolve
from .score.checklist import _task_has_outcome_rule
from .score.judging import normalize_judge_result
from .score.stats import decontaminate
from .tools import schemas as _tool_schemas

SPEC_FILE = "spec.json"
DEFAULT_REWARD = "whileai.simulations.score.checklist:task_checklist"
#: The difficulty band build_tasks() keeps by default, the package-wide one.
DEFAULT_BAND = DIFFICULTY_BAND
#: Rubric weights, in the order the funcs are listed: only ``reward`` trains;
#: ``n_calls``, ``judge_ok``, ``truncated`` and ``trace_clean`` are logged at
#: weight 0 as monitors. Training on a symptom of over-optimization turns
#: it into a proxy the policy games (Gao et al., arXiv:2210.10760;
#: Lambert 2025, chapter Over-optimization, lists the symptoms); neither
#: source prescribes weight-0 logging, that is this package's choice.
RUBRIC_WEIGHTS = (1.0, 0.0, 0.0, 0.0, 0.0)
#: The version an export claims when the package is not installed as a
#: distribution: the first release that carried this module.
_FIRST_ENV_RELEASE = "0.42"
_TASK_META = (
    "scenario_dimensions",
    "stance",
    "tier",
    "history",
    "tool_condition",
    "ask_family",
    "intent_known",
    "tool_known",
)
_MODULE_NAME = re.compile(r"[^a-z0-9_]+")

__all__ = [
    "DEFAULT_BAND",
    "DEFAULT_REWARD",
    "build_tasks",
    "export_environment",
    "load_environment",
    "resolve_ref",
]


# --------------------------------------------------------------------------
# References: a reward or a world is code, and the package names it
# --------------------------------------------------------------------------


def _ref_of(obj: Any) -> str:
    """``module:qualname`` for something a trainer process can import."""
    if isinstance(obj, str):
        if ":" not in obj:
            raise ValueError(f"expected 'module:attr', got {obj!r}")
        return obj
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None) or getattr(type(obj), "__qualname__", None)
    if not isinstance(obj, type) and not callable(obj):
        raise ValueError(f"{obj!r} is not callable")
    # An instance (a Verifier) is referenced by the module-level name it is
    # bound to. Its ``__module__`` is the SDK class's, not the caller's, so
    # look where the caller would have bound it: the module of the function
    # it wraps, then every loaded module (#374).
    if not isinstance(obj, type) and not hasattr(obj, "__name__"):
        bound = _bound_name(obj)
        if bound:
            return bound
    if not module or not qualname or "<locals>" in str(qualname) or module == "__main__":
        raise ValueError(
            "the reward and the world must be importable by name in the trainer "
            "process: pass 'module:attr' (or a module-level function or Verifier "
            "instance defined in an importable module), not a lambda or a local"
        )
    if not isinstance(obj, type) and not hasattr(obj, "__name__"):
        # No name to import it by, so the trainer would build a bare
        # instance. That is only the same object when this one carries no
        # configuration a bare one lacks: CodeExec(tests=...) written inline
        # would otherwise reload as CodeExec() and score with no tests.
        cls = type(obj)
        try:
            bare: Any = cls()
        except Exception:
            bare = None
        if bare is None or getattr(bare, "__dict__", None) != getattr(obj, "__dict__", None):
            raise ValueError(
                f"{cls.__qualname__} instance is configured but not bound to a "
                f"module-level name in {module}, so the trainer could only rebuild "
                f"a bare {cls.__qualname__}(): assign it a name in an importable "
                "module and pass that, or pass 'module:attr'"
            )
        return f"{module}:{cls.__qualname__}"
    return f"{module}:{qualname}"


def _bound_name(obj: Any) -> str | None:
    """``module:name`` where a module binds ``obj`` at top level, or None.

    Tries the module of the function the instance wraps first, then every
    loaded module outside this package. A binding in ``__main__`` is named
    by the script's file stem, which the trainer process imports when the
    script's directory is on its path; the name is what a trainer would
    write by hand anyway.
    """
    wrapped = None
    for attr in ("_fn", "fn", "func", "__wrapped__"):
        wrapped = getattr(obj, attr, None)
        if wrapped is not None:
            break
    candidates: list[str] = []
    if wrapped is not None and getattr(wrapped, "__module__", None):
        candidates.append(str(wrapped.__module__))
    candidates.extend(
        name
        for name in list(sys.modules)
        if name not in candidates and not name.startswith(("whileai", "_", "importlib"))
    )
    for mod_name in candidates:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        try:
            names = vars(mod)
        except TypeError:
            continue
        for name, value in list(names.items()):
            if value is obj and not name.startswith("_"):
                if mod_name == "__main__":
                    file = getattr(mod, "__file__", None)
                    if not file:
                        return None
                    return f"{Path(file).stem}:{name}"
                return f"{mod_name}:{name}"
    return None


def resolve_ref(ref: str) -> Any:
    """Import ``module:attr``. A class is instantiated with no arguments."""
    module_name, _, attr = str(ref).partition(":")
    if not module_name or not attr:
        raise ValueError(f"bad reference {ref!r}; expected 'module:attr'")
    target: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        target = getattr(target, part)
    if isinstance(target, type):
        target = target()
    return target


# --------------------------------------------------------------------------
# Tasks: one per prompt, with what the world and the reward need
# --------------------------------------------------------------------------


def _tool_defs(tools: Sequence[dict] | None) -> list[dict]:
    """verifiers-shaped tool definitions from OpenAI-shaped (or flat) schemas."""
    out: list[dict] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        fn: dict = function if isinstance(function, dict) else tool
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        params = (
            fn.get("parameters") or fn.get("input_schema") or {"type": "object", "properties": {}}
        )
        out.append(
            {
                "name": name,
                "description": str(fn.get("description") or ""),
                "parameters": params,
            }
        )
    return out


def _harness_entries(harnesses: Sequence[Any] | None, fallback_tools: list[dict]) -> list[dict]:
    """The spec's ``harnesses`` list: one JSON entry per ``wai.Harness`` or
    per plain ``{label, instructions, tools}`` dict, hashed the way the
    ``Harness`` fingerprint hashes, so the label and hash a rollout records
    match the ones ``simulate(harness)`` stamps on rows. A harness with no
    tools of its own runs with the environment's tools."""
    if not harnesses:
        return []
    from ..harness import Disclosure, Harness

    entries: list[dict] = []
    for i, item in enumerate(harnesses):
        if isinstance(item, Mapping):
            disclosure = item.get("disclosure")
            harness = Harness(
                item.get("model"),
                instructions=item.get("instructions"),
                tools=item.get("tools") or [],
                label=item.get("label"),
                disclosure=Disclosure(**dict(disclosure))
                if isinstance(disclosure, Mapping)
                else None,
            )
        elif hasattr(item, "stamp") and hasattr(item, "tool_schemas"):
            harness = item
        else:
            raise TypeError(
                f"harnesses[{i}] must be a wai.Harness or a dict with label, instructions and "
                f"tools; got {type(item).__name__}"
            )
        tools = _tool_defs(harness.tool_schemas()) or list(fallback_tools)
        entries.append(
            {
                "label": harness.version,
                "hash": harness.fingerprint,
                "instructions": harness.instructions or "",
                "tools": tools,
                "disclosure": harness.disclosure.items(),
            }
        )
    labels = [e["label"] for e in entries]
    dupes = sorted({x for x in labels if labels.count(x) > 1})
    if dupes:
        raise ValueError(
            f"harnesses= carries the same label twice ({', '.join(dupes)}); give each one its "
            "own label= so a rollout's trace says which harness it ran under"
        )
    return entries


def _harness_weights(mix: Any, n: int) -> list[float]:
    """Normalized draw weights over ``n`` harnesses from ``harness_mix``:
    ``"uniform"`` or a list of ``n`` non-negative numbers."""
    if isinstance(mix, str):
        if mix != ENV_HARNESS_MIX:
            raise ValueError(
                f"harness_mix= must be {ENV_HARNESS_MIX!r} or a list of {n} weights; got {mix!r}"
            )
        return [1.0 / n] * n if n else []
    weights = [float(w) for w in mix]
    if len(weights) != n or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(
            f"harness_mix= needs one non-negative weight per harness ({n}) with a positive "
            f"sum; got {list(mix)!r}"
        )
    total = sum(weights)
    return [w / total for w in weights]


def _draw_index(key: str, seed: int, weights: Sequence[float]) -> int:
    """The harness a task runs under: a sha256 of the seed and the task id
    read as a point on [0, 1) against the cumulative weights. The same task
    and seed give the same harness on every machine and every re-run."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    point = int.from_bytes(digest[:8], "big") / float(1 << 64)
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if point < acc:
            return i
    return len(weights) - 1


def _label(value: Any) -> float | None:
    """A graded reward as it counts toward the prompt's solve rate: 0 and
    1 as they are, partial credit (the checklist's 0.5 when conduct is
    half) as it is, anything else (None, a bool, text) as ungraded."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return min(1.0, max(0.0, v)) if v == v else None


def build_tasks(
    rows: Sequence[dict],
    *,
    holdout: float | Sequence[str] = ENV_HOLDOUT_FRACTION,
    band: tuple[float, float] | None = DEFAULT_BAND,
    ngram: int = ENV_DECONTAMINATION_NGRAM,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """One task per distinct prompt, split into train and holdout.

    When a prompt has two or more graded rollouts its solve rate is known
    (partial credit counts as it is) and, with ``band``, prompts the policy
    always or never solved are dropped: they carry no advantage
    (Lambert 2025, chapter Reasoning, difficulty filtering at 20 to 80
    percent; DAPO's dynamic sampling drops accuracy 0 and 1,
    arXiv:2503.14476). Ungraded prompts and single
    rollouts are kept as they are. ``holdout`` is a fraction, split by
    scenario id (or the prompt) so a task is wholly on one side, or an
    explicit list of holdout prompts. Train and holdout are decontaminated
    against each other at ``ngram``-grams (8: the overlap size
    Lambert 2025, chapter Evaluation, found its contaminations with) and
    the report says what overlapped.
    """
    by_prompt: dict[str, list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("prompt") or "").strip()
        if prompt:
            by_prompt.setdefault(prompt, []).append(row)

    if isinstance(holdout, bool):
        raise ValueError("holdout is a fraction (0.2) or a list of holdout prompts")
    explicit: set[str] | None = None
    fraction = 0.0
    if isinstance(holdout, (int, float)):
        fraction = float(holdout)
    else:
        explicit = {" ".join(str(p).lower().split()) for p in holdout}

    tasks: list[dict] = []
    dropped_band = 0
    mixed = 0
    for prompt, members in by_prompt.items():
        first = members[0]
        scenario = str(first.get("scenario_id") or "")
        key = scenario or prompt
        # The id names the prompt; the split bucket below hashes the scenario
        # so every prompt drawn from one situation lands on the same side.
        example_id = hashlib.sha1(f"{scenario}\n{prompt}".encode()).hexdigest()[:12]
        labels = [v for v in (_label(m.get("reward")) for m in members) if v is not None]
        info: dict[str, Any] = {
            "task_id": example_id,
            "scenario_id": scenario or None,
            "seed": first.get("seed"),
            "world_state": first.get("world_state") or "",
            "faults": first.get("faults") or {},
        }
        # The situation's coordinates on the grid. The checklist reward reads
        # them to know which outcome the task can be checked against.
        for meta_key in _TASK_META:
            if first.get(meta_key) is not None:
                info[meta_key] = first[meta_key]
        privileged = first.get("privileged")
        if isinstance(privileged, dict) and privileged:
            info["privileged"] = privileged
        if len(labels) >= 2:  # noqa: PLR2004  # two graded rollouts before a solve rate exists
            rate = sum(labels) / len(labels)
            info["calibration"] = {"pass_rate": round(rate, 4), "n": len(labels)}
            if min(labels) < max(labels):
                mixed += 1
            if band is not None and not (band[0] <= rate <= band[1]):
                dropped_band += 1
                continue
        if explicit is not None:
            split = "holdout" if " ".join(prompt.lower().split()) in explicit else "train"
        else:
            bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
            split = "holdout" if bucket < fraction else "train"
        info["split"] = split
        tasks.append({"prompt": prompt, "info": info, "example_id": example_id})

    train = [t for t in tasks if t["info"]["split"] == "train"]
    held = [t for t in tasks if t["info"]["split"] == "holdout"]
    decon: dict[str, Any] = {}
    if train and held:
        _, decon_full = decontaminate(
            [{"prompt": t["prompt"], "final_text": ""} for t in train],
            [[{"prompt": t["prompt"], "final_text": ""} for t in held]],
            n=int(ngram),
        )
        decon = {
            k: decon_full[k] for k in ("n_contaminated", "contamination_rate") if k in decon_full
        }
        decon["ngram"] = int(ngram)
        decon["examples"] = [
            e.get("match") for e in decon_full.get("examples", [])[:ENV_DECONTAMINATION_EXAMPLES]
        ]
    report = {
        "prompts": len(by_prompt),
        "tasks": len(tasks),
        "train": len(train),
        "holdout": len(held),
        "graded_prompts": sum(1 for t in tasks if "calibration" in t["info"]) + dropped_band,
        "band": list(band) if band is not None else None,
        "band_dropped": dropped_band,
        # prompts the policy both solved and failed: the ones with an advantage
        "graded_mixed": mixed,
        "decontamination": decon,
    }
    return train, held, report


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

_PACKAGE_INIT = '''"""{name}: a While RL environment. See README.md."""

from pathlib import Path

from whileai.simulations.environment import load_environment as _load

SPEC = Path(__file__).resolve().parent / "spec.json"


def load_environment(**kwargs):
    return _load(SPEC, **kwargs)
'''

_PYPROJECT = """[project]
name = "{dist}"
description = "{description}"
tags = ["whileai", "agents", "tool-use", "train", "eval"]
version = "0.1.0"
requires-python = ">=3.11,<3.14"
dependencies = [
    "verifiers>=0.3.1",
    "whileai>={sdk_version}",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
# Exports often live inside a repo that ignores data files; the wheel carries them anyway.
ignore-vcs = true
include = ["{name}/**", "pyproject.toml", "README.md"]

[tool.hatch.build.targets.wheel]
packages = ["{name}"]

[tool.verifiers.eval]
num_examples = {eval_examples}
rollouts_per_example = {eval_rollouts}
"""


def _sdk_version() -> str:
    try:
        from importlib.metadata import version

        return version("whileai")
    except Exception:
        return _FIRST_ENV_RELEASE


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")


def _readme(name: str, spec: dict, report: dict) -> str:
    decon = report.get("decontamination") or {}
    dist = name.replace("_", "-")
    world = spec.get("execute") or "whileai mock world, seeded per task"
    lines = [
        f"# {dist}",
        "",
        "A While RL environment: the task set, the world that answers tool",
        "calls, and the reward that grades a finished trajectory, packaged for",
        "`verifiers`. The trainer samples its own rollouts from the policy under",
        "training, so nothing here is off-policy.",
        "",
        "### Overview",
        f"- **Environment ID**: `{dist}`",
        f"- **Short description**: {spec.get('system_prompt', '')[:160].strip() or 'tool-using agent'}",
        "- **Tags**: whileai, agents, tool-use, train, eval",
        "",
        "### Datasets",
        "- **Primary dataset(s)**: `data/train.jsonl`, `data/holdout.jsonl` (one task per prompt, written by the While simulator)",
        f"- **Split sizes**: {report['train']} train / {report['holdout']} holdout",
        "",
        "### Task",
        "- **Type**: multi-turn tool use",
        f"- **Tools**: {', '.join('`' + t['name'] + '`' for t in spec['tools'])}",
        f"- **Turn cap**: {spec['max_turns']}",
        f"- **World**: `{world}`",
        f"- **Rubric overview**: `reward` = `{spec['reward']}` through the While judge contract (weight 1.0); `n_calls`, `judge_ok` and per-tool call counts logged at weight 0",
    ]
    if report.get("band"):
        lines.append(
            f"- **Difficulty band**: {report['band'][0]:.0%} to {report['band'][1]:.0%} solve rate; "
            f"{report['band_dropped']} of {report['graded_prompts']} graded prompts dropped"
        )
    if decon:
        lines.append(
            f"- **Train vs holdout {decon.get('ngram', ENV_DECONTAMINATION_NGRAM)}-gram overlap**: "
            f"{decon.get('n_contaminated', 0)} tasks "
            f"({(decon.get('contamination_rate') or 0):.1%})"
        )
    if spec.get("harnesses"):
        lines += ["", "### Harnesses"]
        for h in spec["harnesses"]:
            cap = (h.get("disclosure") or {}).get("max_turns") or spec["max_turns"]
            lines.append(
                f"- **{h['label']}** (`{h['hash']}`): {len(h.get('tools') or [])} tools, "
                f"turn cap {cap}"
            )
        lines += [
            "",
            "Each task draws one of these from its id and a seed, and the rollout runs",
            "under that harness's instructions and tools; `harness = {label, hash}` in the",
            "rollout state says which. Kim et al. 2026 (arXiv:2606.25447): a policy trained",
            "under one fixed harness collapses when the tool environment shifts, and one",
            "trained across harnesses holds up out of distribution.",
        ]
    lines += [
        "",
        "### Quickstart",
        "",
        "```bash",
        f"prime eval run {dist}",
        f'vf-eval {name} -a \'{{"split": "holdout"}}\' -m <policy> -b <base url> -k <key var>',
        "```",
        "",
    ]
    for warning in report.get("warnings") or []:
        lines.append(f"**Warning.** {warning}")
        lines.append("")
    lines += [
        "`info` on every task carries its fault plan, world state, privileged",
        "reference and calibration. The world and the reward read it on the",
        "server; it never enters the prompt.",
        "",
    ]
    return "\n".join(lines)


def export_environment(
    source: Any,
    out: str | Path,
    *,
    name: str | None = None,
    reward: Any = None,
    execute: Any = None,
    system_prompt: str | None = None,
    tools: Sequence[dict] | None = None,
    holdout: float | Sequence[str] = ENV_HOLDOUT_FRACTION,
    band: tuple[float, float] | None = DEFAULT_BAND,
    max_turns: int | None = None,
    description: str = "",
    ngram: int = ENV_DECONTAMINATION_NGRAM,
    world: Mapping[str, Any] | None = None,
    harnesses: Sequence[Any] | None = None,
    runtime: str = ENV_RUNTIME,
) -> dict[str, Any]:
    """Write graded rows as an installable RL environment for an on-policy trainer.

    Reach for it when the next step is RL in a trainer that speaks
    verifiers (prime-rl and the like) and needs the tasks, the world and
    the reward as one package. It writes the package under ``out`` and
    returns the report, which is also the package README: ``path``,
    ``prompts``, ``tasks``, ``train`` and ``holdout`` counts,
    ``graded_prompts``, ``band``, ``band_dropped``, ``graded_mixed``
    (prompts the policy both solved and failed, the ones with an
    advantage) and ``decontamination``. ``warnings`` carries a line when
    ``graded_mixed`` is 0, whether because no prompt has two graded
    rollouts or because every graded prompt was unanimous: a grouped
    update on such tasks has zero advantage everywhere (#684).

    * ``source``: a ``SimulationData`` (system prompt and tools come from
      its profile), a row list, or a JSONL path; graded rows get the
      difficulty band, ungraded rows are exported as they are. For a list
      or path pass ``system_prompt`` and ``tools``.
    * ``reward``: a ``Verifier``, a judge callable honoring the SDK judge
      contract, or ``'module:attr'``; it must be importable in the trainer
      process. A ``@verifier`` or ``All([...])`` bound to a name in your
      own module is referenced by that name (a script run as ``__main__``
      by its file stem, so keep that directory on the trainer's path).
      With no reward the conduct grade is used and the report warns: it
      is a process reward, and a policy trained on it alone learns to
      call nothing (see recipes/03-select/prime-intellect-rl).
    * ``execute``: a live world ``(tool, arguments) -> result``; without
      it the SDK's mock world answers, seeded per task so every rollout
      of a task sees the same world.
    * ``world``: the mock world's dials as a dict (``WorldOptions``
      fields: ``search_hits``, ``exists_share``, ``default_fault_mode``,
      name pools, ...). It is written into ``spec.json`` and the trainer's
      world is built from it, so the world a policy trains against is the
      one the export says.
    * ``holdout`` (0.2): the share of tasks held out, split by scenario id
      (or the prompt) so a task is wholly on one side, or an explicit
      list of holdout prompts.
    * ``band`` (``(0.2, 0.8)``): the pass-rate band a graded prompt must
      sit in; prompts the policy always or never solved carry no
      advantage and are dropped (Lambert 2025, chapter Reasoning, difficulty
      filtering at 20 to 80 percent; DAPO's dynamic sampling, arXiv:2503.14476).
      ``None`` keeps them all.
    * ``ngram`` (8): the train-versus-holdout decontamination size, the
      overlap Lambert 2025, chapter Evaluation, found its contaminations with.
    * ``name``, ``description``, ``max_turns``: the package name, its
      README line, and the rollout turn cap (the SDK default when
      ``None``).
    * ``harnesses``: ``wai.Harness`` objects (or plain ``{label,
      instructions, tools}`` dicts) the trainer's rollouts run under.
      Each task draws one from its id and a seed, and that harness's
      instructions become the system prompt and its tool schemas the tool
      set for the rollout; a harness with no tools of its own uses the
      environment's. The spec lists them as ``{label, hash, instructions,
      tools, disclosure}`` and every rollout records ``harness = {label,
      hash}``. Kim et al. 2026 (arXiv:2606.25447): a policy trained under
      one fixed harness collapses when the tool environment shifts, and
      harness-aware post-training generalizes out of distribution. Left
      out, the spec is what it always was.
    * ``runtime`` (``"verifiers"``): the trainer contract the package
      speaks. ``"verifiers"`` is the Prime Intellect package prime-rl and
      TRL's verifiers path install with ``pip install -e``; ``"openenv"``
      is Meta PyTorch's OpenEnv (``reset``/``step``/``state`` over HTTP,
      the contract TRL, torchforge, SkyRL and Unsloth drive), written as
      an ``openenv.yaml`` package that ``uv run --project . server``
      serves, ``openenv build`` containerizes and ``openenv push`` puts
      on a Hugging Face Space. Same tasks, world and reward either way;
      see ``whileai.simulations.openenv``.

    ```python
    report = wai.export_environment(data, "envs/refunds", reward=my_verifier)
    print(report["train"], report["holdout"], report["path"])
    ```
    """
    if runtime not in ENV_RUNTIMES:
        raise ValueError(f"runtime= must be one of {list(ENV_RUNTIMES)}; got {runtime!r}")
    tools = _tool_schemas(tools)
    from .world.sandbox import WorldOptions

    world_options = dict(world) if world else None
    if world_options is not None:
        WorldOptions.coerce(world_options)  # a typo fails here, not in the trainer
        try:
            json.dumps(world_options)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "world= must be JSON (it is written into spec.json); pass callables such as "
                "fault_modes to load_environment(world=) instead"
            ) from exc
    rows, system, resolved_tools, _ = _resolve(source)
    if system_prompt is not None:
        system = str(system_prompt)
    if tools is not None:
        resolved_tools = list(tools)
    tool_defs = _tool_defs(resolved_tools)
    if not tool_defs:
        raise ValueError("an environment needs tools: pass tools= or a source with a profile")
    harness_entries = _harness_entries(harnesses, tool_defs)

    out_dir = Path(out)
    name = _MODULE_NAME.sub("_", (name or out_dir.name).lower()).strip("_") or "whileai_env"
    warnings: list[str] = []
    reward_ref = DEFAULT_REWARD if reward is None else _ref_of(reward)
    execute_ref = _ref_of(execute) if execute is not None else None

    train, held, report = build_tasks(rows, holdout=holdout, band=band, ngram=ngram)
    if reward is None:
        checkable = sum(1 for t in train + held if _task_has_outcome_rule(t["info"]))
        report["outcome_checkable"] = checkable
        if not checkable:
            warnings.append(
                "no task carries grid metadata (target tool, stance, world state, "
                "history), so the default reward reduces to conduct_grade, a process "
                "reward with no outcome term; a policy trained on it alone learns to "
                "call nothing. Simulate with the writer, or pass reward= (a Verifier "
                "or your judge) before training."
            )
        elif checkable < len(train + held):
            warnings.append(
                f"only {checkable} of {len(train + held)} tasks carry a checkable "
                "outcome; the rest are scored on conduct alone."
            )
    # select_for_rl refuses these rows with no_mixed_groups; an export that
    # said nothing would be the one path that ships them (Lambert 2025,
    # chapter Policy Gradients: a unanimous group has zero advantage).
    if not report["graded_mixed"]:
        if report["graded_prompts"]:
            warnings.append(
                f"no_mixed_groups: every one of the {report['graded_prompts']} graded prompts "
                "was solved or failed on every rollout, so group-relative advantages are zero "
                "everywhere and a run on these tasks trains nothing. Regrade with a stricter "
                "rubric or raise difficulty (fault_rate, harder asks) before training on this."
            )
        else:
            warnings.append(
                "no_mixed_groups: no prompt has two graded rollouts, so no solve rate is known, "
                "the difficulty band could not be applied, and the trainer starts blind to "
                "which tasks carry an advantage. Grade rows from a run with repeats= (or "
                "k>=2 rollouts per prompt) and export again."
            )
    if not train:
        raise ValueError("no train tasks: every prompt fell outside the band or into the holdout")
    if max_turns is None:
        from .generate.agents import default_max_turns

        max_turns = int(default_max_turns(n_tools=len(tool_defs)))
    spec = {
        "name": name,
        "system_prompt": system,
        "tools": tool_defs,
        "max_turns": int(max_turns),
        "reward": reward_ref,
        "execute": execute_ref,
        "sdk_version": _sdk_version(),
    }
    if world_options is not None:
        spec["world"] = world_options
    if harness_entries:
        spec["harnesses"] = harness_entries
        report["harnesses"] = [{"label": h["label"], "hash": h["hash"]} for h in harness_entries]
    report.update(
        {
            "name": name,
            "reward": reward_ref,
            "execute": execute_ref,
            "warnings": warnings,
            "runtime": runtime,
        }
    )

    if runtime == "openenv":
        from .openenv import write_package

        write_package(
            out_dir,
            name=name,
            spec=spec,
            train=train,
            held=held,
            report=report,
            description=description,
        )
        report["path"] = str(out_dir)
        return report

    pkg = out_dir / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(_PACKAGE_INIT.format(name=name), encoding="utf-8")
    (pkg / SPEC_FILE).write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8")
    _write_jsonl(pkg / "data" / "train.jsonl", train)
    _write_jsonl(pkg / "data" / "holdout.jsonl", held)
    (out_dir / "pyproject.toml").write_text(
        _PYPROJECT.format(
            dist=name.replace("_", "-"),
            name=name,
            description=description or f"While RL environment: {name}",
            sdk_version=spec["sdk_version"],
            eval_examples=ENV_EVAL_EXAMPLES,
            eval_rollouts=ENV_EVAL_ROLLOUTS,
        ),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(_readme(name, spec, report), encoding="utf-8")
    report["path"] = str(out_dir)
    return report


# --------------------------------------------------------------------------
# Load: the environment class, built when verifiers is present
# --------------------------------------------------------------------------


def _row_from_state(state: dict, info: dict) -> dict[str, Any]:
    """The SDK row shape, rebuilt from a verifiers rollout state."""
    prompt_msgs = state.get("prompt") or []
    user_text = ""
    for msg in prompt_msgs if isinstance(prompt_msgs, list) else []:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "user":
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            user_text = content if isinstance(content, str) else str(content)
    completion = state.get("completion") or []
    final_text = ""
    for msg in reversed(completion if isinstance(completion, list) else []):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "assistant":
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                final_text = content
                break
    row: dict[str, Any] = {
        "prompt": user_text,
        "steps": list(state.get("zp_steps") or []),
        "final_text": final_text,
        "scenario_id": info.get("scenario_id"),
        "world_state": info.get("world_state") or "",
        "faults": info.get("faults") or {},
    }
    if info.get("privileged"):
        row["privileged"] = info["privileged"]
    if state.get("harness"):
        row["harness"] = dict(state["harness"])
    for meta_key in _TASK_META:
        if info.get(meta_key) is not None:
            row[meta_key] = info[meta_key]
    return row


def _make_env_class() -> type:
    import verifiers as vf

    from .generate.agents import current_rollout
    from .world.sandbox import MockEnvironment, WorldOptions

    class WhileEnv(vf.StatefulToolEnv):
        """One SDK world per rollout; the spec's tools; the reward as the rubric."""

        def __init__(
            self,
            spec: dict,
            *,
            reward: Callable[[dict], Any],
            execute: Callable[[str, dict], Any] | None = None,
            world: WorldOptions | Mapping[str, Any] | None = None,
            harness_mix: Any = ENV_HARNESS_MIX,
            harness_seed: int = ENV_HARNESS_SEED,
            **kwargs: Any,
        ) -> None:
            self.spec = spec
            self.reward = reward
            self.execute = execute
            # the mock world's dials: the call wins, then the spec, then defaults
            self.world = WorldOptions.coerce(world if world is not None else spec.get("world"))
            self._tool_defs_raw = list(spec.get("tools") or [])
            # the harnesses a task may draw (Kim et al. 2026, arXiv:2606.25447);
            # the world answers every tool any of them carries
            self.harnesses: list[dict] = list(spec.get("harnesses") or [])
            self.harness_weights = _harness_weights(harness_mix, len(self.harnesses))
            self.harness_seed = int(harness_seed)
            self._world_tool_defs = list(self._tool_defs_raw)
            known = {t["name"] for t in self._world_tool_defs}
            for harness in self.harnesses:
                for tool in harness.get("tools") or []:
                    if tool["name"] not in known:
                        known.add(tool["name"])
                        self._world_tool_defs.append(tool)
            rubric = vf.Rubric(
                funcs=[
                    self.reward_func,
                    self.n_calls,
                    self.judge_ok,
                    self.truncated,
                    self.trace_clean,
                ],
                weights=list(RUBRIC_WEIGHTS),
            )
            kwargs.setdefault("rubric", rubric)
            super().__init__(
                tools=[],
                max_turns=int(spec.get("max_turns") or ENV_MAX_TURNS_FALLBACK),
                **kwargs,
            )
            self.tool_defs = self._normalize_tool_defs(self._tool_defs_raw)
            for tool in self._world_tool_defs:
                self.tool_monitor_rubric.add_tool_metric(tool["name"])

        def draw_harness(self, state: dict) -> dict | None:
            """The harness this rollout runs under, or ``None`` when the spec
            carries none: a hash of ``harness_seed`` and the task id, so
            every rollout of a task, on every re-run, draws the same one."""
            if not self.harnesses:
                return None
            info = state.get("info") or {}
            key = info.get("task_id") or state.get("src_id") or state.get("example_id")
            if key is None:
                key = _row_from_state(state, info)["prompt"]
            return self.harnesses[_draw_index(str(key), self.harness_seed, self.harness_weights)]

        @staticmethod
        def _prompt_under(harness: dict, prompt: Any) -> list[Any]:
            """The rollout prompt with the harness's instructions as its
            system message, in whatever message shape the prompt already
            uses (dicts, or verifiers' message objects)."""
            messages = list(prompt) if isinstance(prompt, list) else []
            rest = [
                m
                for m in messages
                if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) != "system"
            ]
            instructions = str(harness.get("instructions") or "")
            if not instructions:
                return rest
            if rest and not isinstance(rest[0], dict):
                return [vf.SystemMessage(content=instructions), *rest]
            return [{"role": "system", "content": instructions}, *rest]

        async def setup_state(self, state: dict) -> dict:
            state = (await super().setup_state(state)) or state
            info = dict(state.get("info") or {})
            seed = info.get("seed")
            harness = self.draw_harness(state)
            if harness is not None:
                state["harness"] = {"label": harness["label"], "hash": harness["hash"]}
                state["prompt"] = self._prompt_under(harness, state.get("prompt"))
                state["tool_defs"] = self._normalize_tool_defs(list(harness.get("tools") or []))
            state["zp_info"] = info
            state["zp_steps"] = []
            if self.execute is None:
                state["zp_world"] = MockEnvironment(
                    [{"type": "function", "function": t} for t in self._world_tool_defs],
                    seed=int(seed) if isinstance(seed, int) else 0,
                    faults=dict(info.get("faults") or {}),
                    world_state=str(info.get("world_state") or ""),
                    options=self.world,
                )
            return state

        def update_tool_args(
            self, tool_name: str, tool_args: dict, messages: Any, state: dict, **kwargs: Any
        ) -> dict:
            tool_args["_zp_state"] = state
            return tool_args

        async def call_tool(
            self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs: Any
        ) -> Any:
            state = tool_args.pop("_zp_state", None) or {}
            arguments = dict(tool_args)
            if self.execute is not None:
                info = state.get("zp_info") or {}
                current_rollout.prompt = _row_from_state(state, info)["prompt"]
                current_rollout.rollout_index = state.get("rollout_idx") or id(state)
                current_rollout.seed = info.get("seed")
                try:
                    result = self.execute(tool_name, arguments)
                except Exception as exc:
                    result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
            else:
                result = state["zp_world"].call(tool_name, arguments)
            if not isinstance(result, dict):
                result = {"status": "ok", "result": result}
            state.setdefault("zp_steps", []).append(
                {"tool": tool_name, "arguments": arguments, "result": result}
            )
            return vf.ToolMessage(
                role="tool", content=json.dumps(result, default=str), tool_call_id=tool_call_id
            )

        # -- rubric ----------------------------------------------------------

        def _verdict(self, state: dict) -> dict:
            cached = state.get("zp_verdict")
            if cached is None:
                row = _row_from_state(state, state.get("zp_info") or {})
                try:
                    cached = normalize_judge_result(self.reward(row))
                except Exception as exc:
                    cached = {
                        "reward": None,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "judge_status": "error",
                        "judge_meta": {},
                    }
                state["zp_verdict"] = cached
            return cached

        @staticmethod
        def _was_truncated(state: dict) -> bool:
            # A rollout cut at the turn cap or the token cap never finished
            # the task; scoring it would reward whatever it was doing when
            # the clock ran out (DAPO's overlong filtering, arXiv:2503.14476:
            # a truncated sample gets no reward signal).
            return bool(state.get("is_truncated")) or str(
                state.get("stop_condition") or ""
            ).startswith("max_turns")

        def reward_func(self, state: dict, **kwargs: Any) -> float:
            if self._was_truncated(state):
                return 0.0
            value = self._verdict(state).get("reward")
            return float(value) if isinstance(value, (int, float)) else 0.0

        def truncated(self, state: dict, **kwargs: Any) -> float:
            return 1.0 if self._was_truncated(state) else 0.0

        def trace_clean(self, state: dict, **kwargs: Any) -> float:
            """1.0 when none of the SDK's trace flags fired (fabricated test
            claims, phantom edits, test tampering, ...). Logged, not
            trained on: a monitor for over-optimization symptoms
            (Lambert 2025, chapter Over-optimization)."""
            from .score.trace import trace_flags

            row = _row_from_state(state, state.get("zp_info") or {})
            flags = trace_flags(row) or {}
            state["zp_trace_flags"] = flags
            return 0.0 if any(str(k).startswith(("lie.", "hack.")) for k in flags) else 1.0

        def n_calls(self, state: dict, **kwargs: Any) -> float:
            return float(len(state.get("zp_steps") or []))

        def judge_ok(self, state: dict, **kwargs: Any) -> float:
            return 1.0 if self._verdict(state).get("judge_status") == "ok" else 0.0

    return WhileEnv


def load_environment(
    spec: str | Path | dict,
    *,
    split: str = "train",
    reward: Any = None,
    execute: Any = None,
    world: Any = None,
    harness_mix: str | Sequence[float] = ENV_HARNESS_MIX,
    harness_seed: int = ENV_HARNESS_SEED,
    **kwargs: Any,
) -> Any:
    """Build the verifiers environment from an exported ``spec.json``.

    ``split`` picks the training task set; the holdout file, when present,
    becomes ``eval_dataset``. ``reward`` and ``execute`` override the
    spec's references (a callable or ``'module:attr'``). ``world`` (a
    ``WorldOptions`` or a dict of its fields) overrides the mock world's
    dials the spec carries; here callables such as ``fault_modes`` are fine.

    When the spec carries ``harnesses`` (``export_environment(harnesses=)``),
    each task draws one from a hash of ``harness_seed`` and its id, weighted
    by ``harness_mix`` (``"uniform"``, or one weight per harness), and the
    rollout runs with that harness's instructions as the system prompt and
    its tool schemas as the tool set; the rollout state carries
    ``harness = {label, hash}``. Kim et al. 2026 (arXiv:2606.25447).
    """
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - verifiers brings datasets
        raise ImportError("load_environment needs verifiers: pip install 'whileai[rl]'") from exc

    if isinstance(spec, dict):
        spec_dict = dict(spec)
        base = Path(spec_dict.get("_dir") or ".")
    else:
        path = Path(spec)
        spec_dict = json.loads(path.read_text(encoding="utf-8"))
        base = path.parent

    def _tasks(which: str) -> list[dict]:
        file = base / "data" / f"{which}.jsonl"
        if not file.exists():
            return []
        return [
            json.loads(line)
            for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    system = str(spec_dict.get("system_prompt") or "")

    def _rows(tasks: list[dict]) -> list[dict]:
        return [
            {
                "prompt": ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": t["prompt"]}],
                "info": t.get("info") or {},
                "example_id": t.get("example_id"),
            }
            for t in tasks
        ]

    train = _rows(_tasks(split))
    held = _rows(_tasks("holdout")) if split != "holdout" else []
    if not train:
        raise ValueError(f"no tasks for split {split!r} under {base}")
    reward_obj = resolve_ref(reward) if isinstance(reward, str) else reward
    if reward_obj is None:
        reward_obj = resolve_ref(str(spec_dict.get("reward") or DEFAULT_REWARD))
    execute_obj = resolve_ref(execute) if isinstance(execute, str) else execute
    if execute_obj is None and spec_dict.get("execute"):
        execute_obj = resolve_ref(str(spec_dict["execute"]))
    env_class = _make_env_class()
    return env_class(
        spec_dict,
        reward=reward_obj,
        execute=execute_obj,
        world=world,
        harness_mix=harness_mix,
        harness_seed=harness_seed,
        dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(held) if held else None,
        **kwargs,
    )


def summarize_tasks(tasks: Sequence[dict]) -> dict[str, Any]:
    """Counts a reviewer asks for: calibration spread and fault coverage."""
    rates = [
        t["info"]["calibration"]["pass_rate"] for t in tasks if t.get("info", {}).get("calibration")
    ]
    faults = sum(1 for t in tasks if t.get("info", {}).get("faults"))
    return {
        "tasks": len(tasks),
        "with_calibration": len(rates),
        "pass_rate_median": round(statistics.median(rates), 3) if rates else None,
        "with_faults": faults,
    }
