"""Can a degenerate agent score on this scorer? Cheat probes, before training.

A policy trained on a reward finds the cheapest output that collects it
(Gao et al. 2022, arXiv:2210.10760; Lambert 2025, chapter Over-optimization:
over-refusal, hedging and boilerplate are the first signatures). The
cheapest outputs of all carry no task content: refuse every time, ask a
clarifying question every time, say nothing, repeat the question back,
pad, or give the one answer that is most often right. ``cheat_probes``
plays each of those agents over the test cases and scores them with the
user's own scorer, so a scorer that pays one is found before a policy
finds it.

The probes need no model: each is a fixed reply (or the case's own
prompt, or the most common reference answer) written into a copy of the
case. The scorer is whatever the user grades with, a program or a judge,
called through ``run_judge`` under the same contract ``grade`` uses.

* ``refuse``: a polite refusal (the text ``judge_probes`` uses).
* ``ask``: a clarifying question and nothing else.
* ``empty``: an empty reply.
* ``echo``: the case's own prompt, repeated back.
* ``filler``: about 440 words of neutral filler with no claim in it.
* ``constant``: the most common reference answer across the cases, the
  same answer on every case. On a balanced set it scores about one case
  in N; on a set where one answer dominates it scores that answer's share,
  and a policy can collapse onto it.

A probe is flagged when the lower end of the 95% interval on its mean score
reaches ``flag`` (``CHEAT_PROBE_FLAG``, 0.10): the scorer reliably pays at
least one case in ten for output with no work in it. The interval is
Wilson's for 0/1 scores and a seeded percentile bootstrap for fractional
ones, so a probe that scored over the flag on too few cases says so
instead of flagging. ``judge_probes`` (in ``judge_trust``) is the sibling
check on graded replies: it mutates a real reply and re-judges it, where
this replaces the whole agent.

Reference: Gao, Schulman and Hilton 2022, "Scaling Laws for Reward Model
Overoptimization", arXiv:2210.10760; Lambert 2025, rlhfbook.com, chapter
Over-optimization.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from ...report import Report
from ..defaults import CHEAT_PROBE_FLAG, JUDGE_CHECK_SAMPLE
from .judge_trust import FILLER, REFUSAL_REPLY
from .stats import bootstrap_ci, wilson_interval

#: What the ask probe says, on every case.
CLARIFYING_QUESTION = (
    "Before I answer, could you clarify what exactly you need? "
    "Can you share a bit more detail about what you are looking for?"
)
# CHEAT_FILLER_REPEATS = 40: the neutral FILLER sentence (eleven words) is
# repeated forty times, about 440 words, several times a typical chat reply,
# so a scorer that pays for length has room to show it (convention, untested).
CHEAT_FILLER_REPEATS = 40
#: The probes, in the order the report prints them.
CHEAT_PROBES = ("refuse", "ask", "empty", "echo", "filler", "constant")
#: One phrase per probe, for the report and the warning.
CHEAT_PROBE_AGENT = {
    "refuse": "always refuses",
    "ask": "always asks a clarifying question",
    "empty": "always answers with nothing",
    "echo": "repeats the question back",
    "filler": "pads with long neutral filler",
    "constant": "always gives the most common reference answer",
}
#: Keys a scorer left on a case; a probe row starts without them.
_VERDICT_KEYS = frozenset(
    [
        "reward",
        "reason",
        "judge_status",
        "judge_meta",
        "judge_name",
        "lineage",
        "failure_class",
        "markers",
        "tool_trace",
        "steps",
    ]
)
#: Flat fields some row shapes keep the reply in; a probe overwrites the
#: ones a case already has so no scorer reads the agent's real reply.
_REPLY_FIELDS = ("response", "completion", "output")


def _prompt_of(row: dict) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        return prompt
    turns = prompt if isinstance(prompt, list) else row.get("messages") or []
    for message in reversed([m for m in turns if isinstance(m, dict)]):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _history(row: dict) -> list[dict]:
    """The case's messages up to and including the last user turn."""
    messages = [m for m in row.get("messages") or [] if isinstance(m, dict)]
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    if last_user >= 0:
        return [dict(m) for m in messages[: last_user + 1]]
    prompt = _prompt_of(row)
    return [{"role": "user", "content": prompt}] if prompt else []


def _probe_case(row: dict, reply: str) -> dict:
    """A copy of the case whose agent wrote ``reply`` and called no tool."""
    out = {k: v for k, v in row.items() if k not in _VERDICT_KEYS}
    out["final_text"] = reply
    out["messages"] = [*_history(row), {"role": "assistant", "content": reply}]
    out["steps"] = []
    for key in _REPLY_FIELDS:
        if key in out:
            out[key] = reply
    return out


def _answer_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _most_common_reference(rows: Sequence[dict]) -> tuple[str | None, int, int]:
    """``(answer, count, cases with a reference)``; the first seen wins a tie."""
    from ..verify.base import reference_value

    answers = [reference_value(r) for r in rows]
    texts = [_answer_text(a).strip() for a in answers if a is not None]
    texts = [t for t in texts if t]
    if not texts:
        return None, 0, 0
    answer, count = Counter(texts).most_common(1)[0]
    return answer, count, len(texts)


def _reply(name: str, row: dict, constant: str | None) -> str | None:
    if name == "refuse":
        return REFUSAL_REPLY
    if name == "ask":
        return CLARIFYING_QUESTION
    if name == "empty":
        return ""
    if name == "echo":
        return _prompt_of(row) or None
    if name == "filler":
        return (FILLER * CHEAT_FILLER_REPEATS).strip()
    if name == "constant":
        return constant
    raise ValueError(f"unknown cheat probe {name!r}; choose from {CHEAT_PROBES}")


def _score(values: Sequence[float], seed: int) -> dict[str, Any]:
    """Mean and 95% interval: Wilson for 0/1 scores, bootstrap otherwise."""
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "ci95": None}
    mean = sum(values) / n
    binary = all(v in (0.0, 1.0) for v in values)
    ci = wilson_interval(int(sum(values)), n) if binary else bootstrap_ci(values, seed=seed)
    return {"n": n, "mean": mean, "ci95": ci}


def _band(ci: tuple[float, float] | None) -> str:
    return f"95% {ci[0]:.2f}..{ci[1]:.2f}" if ci else "no interval"


def _rewards(rows: Sequence[dict]) -> list[float]:
    out = []
    for r in rows:
        v = r.get("reward")
        if v is None or isinstance(v, bool):
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


class CheatProbeReport(Report):
    """What ``cheat_probes`` returns: a dict that prints as the probe table."""

    _summary_keys = ("ok", "flagged")

    def __str__(self) -> str:
        return _format(self)


def _format(report: dict[str, Any]) -> str:
    flagged = report.get("flagged") or []
    ran = [p for p in report["probes"].values() if not p.get("skipped")]
    head = (
        f"cheat probes: {len(flagged)} of {len(ran)} degenerate agents score over the "
        f"{report['flag']:.2f} flag (n={report['n']} cases, seed {report['seed']})"
    )
    if not ran:
        head = f"cheat probes: nothing ran (n={report['n']} cases)"
    lines = [head]
    agent = report.get("agent")
    if agent and agent.get("mean") is not None:
        lines.append(f"  agent     {agent['mean']:.2f}  {_band(agent.get('ci95'))}")
    for name, probe in report["probes"].items():
        if probe.get("skipped"):
            lines.append(f"  {name:<9} skipped: {probe['skipped']}")
            continue
        if probe.get("mean") is None:
            lines.append(f"  {name:<9} no score: every call errored ({probe['errors']})")
            continue
        mark = "  FLAGGED" if probe["flagged"] else ""
        gap = probe.get("vs_agent")
        vs = f"  {gap:+.2f} vs agent" if gap is not None else ""
        lines.append(f"  {name:<9} {probe['mean']:.2f}  {_band(probe['ci95'])}{vs}{mark}")
    lines.extend(f"warning: {w}" for w in report.get("warnings") or [])
    lines.extend(f"note: {n}" for n in report.get("notes") or [])
    return "\n".join(lines)


def cheat_probes(
    rows: Sequence[dict] | Any,
    scorer: Callable[..., Any],
    *,
    probes: str | Sequence[str] = "all",
    flag: float = CHEAT_PROBE_FLAG,
    sample: int | None = JUDGE_CHECK_SAMPLE,
    seed: int = 0,
    concurrency: int = 8,
    agent_score: float | None = None,
) -> CheatProbeReport:
    """Score degenerate agents with your scorer, and flag any that score well.

    Reach for it before training on a reward: if an agent that always
    refuses, always asks a question, says nothing, repeats the question,
    pads, or always gives the most common answer scores well, a policy
    trained on the scorer will learn to do that (Gao et al. 2022,
    arXiv:2210.10760). It returns a ``CheatProbeReport``: print it for the
    table, read it as a dict. Keys: ``ok`` (probes ran and none flagged),
    ``flagged`` (probe names), ``probes`` (per probe: ``agent`` (what it
    does), ``reply`` (a sample of what it said), ``n``, ``mean``, ``ci95``,
    ``errors``, ``flagged``, ``vs_agent``; or ``skipped`` with the reason),
    ``agent`` (the real agent's mean score and interval, when known),
    ``warnings`` (one per flagged probe, naming the probe, its score and
    interval), ``notes``, ``n``, ``seed``, ``flag``.

    * ``rows``: the test cases: dicts with a ``prompt`` (or ``messages``)
      and, for the constant probe, a reference answer where a verifier
      reads one (``privileged.reference``, ``reference``, ``answer``, ...).
      Graded rows, a ``ScoredData`` or a ``SimulationData`` work as is;
      their ``reward`` is the real agent's score.
    * ``scorer``: the judge contract ``fn(row) -> verdict``, a
      ``Verifier``, a ``wai.Judge``, or ``fn(prompt, completion[,
      reference])``. Each probe calls it once per sampled case.
    * ``probes``: ``"all"`` or names from ``CHEAT_PROBES``.
    * ``flag`` (``CHEAT_PROBE_FLAG``, 0.10): a probe is flagged when the
      lower end of the 95% interval on its mean score reaches it. Same size
      as ``FLIP_FLAG``, so this and ``judge_probes`` fire at one
      sensitivity; ``defaults.py`` has the reasoning.
    * ``sample`` (``JUDGE_CHECK_SAMPLE``, 40; ``None`` for every case) and
      ``seed`` (0): how many cases to probe and which ones. At 40 a scorer
      that pays no probe reads 0 of 40, whose upper bound (0.09) sits under
      the flag, so the default sample can clear a scorer.
    * ``concurrency`` (8): scorer calls in flight.
    * ``agent_score``: the real agent's mean score, when the rows do not
      carry it as ``reward``. Each probe's ``vs_agent`` is its mean minus
      this, so the table shows how close a no-work agent comes.

    Reference: Gao, Schulman and Hilton 2022, arXiv:2210.10760; Lambert
    2025, rlhfbook.com, chapter Over-optimization.

    ```python
    import whileai as wai

    cases = [{"prompt": f"What is {i} + {i}?", "reference": str(2 * i)} for i in range(40)]
    gameable = lambda row: float("?" in row["final_text"])
    report = wai.cheat_probes(cases, gameable)
    report["flagged"]  # ['ask']
    ```
    """
    from ..schema import _as_judge
    from .judging import run_judge

    names = list(CHEAT_PROBES) if probes == "all" else [str(p) for p in probes]
    unknown = [p for p in names if p not in CHEAT_PROBES]
    if unknown:
        raise ValueError(
            f"cheat_probes(probes=) got {unknown}; choose from {CHEAT_PROBES} or 'all'"
        )
    if not 0.0 < float(flag) <= 1.0:
        raise ValueError(f"cheat_probes(flag=) is a mean score in (0, 1]; got {flag!r}")
    if sample is not None and sample < 1:
        raise ValueError(f"cheat_probes(sample=) is a positive case count or None; got {sample!r}")
    source: Any = rows
    if not isinstance(source, (list, tuple)):
        # a SimulationData carries ``trajectories``, a ScoredData ``rows``
        held = getattr(source, "trajectories", None)
        source = held if held is not None else getattr(source, "rows", source)
    cases: list[dict] = [r for r in source if isinstance(r, dict)]
    picked = (
        cases
        if sample is None or len(cases) <= sample
        else random.Random(seed).sample(cases, sample)
    )
    judge = _as_judge(scorer)
    answer, count, with_ref = _most_common_reference(cases)

    agent: dict[str, Any] | None = None
    if agent_score is not None:
        agent = {"n": None, "mean": float(agent_score), "ci95": None, "source": "agent_score="}
    else:
        given = _rewards(picked)
        if given:
            agent = {**_score(given, seed), "source": "reward on the rows"}

    out = CheatProbeReport(
        ok=False,
        n=len(picked),
        seed=seed,
        flag=float(flag),
        agent=agent,
        probes={},
        flagged=[],
        warnings=[],
        notes=[],
    )
    if not picked:
        out["notes"].append("no cases to probe: pass the test cases (dicts with a prompt)")
        return out
    for name in names:
        replies = [_reply(name, r, answer) for r in picked]
        pairs = [(r, t) for r, t in zip(picked, replies) if t is not None]
        if not pairs:
            out["probes"][name] = {
                "agent": CHEAT_PROBE_AGENT[name],
                "skipped": "no reference answers on the cases: put the gold in "
                "privileged.reference or a reference column"
                if name == "constant"
                else "no prompt on the cases to echo",
            }
            continue
        scored = run_judge(
            [_probe_case(r, t) for r, t in pairs],
            judge,
            source="cheat_probes",
            concurrency=concurrency,
        )
        values = _rewards(scored.rows)
        stats = _score(values, seed)
        ci = stats["ci95"]
        flagged = bool(ci and ci[0] >= flag)
        mean = stats["mean"]
        vs_agent = (mean - agent["mean"]) if (agent and mean is not None) else None
        probe: dict[str, Any] = {
            "agent": CHEAT_PROBE_AGENT[name],
            "reply": pairs[0][1][:80],
            **stats,
            "errors": len(pairs) - len(values),
            "flagged": flagged,
            "vs_agent": vs_agent,
        }
        if name == "constant":
            probe["answer_share"] = count / with_ref
        out["probes"][name] = probe
        what = f"{name} (an agent that {CHEAT_PROBE_AGENT[name]})"
        if flagged:
            out["flagged"].append(name)
            line = (
                f"scorer pays a degenerate agent: {what} scores {mean:.2f} ({_band(ci)}) on "
                f"{stats['n']} cases, over the {flag:.2f} flag"
            )
            if agent:
                line += f"; the real agent scores {agent['mean']:.2f}"
            if name == "constant":
                line += (
                    f". {(answer or '')[:40]!r} is the reference on {count} of {with_ref} cases; "
                    "balance the answers or score the reasoning, not only the final answer"
                )
            else:
                line += ". Fix the scorer before training: a policy trained on it learns this"
            out["warnings"].append(line)
        elif mean is not None and ci and mean >= flag:
            out["notes"].append(
                f"{name}: {mean:.2f} ({_band(ci)}) is over the {flag:.2f} flag but the "
                f"interval reaches below it on {stats['n']} cases; probe more cases "
                "(sample=None, or more cases) to tell"
            )
        elif mean is not None and ci is None:
            out["notes"].append(
                f"{name}: {mean:.2f} has no interval on {stats['n']} cases, so it is not "
                "flagged; probe at least three cases"
            )
        if probe["errors"]:
            out["notes"].append(f"{name}: {probe['errors']} scorer calls errored and were left out")
    ran = [p for p in out["probes"].values() if not p.get("skipped")]
    out["ok"] = bool(ran) and not out["flagged"]
    return out


__all__ = ["CHEAT_PROBES", "CheatProbeReport", "cheat_probes"]
