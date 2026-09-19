"""``compare_judges``: several judges, the same rows, one table.

    labels = {...}                                   # row key -> 0/1, a person's call
    wai.attach_labels(scored.rows, labels, kind="human")
    table = scored.compare_judges({                  # a method on the rows object
        "jev":    "typesafe:jev-latest",
        "phi-4":  wai.Hosted(),
        "haiku":  wai.Anthropic("claude-haiku-4-5"),
        "rules":  my_verifier,                       # any judge callable
    })
    print(table)          # agreement, its interval, kappa, leak, unsure, s/row, ranked
    table.best.name       # the judge to keep, when one clears the floors

``SimulationData`` and ``ScoredData`` carry the method; the function here,
``whileai.judge_comparison.compare_judges(rows, judges)``, is the same call
for a bare list of rows.

Every judge grades a copy of the same rows. Each is then scored against the
rows' gold labels the way ``judge_trust`` scores one judge: agreement with
its Wilson interval, Cohen's kappa, the leak rate (gold failures the judge
passed), how many rows it left unjudged, how many it called unsure (a
decision judge's band around even), and seconds per row. The result prints
as a table ranked by kappa and says which judge, if any, clears the floors.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import spec_of
from .simulations.defaults import JUDGE_COMPARE_CONCURRENCY, MIN_AGREEMENT, MIN_KAPPA


@dataclass
class JudgeScore:
    """One judge's row in the comparison. ``rows`` are its graded copies."""

    name: str
    spec: str | None
    n: int
    agreement: float | None
    ci95: tuple[float, float] | None
    kappa: float | None
    #: gold failures the judge passed, as a share (``pass_when_gold_fail``)
    leak: float | None
    #: gold passes the judge failed, as a share (``fail_when_gold_pass``)
    miss: float | None
    #: rows the judge marked ``unsure`` (a decision judge near even)
    unsure: int
    #: rows the judge returned no reward for (error, timeout, contract break)
    unjudged: int
    seconds: float
    seconds_per_row: float
    ok: bool
    warnings: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list, repr=False)

    @property
    def lower_bound(self) -> float | None:
        return self.ci95[0] if self.ci95 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec,
            "n": self.n,
            "agreement": self.agreement,
            "ci95": list(self.ci95) if self.ci95 else None,
            "kappa": self.kappa,
            "leak": self.leak,
            "miss": self.miss,
            "unsure": self.unsure,
            "unjudged": self.unjudged,
            "seconds": self.seconds,
            "seconds_per_row": self.seconds_per_row,
            "ok": self.ok,
            "warnings": list(self.warnings),
        }


@dataclass
class JudgeComparison:
    """The table ``compare_judges`` returns. ``scores`` is ranked: kappa
    first, agreement second; ``best`` is the first judge that clears both
    floors, else the top of the ranking with ``ok`` false."""

    scores: list[JudgeScore]
    n_rows: int
    gold_kind: str | None
    floors: tuple[float, float]
    warnings: list[str] = field(default_factory=list)

    @property
    def best(self) -> JudgeScore | None:
        for score in self.scores:
            if score.ok:
                return score
        return self.scores[0] if self.scores else None

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.scores]

    def __getitem__(self, name: str) -> JudgeScore:
        for score in self.scores:
            if score.name == name:
                return score
        raise KeyError(f"no judge named {name!r}; judges: {self.names}")

    def __iter__(self):
        return iter(self.scores)

    def __len__(self) -> int:
        return len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "gold_kind": self.gold_kind,
            "floors": {"agreement": self.floors[0], "kappa": self.floors[1]},
            "best": self.best.name if self.best else None,
            "judges": [s.to_dict() for s in self.scores],
            "warnings": list(self.warnings),
        }

    def _table(self) -> list[list[str]]:
        head = ["judge", "agree", "95% CI", "kappa", "leak", "unsure", "unjudged", "s/row", "ok"]
        body: list[list[str]] = []
        for s in self.scores:
            body.append(
                [
                    s.name,
                    _num(s.agreement),
                    f"[{s.ci95[0]:.2f}..{s.ci95[1]:.2f}]" if s.ci95 else "-",
                    _num(s.kappa),
                    _num(s.leak),
                    f"{s.unsure}" if s.unsure else "-",
                    f"{s.unjudged}" if s.unjudged else "-",
                    f"{s.seconds_per_row:.2f}",
                    "yes" if s.ok else "no",
                ]
            )
        return [head, *body]

    def __str__(self) -> str:
        gold = self.gold_kind or "none"
        lines = [
            f"compare_judges: {len(self.scores)} judges on {self.n_rows} rows, gold={gold}, "
            f"floors agreement>={self.floors[0]:.2f} kappa>={self.floors[1]:.2f}"
        ]
        table = self._table()
        widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
        for row in table:
            lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        best = self.best
        if best is None:
            lines.append("best: none (no judge was scored)")
        elif best.ok:
            lines.append(f"best: {best.name} (kappa {_num(best.kappa)}), clears both floors")
        else:
            lines.append(
                f"best: {best.name} (kappa {_num(best.kappa)}); no judge clears the floors. "
                "Change the judge prompt or the judge model, or label more rows, "
                "then compare again."
            )
        for s in self.scores:
            for w in s.warnings:
                lines.append(f"! {s.name}: {w}")
        for w in self.warnings:
            lines.append(f"! {w}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        table = self._table()
        head = "".join(f"<th>{c}</th>" for c in table[0])
        rows = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in table[1:])
        return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _rows_of(source: Any) -> list[dict]:
    rows = getattr(source, "rows", None)
    if rows is None:
        return [r for r in source if isinstance(r, dict)]
    if callable(rows) and not isinstance(rows, (list, tuple)):
        rows = rows()
    return [r for r in rows if isinstance(r, dict)]


def _name_of(value: Any) -> str:
    spec = spec_of(value)
    if isinstance(spec, str):
        return spec
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    dunder = getattr(value, "__name__", None)
    if isinstance(dunder, str) and dunder and dunder != "<lambda>":
        return dunder
    if spec is None and not callable(value):
        return "hosted"
    return type(value).__name__


def _named(judges: Any) -> list[tuple[str, Any]]:
    if isinstance(judges, Mapping):
        pairs = [(str(k), v) for k, v in judges.items()]
    else:
        pairs = []
        for value in judges:
            name = _name_of(value)
            taken = {n for n, _ in pairs}
            base, i = name, 2
            while name in taken:
                name, i = f"{base}#{i}", i + 1
            pairs.append((name, value))
    if not pairs:
        raise ValueError("compare_judges needs at least one judge in judges=")
    return pairs


def _judge_for(value: Any, *, policy: str, tools: Sequence[dict] | None) -> Callable[[dict], Any]:
    """A judge callable for a spec string, a backend object or a callable."""
    if callable(value):
        return value
    from .judge import Judge

    return Judge(model=value, policy=policy, tools=tools)


def _warm(judge: Any) -> tuple[float, str | None]:
    """Warm a spec-backed judge once so a scaled-to-zero server does not
    time out the first rows. A callable that is not a ``Judge`` is not warmed."""
    from .judge import Judge

    if not isinstance(judge, Judge):
        return 0.0, None
    from .simulations.generate.agents import default_judge_spec
    from .simulations.score.grade_llm import warm_judge

    report = warm_judge(judge.spec or default_judge_spec(), api_key=judge.api_key)
    error = None if report.get("ok") else str(report.get("error") or "warm-up failed")
    return float(report.get("seconds") or 0.0), error


def compare_judges(
    rows: Sequence[dict] | Any,
    judges: Mapping[str, Any] | Sequence[Any],
    *,
    gold: str = "gold_reward",
    allow_model_gold: bool = False,
    concurrency: int = JUDGE_COMPARE_CONCURRENCY,
    policy: str = "",
    tools: Sequence[dict] | None = None,
    floors: tuple[float, float] = (MIN_AGREEMENT, MIN_KAPPA),
) -> JudgeComparison:
    """Grade the same rows with several judges and rank them against gold labels.

    ``rows`` carry gold labels (``attach_labels`` writes ``gold_reward``
    and ``gold_kind``); a ``SimulationData`` or ``ScoredData`` is read
    through its ``rows``. ``judges`` maps a display name to a judge: a spec
    string (``"typesafe:jev-latest"``, ``"anthropic:claude-haiku-4-5"``), a
    backend object (``wai.Hosted()``, ``wai.OpenAI(...)``), a ``wai.Judge``,
    or any callable under the judge contract (a verifier, a rules function,
    your own model behind a function). A list works too and is named by
    spec or ``__name__``. Spec strings and backends become the package's
    conduct-floor ``Judge`` with ``policy=`` and ``tools=`` so every model
    reads the same prompt; a callable is used as given. Each spec-backed
    judge is warmed once before its rows fan out.

    Each judge grades its own copy of the rows (originals stay untouched),
    then is scored the way ``judge_trust`` scores one judge: agreement with
    a Wilson 95% interval, Cohen's kappa, the leak rate (gold failures the
    judge passed), rows left unjudged, rows called unsure, and seconds per
    row. ``floors`` are the agreement lower bound and kappa a judge must
    reach for ``ok`` (defaults ``MIN_AGREEMENT`` 0.8, the human-human
    agreement of MT-Bench, and ``MIN_KAPPA`` 0.6, "substantial"). Model or
    unknown gold keeps every ``ok`` false unless ``allow_model_gold=True``,
    as in ``judge_agreement``.

    Returns a ``JudgeComparison``: ``scores`` ranked by kappa then agreement,
    ``best``, ``gold_kind``, and ``[name].rows`` for the graded copies, so a
    disagreement can be read row by row. ``print(table)`` is the report.

    Reference: [7] Zheng et al. 2023 (MT-Bench agreement floor), [6] Cohen
    1960 (kappa), [9] Wilson 1927 (the interval), Lambert 2025, chapter Reward
    Modeling.
    """
    from .simulations.score.agreement import judge_agreement
    from .simulations.score.judging import run_judge
    from .simulations.score.stats import wilson_interval

    source = _rows_of(rows)
    if not source:
        raise ValueError("compare_judges needs rows; got none")
    min_agreement, min_kappa = float(floors[0]), float(floors[1])
    scores: list[JudgeScore] = []
    kinds: set[str] = set()
    for name, value in _named(judges):
        judge = _judge_for(value, policy=policy, tools=tools)
        spec = spec_of(value)
        warnings: list[str] = []
        _, warm_error = _warm(judge)
        if warm_error:
            warnings.append(f"warm-up failed: {warm_error}")
        started = time.monotonic()
        scored = run_judge(source, judge, judge_name=name, concurrency=int(concurrency))
        seconds = time.monotonic() - started
        graded = list(scored.rows)
        agreement = judge_agreement(graded, gold, allow_model_gold=allow_model_gold)
        warnings.extend(agreement.get("warnings") or [])
        n = int(agreement.get("n") or 0)
        conf = agreement.get("confusion") or {}
        ci = wilson_interval(int(conf.get("tp", 0)) + int(conf.get("tn", 0)), n) if n else None
        kappa = agreement.get("kappa")
        if agreement.get("gold_kind"):
            kinds.add(str(agreement["gold_kind"]))
        unjudged = sum(1 for r in graded if r.get("reward") is None)
        unsure = sum(1 for r in graded if (r.get("judge_meta") or {}).get("unsure"))
        ok = (
            bool(agreement.get("ok"))
            and ci is not None
            and ci[0] >= min_agreement
            and kappa is not None
            and float(kappa) >= min_kappa
        )
        scores.append(
            JudgeScore(
                name=name,
                spec=spec if isinstance(spec, str) else None,
                n=n,
                agreement=agreement.get("agreement"),
                ci95=(round(ci[0], 4), round(ci[1], 4)) if ci else None,
                kappa=kappa,
                leak=agreement.get("pass_when_gold_fail"),
                miss=agreement.get("fail_when_gold_pass"),
                unsure=unsure,
                unjudged=unjudged,
                seconds=round(seconds, 3),
                seconds_per_row=round(seconds / max(1, len(graded)), 4),
                ok=ok,
                warnings=warnings,
                rows=graded,
            )
        )
    scores.sort(
        key=lambda s: (
            s.kappa if s.kappa is not None else float("-inf"),
            s.agreement if s.agreement is not None else float("-inf"),
        ),
        reverse=True,
    )
    notes: list[str] = []
    if len(kinds) > 1:
        notes.append(f"gold labels of mixed kinds: {sorted(kinds)}")
    gold_kind = next(iter(kinds)) if len(kinds) == 1 else (None if not kinds else "mixed")
    return JudgeComparison(
        scores=scores,
        n_rows=len(source),
        gold_kind=gold_kind,
        floors=(min_agreement, min_kappa),
        warnings=notes,
    )


__all__ = ["JudgeComparison", "JudgeScore", "compare_judges"]
