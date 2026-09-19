"""One judge contract for grading and evaluation, and the loop around it.

A judge is any callable that takes a trajectory dict and returns a verdict.
LLM judge, rules engine, reward model, human-label lookup, API call: the
SDK does not care how the reward was produced, only that the result honors
the contract below. The same engine scores generated simulation data and
held-out model rollouts, so training-time grading and eval-time judging
share one schema.

Judge contract (minimal now, extensible later)::

    judge(trajectory) -> {"reward": 0 or 1}            # minimum
    judge(trajectory) -> {"reward": 0.7,               # floats allowed
                          "reason": "...",             # optional
                          "markers": {"name": 1.0},    # optional
                          "failure_class": "...",      # optional
                          "failures": [...],           # optional
                          ...anything else}            # kept in judge_meta
    judge(trajectory) -> 0 or 1 or 0.7                 # bare number works

``reward``, ``reason``, ``markers`` and ``failure_class`` are the keys
that land on the row itself. Every other key the judge returns is kept
under ``row["judge_meta"]`` and nowhere else, so a judge that returns
``failures`` reads back as ``row["judge_meta"]["failures"]``, not
``row["failures"]``. A judge may also hand those extras over already
gathered in its own ``judge_meta`` dict; both spellings land in the same
place.

``markers`` is lifted onto ``row["markers"]``, which is what
``marker_summary``, ``delta_report`` and ``from_row`` read. Marker
polarity is a convention the whole SDK depends on: **1.0 is the good
outcome, higher is better, and a significant drop is the regression.**
Name a marker for the behavior you want (``refund_correct``, not
``false_refund_success``) or ``delta_report`` reads your improvement as a
regression and ``must_not_regress=`` fails the run that fixed the bug.

Anything else — missing reward, unsupported type, an exception — marks the
row (``judge_status`` of ``missing_reward`` / ``invalid_result`` /
``error`` / ``timeout``) with ``reward=None``. Nothing is silently zero.

The five-line loop::

    import whileai.simulations as wai
    judge = lambda t: {"reward": int("sorry" not in t["final_text"])}
    scored = wai.run_judge(data.trajectories, judge)     # or data.grade(judge=judge)
    wai.export_training(scored.passes(), output="train.jsonl",
                        system_prompt=POLICY, tools=TOOLS)
    # ...train externally, roll the model on a holdout...
    evald = wai.evaluate(rollouts, judge, model="my-tuned-v1")
    nxt = wai.simulate(tools=TOOLS, system_prompt=POLICY,
                       traces=evald.failed_traces())     # loop closed
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from ..defaults import DIFFICULTY_BAND, JUDGE_COMPARE_CONCURRENCY, MIN_AGREEMENT, MIN_KAPPA
from ..tools import schemas as _tool_schemas
from .hygiene import coverage_warnings

log = logging.getLogger("whileai.simulations")

_VALID_STATUSES = ("ok", "missing_reward", "invalid_result", "error", "timeout")
# LENGTH_CONFOUND_MIN_PAIRS = 8 / LENGTH_CONFOUND_SHARE = 0.75: the chosen
# reply being the longer one in three quarters of eight or more pairs is
# flagged as a length confound; LENGTH_CONFOUND_ALL_FROM = 3: in every pair
# once there are three, since a total confound is one at any size
# (length is the confound a judge is told to ignore: Zheng et al. 2023,
# arXiv:2306.05685; Lambert 2025, chapter Reward Modeling). The numbers
# are a convention, untested.
LENGTH_CONFOUND_MIN_PAIRS = 8
LENGTH_CONFOUND_SHARE = 0.75
LENGTH_CONFOUND_ALL_FROM = 3


def _scaled(value: float, scale: tuple[float, float]) -> tuple[float | int, dict[str, Any]]:
    """A rating on ``scale`` (lo, hi) as a [0, 1] reward, with the raw
    rating kept as judge_meta (Lambert 2025, chapter Preference Data:
    ratings are metadata worth keeping next to the normalized preference)."""
    lo, hi = float(scale[0]), float(scale[1])
    reward = (value - lo) / (hi - lo)
    out: float | int = int(reward) if reward in (0.0, 1.0) else round(reward, 6)
    return out, {"rating": value, "scale": [lo, hi]}


def _verdict_meta(raw: dict, drop: set[str]) -> dict[str, Any]:
    """The verdict's metadata: its own keys, plus its ``judge_meta`` flattened.

    A judge may report metadata either way -- loose keys alongside
    ``reward``, or gathered under ``judge_meta``. The SDK's own verifiers
    use the second shape, so sweeping ``judge_meta`` in as an ordinary key
    would nest it under itself and hide ``verifier``, ``failure_class`` and
    ``markers`` one level below where every reader looks. A non-dict
    ``judge_meta`` is not a metadata block, so it stays a plain key.
    """
    nested = raw.get("judge_meta")
    if not isinstance(nested, dict):
        return {k: v for k, v in raw.items() if k not in drop}
    outer = {k: v for k, v in raw.items() if k not in drop and k != "judge_meta"}
    # The inner block is the judge's considered metadata; it wins a collision.
    return {**outer, **nested}


def normalize_judge_result(raw: Any, *, scale: tuple[float, float] | None = None) -> dict[str, Any]:
    """Coerce one judge return into the contract; never invent a reward.

    ``scale=(lo, hi)`` reads the judge's number as a rating on that scale
    (a 1 to 5 Likert, a 0 to 10 score): the row's ``reward`` is the
    rating mapped onto [0, 1] and ``judge_meta`` keeps ``rating`` and
    ``scale``. A rating outside the scale is a contract break, as a
    reward outside [0, 1] is without one. A dict may carry the number as
    ``rating`` instead of ``score`` when a scale is set.
    """
    if scale is not None:
        lo, hi = float(scale[0]), float(scale[1])
        if not hi > lo:
            raise ValueError("scale must be (lo, hi) with hi > lo")
        number: Any = raw
        reason = ""
        meta: dict[str, Any] = {}
        if isinstance(raw, dict):
            reason = str(raw.get("reason") or "")
            number = raw.get("rating", raw.get("score", raw.get("reward")))
            meta = _verdict_meta(raw, {"rating", "score", "reward", "reason"})
            if number is None:
                return {
                    "reward": None,
                    "reason": reason,
                    "judge_status": "missing_reward",
                    "judge_meta": {"returned_keys": sorted(map(str, raw))},
                }
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            return {
                "reward": None,
                "reason": reason,
                "judge_status": "invalid_result",
                "judge_meta": {"reward_type": type(number).__name__},
            }
        if not lo <= float(number) <= hi:
            return {
                "reward": None,
                "reason": reason,
                "judge_status": "invalid_result",
                "judge_meta": {"rating_out_of_scale": float(number), "scale": [lo, hi]},
            }
        reward, rating_meta = _scaled(float(number), (lo, hi))
        return {
            "reward": reward,
            "reason": reason,
            "judge_status": "ok",
            "judge_meta": {**meta, **rating_meta},
        }
    if isinstance(raw, bool):
        return {"reward": int(raw), "reason": "", "judge_status": "ok", "judge_meta": {}}
    if isinstance(raw, (int, float)):
        value: Any = float(raw)
        # Scalar lane: 1 pass, 0 fail, (0, 1) partial. Anything outside
        # [0, 1] fits no lane and is a contract break, not a reward.
        if not 0.0 <= value <= 1.0:
            return {
                "reward": None,
                "reason": "",
                "judge_status": "invalid_result",
                "judge_meta": {"reward_out_of_range": value},
            }
        reward = int(value) if value in (0.0, 1.0) else value
        return {"reward": reward, "reason": "", "judge_status": "ok", "judge_meta": {}}
    if isinstance(raw, dict):
        if "reward" not in raw and "score" not in raw:
            return {
                "reward": None,
                "reason": "",
                "judge_status": "missing_reward",
                "judge_meta": {"returned_keys": sorted(map(str, raw))},
            }
        value = raw.get("reward", raw.get("score"))
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, (int, float)):
            return {
                "reward": None,
                "reason": str(raw.get("reason") or ""),
                "judge_status": "invalid_result",
                "judge_meta": {"reward_type": type(value).__name__},
            }
        if not 0.0 <= float(value) <= 1.0:
            return {
                "reward": None,
                "reason": str(raw.get("reason") or ""),
                "judge_status": "invalid_result",
                "judge_meta": {"reward_out_of_range": float(value)},
            }
        reward = int(value) if float(value) in (0.0, 1.0) else float(value)
        meta = _verdict_meta(raw, {"reward", "score", "reason"})
        return {
            "reward": reward,
            "reason": str(raw.get("reason") or ""),
            "judge_status": "ok",
            "judge_meta": meta,
        }
    return {
        "reward": None,
        "reason": "",
        "judge_status": "invalid_result",
        "judge_meta": {"returned_type": type(raw).__name__},
    }


class ScoredData:
    """Scored trajectories: the one representation grade and eval share.

    Iterates as plain dicts, so it feeds ``simulate(traces=...)``,
    ``mine_traces``, ``export_training`` and JSONL writers directly —
    no conversion scripts.

    ``.rows`` is a ``RowList``: a list that also answers to being
    called, so both ``scored.rows`` and ``scored.rows()`` give the
    scored rows. ``SimulationData.rows``, what ``simulate()`` returns,
    behaves the same way, so the two spellings are interchangeable
    across ``simulate() -> run_judge()``. ``.warnings`` is the list of
    hollow-run notes ``run_judge`` filled; print it before reading any
    number.
    """

    def __init__(
        self,
        rows: list[dict],
        *,
        run_id: str,
        source: str,
        judge_name: str,
        model: str | None = None,
    ):
        # A RowList, not the plain list handed in: ``scored.rows()`` used
        # to raise ``TypeError: 'list' object is not callable`` while
        # ``SimulationData.rows()`` worked, so a ``hasattr(x, "rows")``
        # guard picked the wrong branch on the judging path (#344).
        # RowList subclasses list, so every existing ``scored.rows``
        # use keeps working unchanged.
        from ..data import RowList

        self.rows = RowList(rows)
        self.run_id = run_id
        self.source = source
        self.eval_coverage: dict[str, Any] | None = None
        self.judge_name = judge_name
        self.model = model
        # the AgentProfile of the run these rows came from, when grade()
        # made them; select() and export read the prompt and tools off it
        self.profile: Any = None
        # Plain-words notes on whether the score means anything: no row
        # called a tool, a marker that never fired, a unanimous verdict.
        # Filled by ``run_judge`` from ``coverage_warnings``; printed once.
        self.warnings: list[str] = []

    def __iter__(self) -> Iterator[dict]:
        return iter(self.rows)

    def compare_judges(
        self,
        judges: Mapping[str, Any] | Sequence[Any],
        *,
        gold: str = "gold_reward",
        allow_model_gold: bool = False,
        concurrency: int = JUDGE_COMPARE_CONCURRENCY,
        floors: tuple[float, float] = (MIN_AGREEMENT, MIN_KAPPA),
    ):
        """Grade these rows with several judges and rank them against the gold labels.

        ``judges`` maps a name to a spec string (``"typesafe:jev-latest"``),
        a backend object, a ``wai.Judge`` or any judge callable. Each grades
        its own copy of the rows under this run's system prompt and tools,
        then is scored the way ``judge_trust`` scores one judge: agreement
        with a Wilson interval, kappa, leak rate, unsure and unjudged
        counts, seconds per row. Returns a ``JudgeComparison`` that prints
        as a table ranked by kappa; ``whileai.judge_comparison.compare_judges``
        has the full account and takes a bare row list.
        """
        from whileai.judge_comparison import compare_judges

        profile = self.profile
        return compare_judges(
            self.rows,
            judges,
            gold=gold,
            allow_model_gold=allow_model_gold,
            concurrency=concurrency,
            policy=str(getattr(profile, "policy", "") or "") if profile else "",
            tools=list(getattr(profile, "tools", None) or []) if profile else None,
            floors=floors,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    def partials(self) -> list[dict]:
        """Rows with a continuous reward strictly between 0 and 1. The
        scalar lane: 1 pass, 0 fail, partials here, None unjudged - every
        contract-legal reward is visible in exactly one view."""
        return [
            r for r in self.rows if isinstance(r.get("reward"), float) and 0.0 < r["reward"] < 1.0
        ]

    def select_by_reward_range(self, lo: float, hi: float, *, inclusive: bool = True) -> list[dict]:
        """Rows whose numeric reward falls in [lo, hi] (or (lo, hi))."""
        out = []
        for r in self.rows:
            value = r.get("reward")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if lo <= value <= hi if inclusive else lo < value < hi:
                out.append(r)
        return out

    def select_by_reward(self, reward) -> list[dict]:
        return [r for r in self.rows if r.get("reward") == reward]

    def passes(self) -> list[dict]:
        return self.select_by_reward(1)

    def failures(self) -> list[dict]:
        return self.select_by_reward(0)

    @property
    def traces(self) -> list[dict]:
        """The doctrine-sketch spelling of the loop-closing move:
        ``simulate(traces=results.traces)``. Failed trajectories, ready
        to aim the next generation round."""
        return self.failed_traces()

    def failed_traces(self) -> list[dict]:
        """Failures, ready to hand to ``simulate(traces=...)``."""
        return self.failures()

    @property
    def pass_at(self):
        """pass@1 / pass^k / pass@k over the graded groups (``PassAt``).
        pass@1 for measurement, pass^k for reliability, pass@k - pass@1
        for RL headroom. See ``whileai.simulations.score.passat``."""
        from .passat import pass_at

        return pass_at(self.rows)

    def unjudged(self) -> list[dict]:
        """Rows the judge could not score. Never treated as failures."""
        return [r for r in self.rows if r.get("judge_status") != "ok"]

    def select(
        self,
        *,
        mode: str = "rl",
        target: int = 1000,
        band: tuple[float, float] | None = None,
        endorsed: Sequence[str] = (),
        truncated: str = "drop",
    ):
        """The rows worth training on, as a ``Selection`` that prints its
        report: ``optimize`` over the graded copies, ``mode="rl"`` by
        default. ``band``, ``endorsed`` and ``truncated`` as there."""
        from ...selection import select

        return select(
            self, mode=mode, target=target, band=band, endorsed=endorsed, truncated=truncated
        )

    def select_for_sft(self, *, target: int = 1000) -> tuple[list[dict], dict[str, Any]]:
        """Diverse correct demonstrations: 1-labeled, deduped by behavior."""
        from .optimize import select_for_sft

        return select_for_sft(self.rows, target=target)

    def agreement(
        self, gold: str | Sequence[dict] = "gold_reward", *, reward: str = "reward"
    ) -> dict[str, Any]:
        """Agreement of this run's rewards with a trusted label. See ``judge_agreement``."""
        from .agreement import judge_agreement

        return judge_agreement(self.rows, gold, reward=reward)

    def select_for_preference(
        self, *, max_pairs_per_prompt: int = 1, min_margin: float = 1.0, length_match: bool = True
    ) -> tuple[list[dict], dict[str, Any]]:
        """Chosen/rejected pairs from same-task contrast. Failures earn here."""
        return build_preference_pairs(
            self.rows,
            max_pairs_per_prompt=max_pairs_per_prompt,
            min_margin=min_margin,
            length_match=length_match,
        )

    def select_for_rl(
        self,
        *,
        target: int = 1000,
        lo: float = DIFFICULTY_BAND[0],
        hi: float = DIFFICULTY_BAND[1],
        has_tools: bool = True,
    ) -> tuple[list[dict], dict[str, Any]]:
        """Whole mixed-reward groups for RL; groups never split. ``lo`` and
        ``hi`` default to ``DIFFICULTY_BAND`` (0.2, 0.8), the same band
        ``select_for_rl`` and ``optimize`` use; they used to be 0.3 and
        0.7 here alone."""
        from .optimize import select_for_rl

        return select_for_rl(self.rows, target=target, lo=lo, hi=hi, has_tools=has_tools)

    def report(
        self, *, tools: Sequence[dict] | None = None, system_prompt: str = ""
    ) -> dict[str, Any]:
        from .preflight import dataset_report

        out = dataset_report(self.rows, tools=tools, system_prompt=system_prompt)
        rewards: list[Any] = [
            r.get("reward") for r in self.rows if isinstance(r.get("reward"), (int, float))
        ]
        out["reward_distribution"] = {
            str(k): rewards.count(k) for k in sorted(set(rewards), key=lambda v: float(v))
        }
        out["unjudged"] = len(self.unjudged())
        out["scoring_run_id"] = self.run_id
        out["judge"] = self.judge_name
        return out

    def save(self, path: str) -> str:
        with open(path, "w") as fh:
            for row in self.rows:
                fh.write(json.dumps(row, default=str) + "\n")
        return path

    def push(self, name: str, **kwargs: Any) -> dict:
        """Upload the scored rows to the platform: ``push_rows(self.rows, name, ...)``.

        Same keywords as ``push_rows`` (``gate=``, ``mode=``, ``agent=``,
        ``purpose=``, ``parent=``, ``endorsed=``, ``strict_hacks=``,
        ``holdout=`` for a linked holdout set split by task, ``publish=``
        for a public card, ``timeout=`` for the upload). The graded copy is
        what a gated RL push needs, and ``SimulationData.push`` cannot see
        it: ``grade(judge=)`` leaves the run's trajectories ungraded on
        purpose.
        """
        from ..ingest.platform import push_rows

        return push_rows(self.rows, name, **kwargs)


def _instance_name(judge: Any) -> str:
    """A callable instance's own name, when it has a usable one.

    ``Verifier`` and anything else honoring the judge contract as an object
    carries ``name``. Read defensively: ``name`` on an arbitrary callable may
    be absent, not a string, or a property that raises.
    """
    try:
        value = getattr(judge, "name", None)
    except Exception:
        return ""
    return value.strip() if isinstance(value, str) else ""


def _score_one(
    judge: Callable, row: dict, scale: tuple[float, float] | None = None
) -> dict[str, Any]:
    try:
        return normalize_judge_result(judge(row), scale=scale)
    except Exception as exc:  # judge bugs mark the row, never crash the run
        return {
            "reward": None,
            "reason": "",
            "judge_status": "error",
            "judge_meta": {"error": f"{type(exc).__name__}: {exc}"},
        }


def run_judge(
    rows: Sequence[dict] | Any,
    judge: Callable[[dict], Any],
    *,
    judge_name: str | None = None,
    source: str = "grade",
    model: str | None = None,
    run_id: str | None = None,
    concurrency: int = 8,
    timeout: float | None = None,
    version: str | None = None,
    scale: tuple[float, float] | None = None,
    tools: Sequence[dict] | Sequence[str] | None = None,
) -> ScoredData:
    """Score trajectories with any judge. Originals are left unmodified.

    ``tools=`` is the agent's declared tool list (or names); with it the
    result's ``warnings`` also say which declared tools no rollout called.
    Passing the ``SimulationData`` itself as ``rows`` supplies it.

    Each scored row is a copy of the input row plus ``reward``, ``reason``,
    ``judge_status``, ``judge_meta``, and a ``lineage`` record naming the
    scoring run, its source (grade or eval), the judged model, and the
    parent trajectory. Rows whose judge result breaks the contract keep
    ``reward=None`` and a non-ok status; they are counted, not hidden.
    ``version`` names the judge's version (model, prompt hash, whatever
    would change its labels); it lands in ``lineage.judge_version`` and
    reads back as ``Judgment.scorer.version``.
    """
    # A SimulationData passed whole supplies its rows and its declared tools.
    if tools is None:
        declared = getattr(rows, "declared_tools", None)
        if declared:
            tools = sorted(str(t) for t in declared)
    trajectories = getattr(rows, "trajectories", None)
    if trajectories is not None and not isinstance(rows, (list, tuple)):
        rows = trajectories
    src_rows = [r for r in rows if isinstance(r, dict)]
    rid = run_id or f"score_{uuid.uuid4().hex[:12]}"
    # A function judge is named by __name__; a Verifier is an instance and
    # carries .name instead, so without the second fallback every verifier
    # -graded row records the same "judge" and the scored rows no longer say
    # what checked them.
    name = judge_name or getattr(judge, "__name__", "") or _instance_name(judge) or "judge"
    if name == "<lambda>":
        name = "lambda_judge"
    # A Verifier says what it is (``kind="rule"``); a function judge does
    # not, and the schema then infers "judge" from the name. Stamp the
    # declared kind so a verifier does not read back as a model judge (#250).
    kind = getattr(judge, "kind", None)
    scorer_kind = kind if kind in ("rule", "reward_model", "human") else None
    verdicts: list[dict[str, Any]]
    if concurrency > 1 and len(src_rows) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_score_one, judge, r, scale) for r in src_rows]
            verdicts = []
            for future in futures:
                try:
                    verdicts.append(future.result(timeout=timeout))
                except concurrent.futures.TimeoutError:
                    verdicts.append(
                        {
                            "reward": None,
                            "reason": "",
                            "judge_status": "timeout",
                            "judge_meta": {"timeout_s": timeout},
                        }
                    )
    else:
        verdicts = [_score_one(judge, r, scale) for r in src_rows]
    scored: list[dict] = []
    for i, (row, verdict) in enumerate(zip(src_rows, verdicts)):
        out = dict(row)
        out["reward"] = verdict["reward"]
        if verdict["reason"]:
            out["reason"] = verdict["reason"]
        out["judge_status"] = verdict["judge_status"]
        out["judge_name"] = name
        meta = dict(verdict["judge_meta"] or {})
        if scorer_kind:
            meta["scorer_kind"] = scorer_kind
        if meta:
            out["judge_meta"] = meta
        fc = (verdict["judge_meta"] or {}).get("failure_class")
        if fc:
            out["failure_class"] = str(fc)
        # normalize_judge_result sweeps every non-reward key into judge_meta,
        # so a judge that returns markers left them where marker_summary does
        # not look: it reads row["markers"], the flat shape to_row emits. Lift
        # them onto the row. The judge's markers win on a name collision,
        # because it just measured the row.
        markers = (verdict["judge_meta"] or {}).get("markers")
        if isinstance(markers, dict) and markers:
            prior = row.get("markers")
            merged = dict(prior) if isinstance(prior, dict) else {}
            merged.update({str(k): v for k, v in markers.items()})
            out["markers"] = merged
        parent = row.get("scenario_id") or row.get("id") or f"row_{i}"
        lineage = dict(row.get("lineage") or {})
        lineage.update(
            {"scoring_run_id": rid, "source": source, "judge": name, "parent": str(parent)}
        )
        if model:
            lineage["model"] = model
        if version:
            lineage["judge_version"] = version
        if row.get("lineage", {}).get("scoring_run_id"):
            lineage["prior_scoring_run_id"] = row["lineage"]["scoring_run_id"]
        out["lineage"] = lineage
        scored.append(out)
    result = ScoredData(scored, run_id=rid, source=source, judge_name=name, model=model)
    # A confident pass@1 on rows where the agent never touched a tool, or
    # a marker that fired on no row, is the most expensive eval failure
    # there is: it reads as a result. Say so once, and name the fix. The
    # declared tools come from ``tools=`` or off a SimulationData.
    result.warnings = coverage_warnings(scored, tools=tools)
    for note in result.warnings:
        log.warning(note)
    return result


def evaluate(
    rows: Sequence[dict] | Any = None,
    judge: Callable[[dict], Any] | None = None,
    *,
    grader: Callable[[dict], Any] | None = None,
    model: str | None = None,
    eval_set: Sequence[Any] | None = None,
    judge_name: str | None = None,
    run_id: str | None = None,
    concurrency: int = 8,
    timeout: float | None = None,
    scale: tuple[float, float] | None = None,
    tools: Sequence[dict] | Sequence[str] | None = None,
) -> ScoredData:
    """Judge held-out rollouts under the exact contract ``grade`` uses.

    Reach for it to score rows that did not come from the run in hand:
    production traces, another run's rollouts, a frozen eval set. Same
    engine, same schema; only the lineage source differs. It returns a
    ``ScoredData``: ``rows`` (scored copies, each with ``reward``,
    ``reason``, ``judge_status``, ``judge_meta`` and a ``lineage`` record;
    a judge error is marked on its row, never coerced to 0), ``warnings``
    (read them before the numbers: no rollout called a tool, a declared
    tool none touched, a marker that fired on no row), ``eval_coverage``
    when ``eval_set`` was given, and ``traces``, which feeds
    ``simulate(traces=...)`` to close the loop.

    * ``rows``: the rollouts, or the ``SimulationData`` holding them,
      which also supplies ``tools``.
    * ``judge`` or ``grader``: the judge callable, either spelling, not
      both. ``grader`` is the doctrine-sketch name.
    * ``eval_set``: the frozen evaluation set, as prompt strings or rows.
      The result's ``eval_coverage`` reports which asks the rollouts
      actually covered and the gap, instead of letting a silent partial
      eval pass as a full one.
    * ``tools``: the agent's declared tool list (schemas or names), so a
      declared tool no rollout called is warned about.
    * ``judge_name``, ``model``, ``run_id``: recorded in each row's
      ``lineage`` as the judge, the judged model, and the scoring run
      (a fresh id per call unless pinned).
    * ``scale``: ``(lo, hi)`` reads a numeric verdict as a rating on that
      scale and maps it to a 0 to 1 reward, keeping the raw rating in
      ``judge_meta``.
    * ``concurrency`` (8) and ``timeout``: judge calls in flight and
      seconds per call.

    ```python
    scored = wai.evaluate(traces, judge=my_judge, eval_set=asks)
    print(scored.warnings, wai.pass_at(scored.rows).pass_at_1)
    ```
    """
    if tools and not all(isinstance(t, str) for t in tools):
        tools = _tool_schemas(tools)
    if grader is not None and judge is not None:
        raise ValueError("pass judge= or grader=, not both")
    judge = judge if judge is not None else grader
    if judge is None:
        raise ValueError("evaluate() needs a judge (judge= or grader=)")
    scored = run_judge(
        rows,
        judge,
        judge_name=judge_name,
        source="eval",
        model=model,
        run_id=run_id,
        concurrency=concurrency,
        timeout=timeout,
        scale=scale,
        tools=tools,
    )
    if eval_set is not None:
        wanted = {
            " ".join(
                str(item.get("prompt") if isinstance(item, dict) else item or "").lower().split()
            )
            for item in eval_set
        }
        wanted.discard("")
        got = {" ".join(str(r.get("prompt") or "").lower().split()) for r in scored.rows}
        missing = sorted(wanted - got)
        scored.eval_coverage = {
            "eval_set": len(wanted),
            "covered": len(wanted) - len(missing),
            "missing": missing[:20],
        }
    return scored


def _pair_key(row: dict) -> str:
    return " ".join(str(row.get("prompt") or "").lower().split())


def _score(row: dict) -> float | None:
    reward = row.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        return None
    value = float(reward)
    return value if 0.0 <= value <= 1.0 else None


def _model_of(row: dict) -> str | None:
    model = row.get("model_version")
    return str(model) if model else None


def length_confound_warning(chosen_longer: int, n: int) -> str | None:
    """The length-confound note ``build_preference_pairs`` and ``export_preference`` share.

    Fires when the chosen side is longer in three quarters of eight or more
    pairs, or in *every* pair once there are at least three: a total confound
    is a confound at any size, and a small hand-built set is exactly where it
    goes unnoticed (Lambert 2025, chapter Direct Alignment).
    """
    if n <= 0:
        return None
    frac = chosen_longer / n
    if (n >= LENGTH_CONFOUND_MIN_PAIRS and frac >= LENGTH_CONFOUND_SHARE) or (
        n >= LENGTH_CONFOUND_ALL_FROM and chosen_longer == n
    ):
        return (
            f"chosen is the longer reply in {chosen_longer}/{n} pairs; a preference "
            "trainer learns length before behavior (Lambert 2025, chapter Direct "
            "Alignment)"
        )
    return None


def _first_turn(row: dict) -> str:
    """What the policy emitted first, read the way the hosted DPO trainer
    reads it: the first tool step as its call, else the first assistant
    text, else ``final_text``."""
    steps = [s for s in (row.get("steps") or []) if isinstance(s, dict)]
    for step in steps:
        if step.get("tool"):
            args = step.get("arguments")
            if args is None:
                args = step.get("args")
            return json.dumps({"name": step["tool"], "arguments": args or {}}, sort_keys=True)
    for step in steps:
        if str(step.get("text") or "").strip():
            return str(step["text"]).strip()
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            text = str(message.get("content") or "").strip()
            if text:
                return text
    return str(row.get("final_text") or "").strip()


def first_turn_note(identical: int, n: int) -> str:
    """The warning for pairs whose first assistant turns read the same."""
    if not identical:
        return ""
    left = n - identical
    return (
        f"{identical}/{n} pairs have identical first assistant turns (same opening tool call "
        "or line); their contrast is later in the rollout. The hosted DPO trainer compares "
        f"first turns only and will drop them, leaving {left} (it needs at least 8). Keep "
        "[p for p in pairs if p['first_turn_differs']] to see what it will train on, or "
        "export_preference(pairs) for a trainer that reads whole conversations."
    )


def build_preference_pairs(
    rows: Sequence[dict],
    *,
    max_pairs_per_prompt: int = 1,
    min_margin: float = 1.0,
    length_match: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """Build same-task chosen/rejected pairs for preference training (DPO-style).

    Reach for it after grading a run with several rollouts per ask: the
    contrast between a pass and a fail on the same prompt is the training
    signal, so failures are supply here, not waste. It returns ``(pairs,
    report)``: each pair carries ``prompt``, ``chosen``, ``rejected``,
    the scores and lineage listed below, and both parents' lineage; the
    report counts the pairs, the prompts that had a contrast, and how
    often chosen is still the longer side.

    A pair exists only where the same prompt has two trajectories whose
    rewards differ by at least ``min_margin``. Rows without a valid judge
    result never pair.

    * ``min_margin`` (1.0): pairs 1-labeled with 0-labeled rows only;
      ``0.5`` also admits partial-credit rows against a full pass or fail.
    * ``max_pairs_per_prompt`` (1): how many pairs one prompt may
      contribute.
    * ``length_match`` (``True``): each chosen row takes the rejected row
      closest to it in length. DPO exploits a length gap faster than it
      learns the behavior (Lambert 2025, chapter Direct Alignment).

    Each pair keeps what the trainer and the reviewer need to trust it:

    * ``chosen_score`` / ``rejected_score`` / ``margin``: the raw scores
      and their gap, so a margin-aware loss (Llama 2 style) can use them
      and a reviewer can see how far apart the two really are.
    * ``chosen_model`` / ``rejected_model`` / ``same_policy``: which policy
      produced each side. Preference data works best when both sides
      come from the policy being trained (Lambert et al. 2024, Tulu 3,
      arXiv:2411.15124; Lambert 2025, chapter Preference Data); a
      mixed pair is still a pair, but it is labeled as off-policy.
    * ``length_delta``: chosen reply chars minus rejected, and the report
      says how often chosen is still the longer side.

    ```python
    pairs, report = wai.build_preference_pairs(data.rows(), min_margin=1.0)
    print(len(pairs), report)
    ```
    """
    if min_margin <= 0:
        raise ValueError("min_margin must be positive; equal scores carry no preference")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("judge_status", "ok") != "ok":
            continue
        if _score(row) is None:
            continue
        key = _pair_key(row)
        if not key:
            continue
        groups.setdefault(key, []).append(row)

    from .hygiene import reply_length

    pairs: list[dict] = []
    contrast_prompts = 0
    for members in groups.values():
        ranked = sorted(members, key=lambda r: -(_score(r) or 0.0))
        used: set[int] = set()
        made = 0
        for ci, chosen in enumerate(ranked):
            if made >= max_pairs_per_prompt:
                break
            if ci in used:
                continue
            c_score = _score(chosen) or 0.0
            candidates = [
                (ri, r)
                for ri, r in enumerate(ranked)
                if ri not in used and ri != ci and c_score - (_score(r) or 0.0) >= min_margin
            ]
            if not candidates:
                continue
            if length_match:
                c_len = reply_length(chosen)
                ri, rejected = min(candidates, key=lambda c: abs(reply_length(c[1]) - c_len))
            else:
                ri, rejected = candidates[0]
            used.update({ci, ri})
            made += 1
            r_score = _score(rejected) or 0.0
            c_model, r_model = _model_of(chosen), _model_of(rejected)
            pairs.append(
                {
                    "prompt": chosen.get("prompt"),
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_score": c_score,
                    "rejected_score": r_score,
                    "margin": round(c_score - r_score, 6),
                    "chosen_model": c_model,
                    "rejected_model": r_model,
                    "same_policy": (c_model == r_model) if c_model and r_model else None,
                    "length_delta": reply_length(chosen) - reply_length(rejected),
                    "first_turn_differs": _first_turn(chosen) != _first_turn(rejected),
                    "chosen_reason": str(chosen.get("reason") or ""),
                    "rejected_reason": str(rejected.get("reason") or ""),
                    "rejected_failure_class": rejected.get("failure_class"),
                    "lineage": {
                        "chosen": chosen.get("lineage"),
                        "rejected": rejected.get("lineage"),
                    },
                }
            )
        if made:
            contrast_prompts += 1

    n = len(pairs)
    deltas = sorted(p["length_delta"] for p in pairs)
    chosen_longer = sum(1 for d in deltas if d > 0)
    same_policy = sum(1 for p in pairs if p["same_policy"] is True)
    mixed_policy = sum(1 for p in pairs if p["same_policy"] is False)
    partial = sum(
        1
        for p in pairs
        if p["chosen_score"] not in (0.0, 1.0) or p["rejected_score"] not in (0.0, 1.0)
    )
    identical = sum(1 for p in pairs if not p["first_turn_differs"])
    warnings: list[str] = []
    length_note = length_confound_warning(chosen_longer, n)
    if length_note:
        warnings.append(length_note)
    if identical:
        warnings.append(first_turn_note(identical, n))
    if mixed_policy:
        warnings.append(
            f"{mixed_policy}/{n} pairs mix policies (chosen and rejected from different "
            "models); on-policy pairs train better (Lambert 2025, chapter "
            "Preference Data)"
        )
    from .optimize import _eval_sourced_warning, eval_sourced

    eval_pairs = sum(1 for p in pairs if eval_sourced([p["chosen"], p["rejected"]]))
    if eval_pairs:
        warnings.append(_eval_sourced_warning(eval_pairs, "pair(s)"))
    report = {
        "pairs": n,
        "prompts_seen": len(groups),
        "prompts_with_contrast": contrast_prompts,
        "min_margin": min_margin,
        "mean_margin": round(sum(p["margin"] for p in pairs) / n, 4) if n else None,
        "partial_score_pairs": partial,
        "first_turn_identical": identical,
        "trainer_pairs": n - identical,
        "same_policy_pairs": same_policy,
        "mixed_policy_pairs": mixed_policy,
        "eval_sourced": eval_pairs,
        "length": {
            "median_delta": deltas[n // 2] if n else None,
            "chosen_longer": chosen_longer,
            "chosen_longer_frac": round(chosen_longer / n, 3) if n else None,
        },
        "warnings": warnings,
        "note": (
            "pairs require the same prompt to have two rollouts whose "
            f"rewards differ by at least {min_margin}; raise "
            "rollouts_per_request to create contrast"
            if not pairs
            else ""
        ),
    }
    return pairs, report


__all__ = [
    "ScoredData",
    "build_preference_pairs",
    "evaluate",
    "normalize_judge_result",
    "run_judge",
]
