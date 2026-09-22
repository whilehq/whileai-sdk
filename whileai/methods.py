"""Training methods as objects, and the config a trainer reads from them.

    import whileai as wai

    teacher = wai.Endpoint(url="https://my-vllm.example/v1", model="Qwen/Qwen3-32B")
    cfg = wai.prime_rl_config("refunds-v1", wai.OPD(teacher), model="Qwen/Qwen3-4B", out="opd.toml")
    print(cfg)                 # what was written, which knobs the trainer reads, which it ignores
    # then, on a box with two GPUs and your keys:  uv run rl @ opd.toml

``method=`` takes a string (``"grpo"``, ``"sft"``, ...) or one of the objects
here. An object is the string with its knobs attached, the way
``torch.optim.Adam(lr=)`` is "adam" with its knobs: every default is named
and cited in ``defaults.py``, a bad value is refused on construction with
the fix in the message, and the object prints. Nothing here trains. The
trainer is prime-rl, TRL or Tinker, on your GPUs with your keys; the
objects say what to run and ``prime_rl_config`` says it in the
trainer's own words.

* ``OPD``, on-policy distillation. The student samples, a frozen
  teacher scores every sampled token, and the loss is the per-token
  reverse KL (Agarwal et al. 2023, arXiv:2306.13649; Thinking Machines
  2025). Qwen3 reports the same AIME score as RL at a tenth of the GPU
  hours (arXiv:2505.09388).
* ``OPSD``, on-policy self-distillation. The teacher is the same
  model given privileged context the student never sees: a passing
  demonstration (Shenfeld et al. 2026, arXiv:2601.19897), the reference
  answer (Zhao et al. 2026, arXiv:2601.18734), or a successful rollout
  plus the environment's feedback (Hübotter et al. 2026,
  arXiv:2601.20802). It learns where GRPO has no gradient, and it hurts
  thinking models (Kaur et al. 2026, arXiv:2607.05184), so the writer
  says so.
* ``Async``, any method with a bound on how many optimizer steps a
  rollout may lag the policy, and the per-token correction for the gap
  (Noukhovitch et al. 2024, arXiv:2410.18252; Khatri et al. 2025,
  arXiv:2510.13786).
* ``FlashReinforce``, ``SAO`` and ``BPCO``, the single-rollout methods:
  one trajectory per prompt, no group to take a baseline over, so the
  baseline is the batch mean (FlashReinforce, Hu et al. 2026) or a critic
  (SAO, Hou et al. 2026, arXiv:2607.07508; BPCO, Qi et al. 2026,
  arXiv:2608.23566). They are how a production trace, which comes one
  per prompt and cannot be re-run, becomes a training signal. Each
  object's ``update(batch)`` is the update rule itself, in plain Python,
  so a trainer (or a test) can apply it to any batch of trajectories and
  read what it kept and why. On prime-rl all three are refused, with the
  reason and the fix in the message: its reward advantages are group
  relative (``grpo``, ``max_rl``: reward minus the group mean, zero over
  a group of one) or an EMA baseline (``rae``), it hosts no value model,
  and its losses mask per token (``ipo``, ``icepop``) with no sequence
  trust region and no clip. The nearest thing it runs is ``"rae"``.
* ``GroupwiseGrading``, a grader that tells passing rollouts apart, in
  the reward (GRS) or in the advantage (GAR) (MiMo-V2.6, Xiaomi 2026,
  technical report section 4.3). It lives in ``whileai.groupwise`` and
  is re-exported here; it shapes a TRL reward function rather than
  writing a prime-rl config.
* ``prime_rl_config``, the TOML prime-rl reads, from a method object
  and a taskset.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, NoReturn

from .groupwise import (
    GroupwiseGrading,
    GroupwiseStats,
    SpreadReport,
    factors_from_ranking,
    redistribute,
    spread,
)
from .models import Backend
from .simulations.defaults import (
    ASYNC_CORRECTION,
    ASYNC_ICEPOP_RATIO,
    ASYNC_IPO_EPS,
    ASYNC_OFF_POLICY_STEPS,
    ASYNC_TIS_CAP,
    BPCO_CLIP,
    BPCO_CRITIC_LEARNING_RATE,
    BPCO_CRITIC_WARMUP,
    BPCO_GAE_ALPHA,
    BPCO_LEARNING_RATE,
    BPCO_LOG_RATIO_CAP,
    BPCO_MAX_TOKENS,
    BPCO_REWARD_RANGE,
    BPCO_TEMPERATURE,
    FLASH_REINFORCE_BATCH,
    FLASH_REINFORCE_LEARNING_RATE,
    FLASH_REINFORCE_LEARNING_RATE_LORA,
    FLASH_REINFORCE_LOG_RATIO_CLAMP,
    FLASH_REINFORCE_MAX_TOKENS,
    FLASH_REINFORCE_OFF_POLICY_STEPS,
    FLASH_REINFORCE_PROBABILITY_FLOOR,
    FLASH_REINFORCE_TEMPERATURE,
    FLASH_REINFORCE_TRUST,
    MESSAGE_EXAMPLES,
    OPD_DIVERGENCE,
    OPD_LEARNING_RATE_FULL,
    OPD_LEARNING_RATE_LORA,
    OPD_MAX_TOKENS,
    OPD_SAMPLES,
    OPD_TEMPERATURE,
    OPD_TOP_K,
    OPSD_ANCHOR,
    OPSD_ANCHOR_ALPHA,
    OPSD_DIVERGENCE,
    OPSD_LEARNING_RATE,
    OPSD_MAX_TOKENS,
    OPSD_PRIVILEGED,
    OPSD_SAMPLES,
    OPSD_TEMPERATURE,
    OPSD_TEMPLATE,
    PRIME_RL_BATCH,
    PRIME_RL_EVAL_EXAMPLES,
    PRIME_RL_EVAL_GROUP,
    PRIME_RL_GPUS,
    PRIME_RL_LEARNING_RATE_FULL,
    PRIME_RL_LEARNING_RATE_LORA,
    PRIME_RL_SEQ_LEN,
    PRIME_RL_STEPS,
    RL_ROLLOUTS_PER_PROMPT,
    SAO_CRITIC_LEARNING_RATE,
    SAO_CRITIC_STEPS,
    SAO_CRITIC_WARMUP,
    SAO_GAE_ALPHA,
    SAO_GAMMA,
    SAO_LEARNING_RATE,
    SAO_MAX_TOKENS,
    SAO_RATIO,
    SAO_RATIO_CODING,
    SAO_TEMPERATURE,
    TRAINING_LORA_ALPHA,
    TRAINING_LORA_RANK,
)

DIVERGENCES = ("reverse_kl", "forward_kl", "jsd")
PRIVILEGED = ("demonstration", "reference", "hint", "feedback")
ANCHORS = ("ema", "initial", "live")
CORRECTIONS = ("ipo", "icepop", "tis")
#: the reward-based algorithms prime-rl names, usable as the string form
#: of ``method=`` and inside ``Async``: ``grpo`` and ``max_rl`` take the
#: group mean as the baseline, ``rae`` an EMA of the agent's past rewards
#: (SPIRAL, arXiv:2506.24119), the one that runs at ``group_size = 1``
#: (prime-rl ``configs/algorithm.py``, ``RAEAlgoConfig``)
PRIME_RL_ALGORITHMS = ("grpo", "max_rl", "rae")

ISSUE = "https://github.com/whilehq/whileai-sdk/issues/564"


def _teacher_ref(teacher: Any) -> tuple[str, str]:
    """(model, base_url) for a frozen teacher, from a backend object or a
    ``vllm:<model>@<url>`` spec. Only a server you run exposes the
    prompt log-probabilities a distillation teacher is scored on."""
    if isinstance(teacher, Backend):
        url = getattr(teacher, "url", "")
        if url and teacher.model:
            return str(teacher.model), str(url)
        raise ValueError(
            f"OPD teacher: {teacher!r} has no URL. The teacher is scored on the student's own "
            "tokens, which needs a server that returns prompt log-probabilities (vLLM, SGLang); "
            "pass wai.Endpoint(url='https://.../v1', model='<served name>')."
        )
    if isinstance(teacher, str):
        m = re.fullmatch(r"(?:vllm:)?([^@\s]+)@(\S+)", teacher.strip())
        if m:
            return m.group(1), m.group(2)
        raise ValueError(
            f"OPD teacher: {teacher!r} is not a served model. Pass wai.Endpoint(url=, model=) or "
            "the spec 'vllm:<served name>@https://.../v1'."
        )
    raise TypeError(
        "OPD teacher: pass wai.Endpoint(url=, model=) or 'vllm:<served name>@<url>', "
        f"got {type(teacher).__name__}"
    )


def _check_choice(what: str, value: str, choices: tuple[str, ...]) -> str:
    v = str(value).lower()
    if v not in choices:
        raise ValueError(f"{what} must be one of {', '.join(choices)}; got {value!r}")
    return v


@dataclass(frozen=True)
class OPD:
    """On-policy distillation: the student samples, a frozen teacher scores.

    ``teacher`` is a served model you run (``wai.Endpoint(url=, model=)``
    or ``'vllm:<name>@<url>'``); it is scored on the student's tokens, so
    it needs prompt log-probabilities, which the hosted chat APIs do not
    return. The per-token loss is ``divergence`` (``reverse_kl``: Agarwal
    et al. 2023, arXiv:2306.13649, Table 1; Thinking Machines 2025) over the
    teacher's top ``top_k`` tokens (Li et al. 2026, arXiv:2604.13016; Fu et
    al. 2026, arXiv:2603.25562), on ``samples`` rollouts per prompt drawn
    at ``temperature`` and capped at ``max_tokens``. ``learning_rate``
    left ``None`` is 1e-4 for an adapter and 1e-6 for full weights
    (``OPD_LEARNING_RATE_LORA``, ``OPD_LEARNING_RATE_FULL``).

    Before spending the GPU, know two things the papers say decide the
    run: the teacher has to beat the student on these tasks (the student
    saturates at the teacher's ceiling), and the two have to share a
    tokenizer (a mismatch silently drops the signal, SimCT, arXiv:2605.07711).
    The tokenizer check is the trainer's; the pass-rate check is
    ``wai.pass_at`` on a teacher run of the holdout, until ``train``
    runs it for you (issue 564).
    """

    teacher: Backend | str
    divergence: str = OPD_DIVERGENCE
    top_k: int = OPD_TOP_K
    samples: int = OPD_SAMPLES
    temperature: float = OPD_TEMPERATURE
    max_tokens: int = OPD_MAX_TOKENS
    learning_rate: float | None = None

    name: ClassVar[str] = "opd"

    def __post_init__(self) -> None:
        _teacher_ref(self.teacher)
        object.__setattr__(
            self, "divergence", _check_choice("divergence", self.divergence, DIVERGENCES)
        )
        if int(self.top_k) < 1:
            raise ValueError(
                f"top_k must be at least 1 (OPD_TOP_K is {OPD_TOP_K}); got {self.top_k}"
            )
        if int(self.samples) < 1:
            raise ValueError(
                f"samples must be at least 1 (OPD_SAMPLES is {OPD_SAMPLES}); got {self.samples}"
            )
        if not 0 < float(self.temperature) <= 2:  # noqa: PLR2004  # the sampler's range
            raise ValueError(f"temperature must be in (0, 2]; got {self.temperature}")
        if int(self.max_tokens) < 1:
            raise ValueError(f"max_tokens must be positive; got {self.max_tokens}")

    @property
    def teacher_ref(self) -> tuple[str, str]:
        """(served model name, base URL) of the teacher."""
        return _teacher_ref(self.teacher)

    def default_learning_rate(self, lora: bool) -> float:
        if self.learning_rate is not None:
            return float(self.learning_rate)
        return OPD_LEARNING_RATE_LORA if lora else OPD_LEARNING_RATE_FULL

    def __str__(self) -> str:
        model, url = self.teacher_ref
        return (
            f"OPD(teacher={model} @ {url}, {self.divergence}, top_k={self.top_k}, "
            f"samples={self.samples}, temperature={self.temperature}, max_tokens={self.max_tokens})"
        )


@dataclass(frozen=True)
class OPSD:
    """On-policy self-distillation: the teacher is the student with a hint.

    ``privileged`` names what the teacher sees and the student does not: the
    task field the trainer reads it from (``info`` first, then the task's
    top-level fields, so a public taskset's ``answer`` works as given). The
    four named forms are: ``demonstration`` (a
    passing rollout of the same task, SDFT, Shenfeld et al. 2026,
    arXiv:2601.19897), ``reference`` (the answer, Zhao et al. 2026,
    arXiv:2601.18734), ``hint`` (Penaloza et al. 2026, arXiv:2602.04942) or
    ``feedback`` (a successful rollout plus the environment's error text,
    Hübotter et al. 2026, arXiv:2601.20802). ``template`` is the system
    message that carries it, with ``{demonstration}`` for the text.

    ``anchor`` is what the teacher's weights are: ``"ema:0.01"`` (an
    exponential moving average of the student, the SDFT and SDPO choice;
    unregularized self-distillation diverges, arXiv:2601.20802), ``"initial"``
    (the starting weights, Zhao et al. 2026) or ``"live"`` (the current
    student, what prime-rl runs). ``divergence`` is ``reverse_kl`` (SDFT,
    SDPO, prime-rl) or ``forward_kl`` (Zhao et al. 2026). ``samples`` is
    1 by default: there is no group baseline to form.

    Two things the papers say before you run it: it needs in-context
    learning strong enough to use the hint, so under about 7B it fails
    (arXiv:2601.19897, arXiv:2601.20802), and on a thinking model it costs
    points (Kaur et al. 2026, arXiv:2607.05184, 5.7 on Qwen3-4B/8B; Kim et
    al. 2026, arXiv:2603.24472), so pair it with a GRPO arm on the same
    holdout before believing a number.
    """

    privileged: str = OPSD_PRIVILEGED
    divergence: str = OPSD_DIVERGENCE
    anchor: str = f"{OPSD_ANCHOR}:{OPSD_ANCHOR_ALPHA}"
    samples: int = OPSD_SAMPLES
    temperature: float = OPSD_TEMPERATURE
    max_tokens: int = OPSD_MAX_TOKENS
    template: str = OPSD_TEMPLATE
    learning_rate: float | None = None

    name: ClassVar[str] = "opsd"

    def __post_init__(self) -> None:
        key = str(self.privileged).strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", key):
            raise ValueError(
                f"privileged must be one of {', '.join(PRIVILEGED)} or the name of the task "
                f"field that carries the teacher's context; got {self.privileged!r}"
            )
        object.__setattr__(self, "privileged", key)
        object.__setattr__(
            self, "divergence", _check_choice("divergence", self.divergence, DIVERGENCES)
        )
        self.anchor_parts()  # validates
        if int(self.samples) < 1:
            raise ValueError(
                f"samples must be at least 1 (OPSD_SAMPLES is {OPSD_SAMPLES}); got {self.samples}"
            )
        if not 0 < float(self.temperature) <= 2:  # noqa: PLR2004  # the sampler's range
            raise ValueError(f"temperature must be in (0, 2]; got {self.temperature}")
        if "{demonstration}" not in self.template:
            raise ValueError(
                "template must contain '{demonstration}', where the privileged text goes"
            )

    def anchor_parts(self) -> tuple[str, float | None]:
        """(kind, alpha): ``("ema", 0.01)``, ``("initial", None)`` or ``("live", None)``."""
        kind, _, alpha = str(self.anchor).lower().partition(":")
        kind = _check_choice("anchor", kind, ANCHORS)
        if kind != "ema":
            if alpha:
                raise ValueError(f"anchor {kind!r} takes no rate; got {self.anchor!r}")
            return kind, None
        try:
            rate = float(alpha) if alpha else OPSD_ANCHOR_ALPHA
        except ValueError:
            raise ValueError(f"anchor 'ema:<rate>' needs a number; got {self.anchor!r}") from None
        if not 0 < rate < 1:
            raise ValueError(
                f"anchor ema rate must be in (0, 1) (OPSD_ANCHOR_ALPHA is {OPSD_ANCHOR_ALPHA}); got {rate}"
            )
        return "ema", rate

    def default_learning_rate(self, lora: bool) -> float:
        return float(self.learning_rate) if self.learning_rate is not None else OPSD_LEARNING_RATE

    def __str__(self) -> str:
        return (
            f"OPSD(privileged={self.privileged}, {self.divergence}, anchor={self.anchor}, "
            f"samples={self.samples}, temperature={self.temperature}, max_tokens={self.max_tokens})"
        )


@dataclass(frozen=True)
class Async:
    """A method trained on rollouts that may lag the policy by a bounded number of steps.

    ``method`` is what to train (``"grpo"``, ``"max_rl"``, ``"rae"``, an
    ``OPD`` or an ``OPSD``). ``off_policy_steps`` is the bound: a rollout
    sampled more than this many optimizer steps before the update that
    consumes it is dropped (8: ScaleRL, arXiv:2510.13786; AReaL,
    arXiv:2505.24298, 4 for code; one step costs nothing, Noukhovitch et
    al. 2024, arXiv:2410.18252). ``correction`` is the per-token fix for
    the sampler/trainer gap, which exists even at zero staleness (Yao et
    al. 2025): ``ipo`` masks a token whose probability moved more than
    ``eps`` (prime-rl's default), ``icepop`` masks a token whose
    trainer/sampler ratio leaves ``ratio`` (Ring-1T, arXiv:2510.18855),
    ``tis`` caps the ratio at ``cap`` (verl). Which the trainer offers is
    the trainer's; ``prime_rl_config`` says.

    prime-rl trains one step ahead of its sampler by design, so a plain
    ``"grpo"`` config already runs under the default bound; ``Async`` is
    how you move it or read it off the config.
    """

    method: OPD | OPSD | str = "grpo"
    off_policy_steps: int = ASYNC_OFF_POLICY_STEPS
    correction: str = ASYNC_CORRECTION
    eps: float = ASYNC_IPO_EPS
    ratio: tuple[float, float] = ASYNC_ICEPOP_RATIO
    cap: float = ASYNC_TIS_CAP

    name: ClassVar[str] = "async"

    def __post_init__(self) -> None:
        if isinstance(self.method, Async):
            raise TypeError("Async wraps a method, not another Async")
        if isinstance(self.method, str):
            object.__setattr__(
                self, "method", _check_choice("method", self.method, PRIME_RL_ALGORITHMS)
            )
        elif not isinstance(self.method, (OPD, OPSD)):
            raise TypeError(
                f"method must be a string, wai.OPD or wai.OPSD; got {type(self.method).__name__}"
            )
        if int(self.off_policy_steps) < 0:
            raise ValueError(f"off_policy_steps must be 0 or more; got {self.off_policy_steps}")
        object.__setattr__(
            self, "correction", _check_choice("correction", self.correction, CORRECTIONS)
        )
        if not 0 < float(self.eps) <= 1:
            raise ValueError(
                f"eps is a probability difference in (0, 1] (ASYNC_IPO_EPS is {ASYNC_IPO_EPS}); got {self.eps}"
            )
        lo, hi = (float(x) for x in self.ratio)
        if not 0 < lo <= 1 <= hi:
            raise ValueError(
                f"ratio must bracket 1, (low, high) with 0 < low <= 1 <= high; got {self.ratio}"
            )
        object.__setattr__(self, "ratio", (lo, hi))
        if float(self.cap) < 1:
            raise ValueError(
                f"cap must be at least 1 (ASYNC_TIS_CAP is {ASYNC_TIS_CAP}); got {self.cap}"
            )

    @property
    def inner_name(self) -> str:
        return self.method if isinstance(self.method, str) else self.method.name

    def __str__(self) -> str:
        fix = {
            "ipo": f"ipo eps={self.eps}",
            "icepop": f"icepop ratio={self.ratio}",
            "tis": f"tis cap={self.cap}",
        }
        return f"Async({self.method}, off_policy_steps={self.off_policy_steps}, {fix[self.correction]})"


# --------------------------------------------------------------------------
# single-rollout methods: one trajectory per prompt
# --------------------------------------------------------------------------
#
# A trajectory is a plain dict, the shape a row already has:
#   reward             float, the trajectory's scalar reward
#   logprobs           list[float], per generated token, under the policy being trained
#   behavior_logprobs  list[float], per token, under the policy that sampled it
#                      (a row's ``token_logprobs``; equal to ``logprobs`` when on-policy)
#   values             list[float], optional, the critic's value at each token (SAO, BPCO)
#   action_mask        list[bool], optional, False on tokens the environment wrote
#                      (tool output, observations) so they carry no gradient
#
# Each method's ``update(batch)`` returns an ``Update``: the per-token
# coefficient that multiplies the gradient of the token's log-probability,
# with the baseline, the masks, the ratios and the 1/B and 1/T factors
# already applied. A trainer's loss is ``-(coefficient * logprob).sum()``
# with the coefficient held constant; a test applies the same numbers to a
# tabular softmax policy and watches the reward climb.


@dataclass
class Update:
    """What one single-rollout update does to a batch, and why.

    ``coefficients[i][t]`` multiplies the gradient of the log-probability
    of token ``t`` of trajectory ``i``; ``advantages[i][t]`` is the
    advantage before any ratio or mask; ``admitted[i]`` says whether the
    trajectory contributed at all; ``value_targets`` is what a critic
    trains toward, ``None`` for a critic-free method. ``stats`` holds the
    numbers a run page shows (admitted share, masked-token share, mean
    ratio) and ``notes`` says in words what was dropped and why.
    """

    method: str
    coefficients: list[list[float]]
    advantages: list[list[float]]
    admitted: list[bool]
    value_targets: list[list[float]] | None = None
    stats: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.coefficients)

    @property
    def n_admitted(self) -> int:
        return sum(1 for a in self.admitted if a)

    def __str__(self) -> str:
        lines = [f"{self.method} update: {self.n_admitted} of {self.n} trajectories admitted"]
        for k, v in self.stats.items():
            lines.append(f"  {k.replace('_', ' ')}: {v:.4g}")
        for note in self.notes:
            lines.append(f"  {note}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return "<pre>" + str(self).replace("<", "&lt;") + "</pre>"


def _trajectory_fields(batch: Sequence[Mapping[str, Any]], method: str) -> None:
    """Refuse a batch a single-rollout update cannot read, naming the row and the field."""
    if not batch:
        raise ValueError(f"{method}.update: the batch is empty; pass at least one trajectory")
    for i, traj in enumerate(batch):
        if "reward" not in traj:
            raise ValueError(f"{method}.update: trajectory {i} has no 'reward'")
        lp = traj.get("logprobs")
        if not isinstance(lp, Sequence) or isinstance(lp, str) or len(lp) == 0:
            raise ValueError(
                f"{method}.update: trajectory {i} needs 'logprobs', one per generated token, "
                "under the policy being trained"
            )
        for key in ("behavior_logprobs", "values", "action_mask"):
            v = traj.get(key)
            if v is not None and len(v) != len(lp):
                raise ValueError(
                    f"{method}.update: trajectory {i} has {len(v)} {key} for {len(lp)} logprobs; "
                    "one per token"
                )


def _ratio(log_diff: float) -> float:
    """``exp(log pi_theta - log pi_rollout)``; ``inf`` past the float range,
    which the band then masks like any other excursion."""
    try:
        return math.exp(log_diff)
    except OverflowError:
        return math.inf


def _bernoulli_kl(p: float, q: float) -> float:
    """The sampled-action Bernoulli KL proxy, Eq. (6) of Hu et al. 2026:
    ``d = p log(p/q) + (1-p) log((1-p)/(1-q))`` with ``p`` the sampler's
    probability of the token and ``q`` the learner's, each clamped to
    ``[FLASH_REINFORCE_PROBABILITY_FLOOR, 1 - FLASH_REINFORCE_PROBABILITY_FLOOR]``
    the way the reference loss does. Never below zero (a KL); the ``max``
    only absorbs rounding."""
    lo, hi = FLASH_REINFORCE_PROBABILITY_FLOOR, 1.0 - FLASH_REINFORCE_PROBABILITY_FLOOR
    p = min(max(p, lo), hi)
    q = min(max(q, lo), hi)
    return max(0.0, p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q)))


def _index_list(rows: Sequence[int], limit: int = 8) -> str:
    shown = ", ".join(str(i) for i in rows[:limit])
    return shown if len(rows) <= limit else f"{shown}, ... {len(rows) - limit} more"


@dataclass(frozen=True)
class FlashReinforce:
    """Critic-free REINFORCE on one rollout per prompt, safe under a stale sampler.

    One trajectory per prompt is the shape a production trace arrives in
    and the shape an asynchronous agent trainer produces: no group of
    siblings to take a baseline over, no critic to train. FlashReinforce
    takes the baseline from the batch instead (a trajectory counts as good
    when it beat the batch's mean reward), corrects each token for the gap
    between the policy that sampled it and the policy being trained, throws
    away any trajectory that gap has moved too far, and weights every kept
    trajectory the same whatever its length, so a long failure is not
    punished more than a short one.

    ``trust`` is the sequence trust region delta: the mean per-token drift
    a trajectory may show before it is masked whole (``math.inf`` turns the
    gate off). ``off_policy_steps`` is the lag, in optimizer steps, the
    method is built to absorb; a trainer that takes a bound reads it.
    ``temperature`` and ``max_tokens`` are the sampler's; the sampler is
    untruncated (top-p 1.0), because the ratio assumes the learner never
    puts mass where the sampler could not. ``learning_rate`` left ``None``
    is 1e-6 on full weights (``FLASH_REINFORCE_LEARNING_RATE``) and 1e-4
    on an adapter (``FLASH_REINFORCE_LEARNING_RATE_LORA``, untested). The
    paper's batch is 128 trajectories (``FLASH_REINFORCE_BATCH``); the
    batch you pass is the B below.

    The mechanism, for a batch of ``B`` trajectories with rewards ``R_i``,
    ``T_i`` action tokens each, sampler log-probabilities ``log mu`` and
    learner log-probabilities ``log pi``:

    * advantage ``A_i = R_i - mean_j R_j``, batch-centered, not divided by
      a standard deviation (Eq. 5);
    * per-token ratio ``rho_{i,t} = exp(log pi - log mu)`` (Eq. 1), never
      clipped, only kept finite by clamping the log-ratio to
      ``[-30, 30]`` (``FLASH_REINFORCE_LOG_RATIO_CLAMP``, Appendix A);
    * per-token drift ``d_{i,t} = p log(p/q) + (1-p) log((1-p)/(1-q))``
      with ``p = mu`` and ``q = pi`` of the sampled token (Eq. 6), its
      mean over the trajectory ``D_i`` (Eq. 7), and the mask
      ``m_i = 1[D_i <= trust]`` (Eq. 8);
    * the objective ``J = (1/B) sum_i (m_i A_i / T_i) sum_t rho_{i,t}``
      (Eq. 9), whose gradient with ``rho``, ``A`` and ``m`` held fixed is
      ``(1/B) sum_i (m_i A_i / T_i) sum_t rho_{i,t} grad log pi`` (Eq. 10).

    ``update(batch)`` returns exactly those numbers: ``coefficients[i][t]
    = m_i * A_i * rho_{i,t} / (T_i * B)``, zero on a token ``action_mask``
    turns off (tool output, observations: excluded from ``T_i`` and from
    ``D_i`` too), plus the advantages, the mask and the notes. A row with
    no ``behavior_logprobs`` is taken as on-policy (ratio 1, drift 0),
    which is only right when the sampler was this policy; store the
    sampler's own log-probabilities, never recomputed ones (Sec. 2.1).

    Citation: Hu, Zhang, Zhang, Xu, Zhang, Peng, Yu, Molchanov, Kautz and
    Dong 2026, FlashREINFORCE: Critic-Free Single-Rollout Asynchronous RL
    for Agentic Language Models, NVIDIA (no arXiv id as of 2026-09-21;
    https://yifanzhang-pro.github.io/FlashREINFORCE/FlashREINFORCE.pdf).
    Reference loss: github.com/yifanzhang-pro/FlashREINFORCE; trainer:
    github.com/NVIDIA-NeMo/labs-molt. Reported: stable through 6,000
    updates at lag 4 on DeepSeek-R1-Distill-Qwen-1.5B, 38.0 five-benchmark
    mean on Qwen2.5-Math-1.5B with half GRPO's rollouts, lag 8 on
    Qwen3-30B-A3B, 98.3/96.5 seen/unseen on ALFWorld.

    ```python
    import whileai as wai

    method = wai.FlashReinforce()          # trust=0.003, off_policy_steps=8
    batch = [
        {"reward": 1.0, "logprobs": [-0.5, -1.2, -0.3]},
        {"reward": 0.0, "logprobs": [-0.9, -0.4], "behavior_logprobs": [-0.8, -0.4]},
        {"reward": 1.0, "logprobs": [-0.1] * 20},
    ]
    update = method.update(batch)
    print(update)                          # 3 of 3 admitted, mean ratio, drift
    update.coefficients[0]                 # [A_0 * rho / (3 * 3), ...]
    ```
    """

    trust: float = FLASH_REINFORCE_TRUST
    off_policy_steps: int = FLASH_REINFORCE_OFF_POLICY_STEPS
    temperature: float = FLASH_REINFORCE_TEMPERATURE
    max_tokens: int = FLASH_REINFORCE_MAX_TOKENS
    learning_rate: float | None = None

    name: ClassVar[str] = "flash_reinforce"
    samples: ClassVar[int] = 1
    batch: ClassVar[int] = FLASH_REINFORCE_BATCH

    def __post_init__(self) -> None:
        trust = float(self.trust)
        if not trust > 0:  # also refuses nan
            raise ValueError(
                f"trust must be above 0 (FLASH_REINFORCE_TRUST is {FLASH_REINFORCE_TRUST}, the "
                f"mean sampled-action KL a trajectory may show; math.inf turns the gate off); "
                f"got {self.trust}"
            )
        object.__setattr__(self, "trust", trust)
        if int(self.off_policy_steps) < 0:
            raise ValueError(
                "off_policy_steps must be 0 or more (FLASH_REINFORCE_OFF_POLICY_STEPS is "
                f"{FLASH_REINFORCE_OFF_POLICY_STEPS}; 0 is fully on-policy); got {self.off_policy_steps}"
            )
        object.__setattr__(self, "off_policy_steps", int(self.off_policy_steps))
        if not 0 < float(self.temperature) <= 2:  # noqa: PLR2004  # the sampler's range
            raise ValueError(
                f"temperature must be in (0, 2] (FLASH_REINFORCE_TEMPERATURE is "
                f"{FLASH_REINFORCE_TEMPERATURE}); got {self.temperature}"
            )
        if int(self.max_tokens) < 1:
            raise ValueError(
                f"max_tokens must be positive (FLASH_REINFORCE_MAX_TOKENS is "
                f"{FLASH_REINFORCE_MAX_TOKENS}); got {self.max_tokens}"
            )
        if self.learning_rate is not None and not float(self.learning_rate) > 0:
            raise ValueError(
                "learning_rate must be positive, or None for the paper's "
                f"{FLASH_REINFORCE_LEARNING_RATE} on full weights (FLASH_REINFORCE_LEARNING_RATE); "
                f"got {self.learning_rate}"
            )

    def default_learning_rate(self, lora: bool) -> float:
        if self.learning_rate is not None:
            return float(self.learning_rate)
        return FLASH_REINFORCE_LEARNING_RATE_LORA if lora else FLASH_REINFORCE_LEARNING_RATE

    def update(self, batch: Sequence[Mapping[str, Any]]) -> Update:
        """The one-pass update for one fresh batch: Algorithm 1 of Hu et al. 2026.

        Each trajectory is a dict with ``reward``, ``logprobs`` (under the
        policy being trained), optional ``behavior_logprobs`` (under the
        sampler; on-policy when absent) and optional ``action_mask``. The
        result's ``coefficients[i][t]`` is ``m_i * A_i * rho_{i,t} / (T_i * B)``;
        a trainer's loss is ``-(coefficients * logprobs).sum()`` with the
        coefficients held constant. Refuses an empty batch, a row without a
        field, a non-finite reward, and a log-probability that is not a
        log-probability (non-finite or above 0) on an action token.
        """
        _trajectory_fields(batch, self.name)
        n = len(batch)
        rewards: list[float] = []
        for i, traj in enumerate(batch):
            reward = float(traj["reward"])
            if not math.isfinite(reward):
                raise ValueError(
                    f"{self.name}.update: trajectory {i} has reward {reward!r}; "
                    "a reward is a finite number"
                )
            rewards.append(reward)
        baseline = sum(rewards) / n
        advantages = [r - baseline for r in rewards]

        coefficients: list[list[float]] = []
        advantage_rows: list[list[float]] = []
        admitted: list[bool] = []
        drifts: list[float] = []
        ratios: list[float] = []
        rejected: list[int] = []
        on_policy = 0
        clamped = 0
        for i, traj in enumerate(batch):
            logprobs = [float(x) for x in traj["logprobs"]]
            raw_behavior = traj.get("behavior_logprobs")
            if raw_behavior is None:
                on_policy += 1
                behavior = logprobs
            else:
                behavior = [float(x) for x in raw_behavior]
            raw_mask = traj.get("action_mask")
            mask = [True] * len(logprobs) if raw_mask is None else [bool(x) for x in raw_mask]
            length = sum(mask)
            if length == 0:
                raise ValueError(
                    f"{self.name}.update: trajectory {i} has no action token (action_mask is "
                    "False everywhere); a trajectory needs at least one token the policy wrote"
                )
            drift = 0.0
            for t, on in enumerate(mask):
                if not on:
                    continue
                lp, blp = logprobs[t], behavior[t]
                if not (math.isfinite(lp) and math.isfinite(blp)) or lp > 0 or blp > 0:
                    raise ValueError(
                        f"{self.name}.update: trajectory {i} token {t} has logprob {lp!r} and "
                        f"behavior_logprob {blp!r}; a log-probability is finite and at most 0 "
                        "(pass log p, not p)"
                    )
                drift += _bernoulli_kl(math.exp(blp), math.exp(lp))
            drift /= length
            keep = drift <= self.trust
            row = [0.0] * len(logprobs)
            if keep:
                scale = advantages[i] / (length * n)
                for t, on in enumerate(mask):
                    if not on:
                        continue
                    log_ratio = logprobs[t] - behavior[t]
                    if abs(log_ratio) > FLASH_REINFORCE_LOG_RATIO_CLAMP:
                        clamped += 1
                        log_ratio = math.copysign(FLASH_REINFORCE_LOG_RATIO_CLAMP, log_ratio)
                    rho = math.exp(log_ratio)
                    ratios.append(rho)
                    row[t] = scale * rho
            else:
                rejected.append(i)
            coefficients.append(row)
            advantage_rows.append([advantages[i]] * len(logprobs))
            admitted.append(keep)
            drifts.append(drift)

        notes: list[str] = []
        if rejected:
            worst = max(drifts[i] for i in rejected)
            noun = "trajectory" if len(rejected) == 1 else "trajectories"
            notes.append(
                f"{len(rejected)} {noun} ({_index_list(rejected)}) over trust {self.trust:g} "
                f"masked whole (max mean KL {worst:.3g}); the sampler drifted further than the "
                "gate allows: check behavior_logprobs are the sampler's own, then lower "
                "off_policy_steps or raise trust"
            )
        if len(rejected) == n:
            notes.append("every trajectory is masked, so this update moves nothing")
        elif all(a == 0 for a in advantages):
            notes.append(
                f"every reward is {rewards[0]:g}, so every advantage is 0 and this update moves "
                "nothing; the batch mean is the only baseline, so a batch needs prompts the "
                "policy sometimes passes and sometimes fails: enlarge the batch or select "
                "prompts in the 20..80% band (wai.select)"
            )
        if on_policy:
            carries, taken = ("carries", "is") if on_policy == 1 else ("carry", "are")
            notes.append(
                f"{on_policy} trajector{'y' if on_policy == 1 else 'ies'} {carries} no "
                f"behavior_logprobs and {taken} taken as on-policy (ratio 1, drift 0); store "
                "the sampler's own log-probabilities to correct a stale rollout"
            )
        if clamped:
            notes.append(
                f"{clamped} admitted tokens hit the log-ratio clamp of "
                f"+-{FLASH_REINFORCE_LOG_RATIO_CLAMP:g} (FLASH_REINFORCE_LOG_RATIO_CLAMP)"
            )
        stats = {
            "batch_mean_reward": baseline,
            "admitted_share": (n - len(rejected)) / n,
            "mean_sequence_kl": sum(drifts) / n,
            "max_sequence_kl": max(drifts),
            "mean_ratio": sum(ratios) / len(ratios) if ratios else math.nan,
        }
        return Update(
            method=self.name,
            coefficients=coefficients,
            advantages=advantage_rows,
            admitted=admitted,
            value_targets=None,
            stats=stats,
            notes=notes,
        )

    def __str__(self) -> str:
        return (
            f"FlashReinforce(trust={self.trust:g}, off_policy_steps={self.off_policy_steps}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens})"
        )


@dataclass(frozen=True)
class SAO:
    """Single-rollout asynchronous optimization: a critic and a token band (Hou et al. 2026).

    One rollout per prompt, trained the moment it lands, however stale the
    policy that sampled it. There is no group to take a baseline over, so a
    value model gives every generated token its own advantage, and a token
    the trained policy has already moved too far from is dropped rather
    than clipped. Built for the shape a production trace arrives in: one
    trajectory, tool output interleaved, no re-run.

    The mechanism, per trajectory of ``T`` tokens of which ``L`` are the
    model's own (``action_mask`` True; tool output and observations are
    False and carry no gradient):

    * **Advantage**: skip-observation token-level GAE (arXiv:2607.07508,
      section 3.2, eq. 4 and 5). Over the action tokens in order, with the
      reward on the last one and the bootstrap jumping over any
      observation to the next action token,
      ``delta_k = r_k + gamma V(a_{k+1}) - V(a_k)`` and
      ``A_k = delta_k + gamma lambda A_{k+1}``, ``A_L = delta_L`` with
      ``V`` past the end taken as 0. ``gamma`` is ``SAO_GAMMA`` (1) and
      ``lambda = 1 - 1/(gae_alpha * L)``, the length-adaptive rule of VAPO
      (Yue et al. 2025, arXiv:2504.05118) at the paper's ``alpha = 1.5``
      (section 4.1), so the terminal reward reaches the first token with
      weight ``lambda ** (L - 1)``, about ``exp(-1/alpha)`` at any length.
      No advantage normalization: the critic is the baseline (eq. 1).
    * **Band**: direct double-sided importance sampling (section 3.1,
      eq. 1 to 3). ``r_t = exp(logprobs[t] - behavior_logprobs[t])``, the
      trained policy over the rollout engine's own log-probabilities with
      no ``pi_old`` in between; ``f(r_t) = r_t`` inside
      ``(ratio[0], ratio[1])`` = ``(1 - eps_low, 1 + eps_high)`` and 0
      outside, whichever sign the advantage has. A trajectory with no
      token left inside the band is not admitted.
    * **Coefficient**: ``coefficients[i][t] = f(r_t) * A_t / N`` where
      ``N`` is the number of action tokens in the batch, the token-level
      mean the paper's ``E_t`` (eq. 1) and VAPO's token-level loss take;
      the paper does not spell the aggregation out, and a per-trajectory
      ``1/T`` is the other reading. A trainer's loss is
      ``-(coefficient * logprob).sum()`` with the coefficient held
      constant, which is eq. 1 with ``f(r_t) A_t`` as the stop-gradient
      weight on ``log pi_theta(a_t | s_t)``.
    * **Critic**: ``value_targets[i][t]`` is the Monte Carlo return of the
      trajectory from that token, ``lambda_critic = 1`` (section 4.1), so
      with a terminal reward and ``gamma = 1`` every action token trains
      toward the reward; an observation token carries the return of the
      next action token, and the paper does not say whether the critic
      loss counts observation tokens (its step-level variant masks them,
      appendix A.1). The critic's loss is the squared error (section 2),
      it takes ``critic_steps`` updates per policy update (``K = 2``,
      section 3.2) with ``critic_warmup`` steps first (section 4.1, which
      does not say whether that is an optimizer warmup or critic-only
      steps), and in the paper its attention weights stay frozen and only
      the MoE projections train (section 3.2), a trainer detail with no
      knob here.

    ``ratio`` is the reasoning band by default, ``SAO_RATIO`` (0.7, 6.0);
    the coding run uses ``SAO_RATIO_CODING`` (0.2, 4.0). ``temperature``
    and ``max_tokens`` are the sampler's; ``learning_rate`` left ``None``
    is ``SAO_LEARNING_RATE`` (1e-6, full weights; the paper trains no
    adapter) and ``critic_learning_rate`` is ``SAO_CRITIC_LEARNING_RATE``
    (5e-6). Reference: Hou, Li, Tang and Dong 2026, "Single-Rollout
    Asynchronous Optimization for Agentic Reinforcement Learning",
    arXiv:2607.07508, sections 3.1, 3.2 and 4.1.

    Example, offline::

        import whileai as wai

        method = wai.SAO()
        update = method.update(
            [
                {"reward": 1.0, "logprobs": [-0.5, -1.0, -0.2], "values": [0.2, 0.5, 0.6]},
                {"reward": 0.0, "logprobs": [-0.7, -0.3], "values": [0.4, 0.3]},
            ]
        )
        print(update)  # "sao update: 2 of 2 trajectories admitted" and the stats
        update.coefficients[0][2]  # 0.4 / 5: (1 - 0.6) times ratio 1, over 5 tokens
    """

    ratio: tuple[float, float] = SAO_RATIO
    gae_alpha: float = SAO_GAE_ALPHA
    critic_steps: int = SAO_CRITIC_STEPS
    critic_warmup: int = SAO_CRITIC_WARMUP
    temperature: float = SAO_TEMPERATURE
    max_tokens: int = SAO_MAX_TOKENS
    learning_rate: float | None = None
    critic_learning_rate: float = SAO_CRITIC_LEARNING_RATE

    name: ClassVar[str] = "sao"
    samples: ClassVar[int] = 1

    def __post_init__(self) -> None:
        try:
            low, high = (float(x) for x in self.ratio)
        except (TypeError, ValueError):
            raise ValueError(
                f"ratio must be a (low, high) pair of floats (SAO_RATIO is {SAO_RATIO}); "
                f"got {self.ratio!r}"
            ) from None
        if not 0 < low < 1 < high:
            raise ValueError(
                "ratio must bracket 1, (low, high) with 0 < low < 1 < high: the band a token's "
                f"current/rollout probability ratio must stay inside (SAO_RATIO is {SAO_RATIO}, "
                f"SAO_RATIO_CODING is {SAO_RATIO_CODING}); got {self.ratio}"
            )
        object.__setattr__(self, "ratio", (low, high))
        if not float(self.gae_alpha) > 0:
            raise ValueError(
                "gae_alpha must be positive: lambda = 1 - 1/(gae_alpha * L) "
                f"(SAO_GAE_ALPHA is {SAO_GAE_ALPHA}); got {self.gae_alpha}"
            )
        if int(self.critic_steps) < 1:
            raise ValueError(
                "critic_steps must be at least 1, the value-network updates per policy update "
                f"(SAO_CRITIC_STEPS is {SAO_CRITIC_STEPS}); got {self.critic_steps}"
            )
        if int(self.critic_warmup) < 0:
            raise ValueError(
                "critic_warmup must be 0 or more, the critic's warmup steps "
                f"(SAO_CRITIC_WARMUP is {SAO_CRITIC_WARMUP}); got {self.critic_warmup}"
            )
        if not 0 < float(self.temperature) <= 2:  # noqa: PLR2004  # the sampler's range
            raise ValueError(
                f"temperature must be in (0, 2] (SAO_TEMPERATURE is {SAO_TEMPERATURE}); "
                f"got {self.temperature}"
            )
        if int(self.max_tokens) < 1:
            raise ValueError(
                f"max_tokens must be positive (SAO_MAX_TOKENS is {SAO_MAX_TOKENS}); "
                f"got {self.max_tokens}"
            )
        if self.learning_rate is not None and not float(self.learning_rate) > 0:
            raise ValueError(
                f"learning_rate must be positive or None (SAO_LEARNING_RATE is "
                f"{SAO_LEARNING_RATE}); got {self.learning_rate}"
            )
        if not float(self.critic_learning_rate) > 0:
            raise ValueError(
                "critic_learning_rate must be positive (SAO_CRITIC_LEARNING_RATE is "
                f"{SAO_CRITIC_LEARNING_RATE}); got {self.critic_learning_rate}"
            )

    def default_learning_rate(self, lora: bool) -> float:
        """The policy step: ``learning_rate`` if set, else ``SAO_LEARNING_RATE``.
        The paper trains full weights; no adapter rate is reported, so
        ``lora`` does not change it."""
        if self.learning_rate is not None:
            return float(self.learning_rate)
        return SAO_LEARNING_RATE

    def gae_lambda(self, action_tokens: int) -> float:
        """``1 - 1/(gae_alpha * L)`` for a trajectory of ``L`` action tokens,
        floored at 0 (VAPO, arXiv:2504.05118; alpha from arXiv:2607.07508)."""
        if action_tokens < 1:
            return 0.0
        return max(0.0, 1.0 - 1.0 / (float(self.gae_alpha) * action_tokens))

    def update(self, batch: Sequence[Mapping[str, Any]]) -> Update:
        """The SAO update for a batch of trajectories; see the class docstring
        for the formulas. Every trajectory needs ``values``."""
        _trajectory_fields(batch, self.name)
        low, high = self.ratio
        gamma = float(SAO_GAMMA)
        raw: list[list[float]] = []
        advantages: list[list[float]] = []
        admitted: list[bool] = []
        targets: list[list[float]] = []
        lambdas: list[float] = []
        n_action = n_masked = n_in_band = n_no_action = n_on_policy = 0
        ratio_sum = adv_sum = 0.0
        for i, traj in enumerate(batch):
            values = traj.get("values")
            if values is None:
                raise ValueError(
                    f"SAO.update: trajectory {i} has no 'values'. SAO's advantage is GAE over a "
                    "critic; pass values=[V(token), ...] from the value model, one per generated "
                    "token, or use wai.FlashReinforce, which takes a batch-mean baseline instead"
                )
            logprobs = [float(x) for x in traj["logprobs"]]
            behavior_in = traj.get("behavior_logprobs")
            if behavior_in is None:
                n_on_policy += 1
                behavior = logprobs
            else:
                behavior = [float(x) for x in behavior_in]
            v = [float(x) for x in values]
            n_tokens = len(logprobs)
            mask_in = traj.get("action_mask")
            mask = [True] * n_tokens if mask_in is None else [bool(m) for m in mask_in]
            reward = float(traj["reward"])
            action = [t for t in range(n_tokens) if mask[t]]
            n_act = len(action)
            adv = [0.0] * n_tokens
            ret = [0.0] * n_tokens
            coef = [0.0] * n_tokens
            if n_act == 0:
                n_no_action += 1
                ret = [reward] * n_tokens
                admitted.append(False)
            else:
                lam = self.gae_lambda(n_act)
                next_adv = next_v = next_ret = 0.0
                for k in range(n_act - 1, -1, -1):
                    t = action[k]
                    r_t = reward if k == n_act - 1 else 0.0
                    delta = r_t + gamma * next_v - v[t]
                    adv[t] = delta + gamma * lam * next_adv
                    ret[t] = r_t + gamma * next_ret
                    next_adv, next_v, next_ret = adv[t], v[t], ret[t]
                # an observation token sits before the next action: it carries that return
                carry = ret[action[-1]]
                for t in range(n_tokens - 1, -1, -1):
                    if mask[t]:
                        carry = ret[t]
                    else:
                        ret[t] = carry
                in_band = 0
                for t in action:
                    r = _ratio(logprobs[t] - behavior[t])
                    if low < r < high:
                        coef[t] = r * adv[t]
                        ratio_sum += r
                        in_band += 1
                    else:
                        n_masked += 1
                    adv_sum += adv[t]
                n_action += n_act
                n_in_band += in_band
                admitted.append(in_band > 0)
                lambdas.append(lam)
            raw.append(coef)
            advantages.append(adv)
            targets.append(ret)
        scale = 1.0 / n_action if n_action else 0.0
        coefficients = [[c * scale for c in coef] for coef in raw]
        n_admitted = sum(1 for a in admitted if a)
        stats = {
            "admitted_share": n_admitted / len(batch),
            "action_tokens": float(n_action),
            "masked_token_share": n_masked / n_action if n_action else 0.0,
            "mean_ratio": ratio_sum / n_in_band if n_in_band else 0.0,
            "mean_advantage": adv_sum / n_action if n_action else 0.0,
            "lambda_mean": sum(lambdas) / len(lambdas) if lambdas else 0.0,
        }
        notes: list[str] = []
        if n_on_policy == len(batch):
            notes.append("no behavior_logprobs on any trajectory: on-policy, every ratio is 1")
        if n_masked:
            notes.append(
                f"masked {n_masked} of {n_action} action tokens whose current/rollout ratio left "
                f"({low}, {high}); a rising share means the rollouts lag the policy, so shorten "
                "the lag or widen ratio="
            )
        if n_no_action:
            notes.append(
                f"{n_no_action} trajectories have no action token (action_mask all False): "
                "not admitted"
            )
        dropped = len(batch) - n_admitted - n_no_action
        if dropped:
            notes.append(f"{dropped} trajectories had every token outside the band: not admitted")
        return Update(
            method=self.name,
            coefficients=coefficients,
            advantages=advantages,
            admitted=admitted,
            value_targets=targets,
            stats=stats,
            notes=notes,
        )

    def __str__(self) -> str:
        return (
            f"SAO(ratio={self.ratio}, gae_alpha={self.gae_alpha}, critic_steps={self.critic_steps}, "
            f"critic_warmup={self.critic_warmup}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, critic_learning_rate={self.critic_learning_rate})"
        )


@dataclass(frozen=True)
class BPCO:
    """Best practice critic optimization: a bounded critic, one response (Qi et al. 2026).

    One rollout per prompt, and a critic in place of a group. The critic
    can only say a number inside the reward's own range, it is trained on
    the reward the rollout actually earned, the advantage it hands the
    policy is left at its natural scale, the credit a token gets from the
    final reward does not fade with the length of the response, and the
    policy step clips a token by how much its probability moved rather
    than by how much its ratio moved. Each of those is one ablation in the
    paper; together they let a critic-based method fit a small solvable
    set to nearly 100% where PPO collapses, and match or beat Dr. GRPO at
    16 samples per prompt with one (arXiv:2608.23566, figures 1 to 11).

    The mechanism, per token ``t`` of a response of ``L`` generated tokens
    with terminal reward ``R``:

    * the critic's raw head output ``z`` is bounded to ``reward_range``
      ``(R_min, R_max)`` by ``V = R_min + (R_max - R_min)(1/2 + atan(z)/pi)``
      (equation 9; ``bound()`` and ``unbound()`` are that map and its
      inverse). ``update`` reads the bounded ``values`` and refuses one
      outside the range;
    * the critic target is the Monte Carlo return, ``lambda_V = 1``,
      ``gamma = 1``, so with the reward only at the end it telescopes to
      ``R`` at every token (equation 11); ``value_targets`` is that. The
      paper's critic loss is the squared error to it (equation 6; the
      release keeps verl's clipped form with ``cliprange_value`` 0.5);
    * the policy advantage is GAE with ``delta_t = r_t + V(s_{t+1}) - V(s_t)``,
      ``r_t = 0`` before the last token and ``R`` at it, ``V`` past the end
      ``0``, and the length-adaptive ``lambda_pi(L) = 1 - 1/(gae_alpha * L)``
      (equations 3, 4 and 14), so the terminal residual's weight in the first
      token is ``lambda^L``, near ``exp(-1/gae_alpha)`` at any length. It is
      not normalized: no batch mean is subtracted and no standard deviation
      divides it (section 3.4). Where ``L < 1/gae_alpha`` the formula goes
      negative; the paper does not say, so ``lambda`` is clamped at 0 and a
      note says so;
    * the surrogate is DPPO, binary total variation (equation 2):
      ``min(rho A, clip(rho, 1 - clip/mu, 1 + clip/mu) A)`` with
      ``rho = pi(y_t|s_t) / mu(y_t|s_t)`` against the policy that sampled the
      token (the release reuses the rollout log-probabilities as the old
      ones) and ``mu`` that policy's probability of it. ``coefficients[i][t]``
      is the surrogate's gradient weight: ``rho A`` where the unclipped branch
      is the minimum, ``0`` where the clipped branch is (``A > 0`` and ``rho``
      above the range, or ``A < 0`` and ``rho`` below it). The loss sums a
      sequence's tokens and averages over sequences (verl's
      ``seq-mean-token-sum``, the release's ``AGG_MODE``), so no length
      weight enters the coefficient; the ``1/N`` is the batch mean;
    * a token the ``action_mask`` marks as the environment's carries no
      value, no residual and no coefficient; ``L`` counts the policy's own
      tokens. ``critic_warmup`` is the trainer's schedule: for that many
      updates it fits the critic on ``value_targets`` and skips the policy
      step (section 4; the release's ``trainer.critic_warmup``).

    ``stats``: ``clipped_token_share``, ``mean_ratio``, ``mean_advantage``,
    ``mean_clip_range`` (the mean half-width ``clip/mu``), ``mean_lambda``,
    ``admitted_share`` and, when the batch's rewards differ, the explained
    variance of the critic against the Monte Carlo target (equation 10).

    Reference: Qi, Zhou and Lee 2026, Best Practice Critic Optimization,
    arXiv:2608.23566 (DPPO: Qi et al. 2026, arXiv:2602.04879; LA-GAE:
    VAPO, arXiv:2504.05118, and SAO, arXiv:2607.07508; the Monte Carlo
    target: VC-PPO, arXiv:2503.01491). The numbers behind each default are
    in ``defaults.py`` under ``BPCO_*``.

    Example, offline::

        import whileai as wai

        m = wai.BPCO()
        v = [m.bound(z) for z in (-1.0, 0.0, 1.0)]        # the critic's bounded predictions
        update = m.update([
            {"reward": 1.0, "logprobs": [-0.5, -1.0, -0.2], "values": v},
            {"reward": 0.0, "logprobs": [-0.7, -0.3, -0.9], "values": v},
        ])
        print(update)          # bpco update: 2 of 2 trajectories admitted, then the stats
        update.coefficients    # rho * A per token; on-policy here, so rho = 1
        update.value_targets   # [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    """

    clip: float = BPCO_CLIP
    gae_alpha: float = BPCO_GAE_ALPHA
    reward_range: tuple[float, float] = BPCO_REWARD_RANGE
    critic_warmup: int = BPCO_CRITIC_WARMUP
    temperature: float = BPCO_TEMPERATURE
    max_tokens: int = BPCO_MAX_TOKENS
    learning_rate: float | None = None
    critic_learning_rate: float = BPCO_CRITIC_LEARNING_RATE

    name: ClassVar[str] = "bpco"
    samples: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if not 0 < float(self.clip) <= 1:
            raise ValueError(
                "clip is the DPPO threshold on a token's probability shift, |pi - mu| <= clip, "
                f"in (0, 1] (BPCO_CLIP is {BPCO_CLIP}); got {self.clip}"
            )
        if not float(self.gae_alpha) > 0:
            raise ValueError(
                "gae_alpha must be positive: lambda = 1 - 1/(gae_alpha * L) "
                f"(BPCO_GAE_ALPHA is {BPCO_GAE_ALPHA}); got {self.gae_alpha}"
            )
        try:
            lo, hi = (float(x) for x in self.reward_range)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"reward_range must be a pair (low, high) (BPCO_REWARD_RANGE is "
                f"{BPCO_REWARD_RANGE}); got {self.reward_range!r}"
            ) from e
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(
                "reward_range must be finite with low < high; the critic predicts inside it "
                f"(BPCO_REWARD_RANGE is {BPCO_REWARD_RANGE}); got {self.reward_range}"
            )
        object.__setattr__(self, "reward_range", (lo, hi))
        if int(self.critic_warmup) < 0 or int(self.critic_warmup) != self.critic_warmup:
            raise ValueError(
                "critic_warmup is a count of updates, 0 or more "
                f"(BPCO_CRITIC_WARMUP is {BPCO_CRITIC_WARMUP}); got {self.critic_warmup}"
            )
        if not 0 < float(self.temperature) <= 2:  # noqa: PLR2004  # the sampler's range
            raise ValueError(f"temperature must be in (0, 2]; got {self.temperature}")
        if int(self.max_tokens) < 1:
            raise ValueError(f"max_tokens must be positive; got {self.max_tokens}")
        if self.learning_rate is not None and not float(self.learning_rate) > 0:
            raise ValueError(
                f"learning_rate must be positive, or None for BPCO_LEARNING_RATE "
                f"({BPCO_LEARNING_RATE}); got {self.learning_rate}"
            )
        if not float(self.critic_learning_rate) > 0:
            raise ValueError(
                "critic_learning_rate must be positive (BPCO_CRITIC_LEARNING_RATE is "
                f"{BPCO_CRITIC_LEARNING_RATE}); got {self.critic_learning_rate}"
            )

    def default_learning_rate(self, lora: bool) -> float:
        """The policy step: ``learning_rate`` if set, else ``BPCO_LEARNING_RATE``
        (the paper trains full weights and gives no adapter rate)."""
        if self.learning_rate is not None:
            return float(self.learning_rate)
        return BPCO_LEARNING_RATE

    def bound(self, z: float) -> float:
        """The critic's raw head output mapped into ``reward_range``:
        ``R_min + (R_max - R_min)(1/2 + atan(z)/pi)`` (equation 9). Every
        finite ``z`` lands strictly inside the range; ``0`` at the midpoint."""
        lo, hi = self.reward_range
        return lo + (hi - lo) * (math.atan(float(z)) / math.pi + 1 / 2)

    def unbound(self, v: float) -> float:
        """The inverse of ``bound``: the head output that predicts ``v``.
        Refuses a value at or past an end of the range, which no finite output reaches."""
        lo, hi = self.reward_range
        v = float(v)
        if not lo < v < hi:
            raise ValueError(
                f"unbound({v}): bound() maps onto the open interval ({lo}, {hi}), so a value "
                "at or past an end has no finite head output"
            )
        return math.tan(math.pi * ((v - lo) / (hi - lo) - 1 / 2))

    def update(self, batch: Sequence[Mapping[str, Any]]) -> Update:
        """LA-GAE advantages, DPPO coefficients and Monte Carlo value targets
        for a batch of trajectories, per the class docstring.

        Each trajectory is a mapping with ``reward`` (terminal), ``logprobs``
        (one per generated token, under the policy being trained),
        ``values`` (the critic's bounded prediction at each token, required),
        and optionally ``behavior_logprobs`` (under the policy that sampled
        the token; absent means on-policy, ratio 1) and ``action_mask`` (1
        for the policy's tokens, 0 for the environment's).
        """
        _trajectory_fields(batch, self.name)
        lo, hi = self.reward_range
        coefficients: list[list[float]] = []
        advantages: list[list[float]] = []
        admitted: list[bool] = []
        targets: list[list[float]] = []
        n_action = n_clipped = 0
        sum_ratio = sum_adv = sum_range = sum_lambda = 0.0
        short: list[int] = []
        masked_out: list[int] = []
        pairs: list[tuple[float, float]] = []  # (reward, value) per action token, for EV
        for i, traj in enumerate(batch):
            lp = [float(x) for x in traj["logprobs"]]
            n = len(lp)
            raw_values = traj.get("values")
            if raw_values is None:
                raise ValueError(
                    f"{self.name}.update: trajectory {i} has no 'values'. BPCO is critic-based: "
                    "pass the critic's bounded prediction at every token, "
                    "values=[m.bound(z) for z in head_outputs]"
                )
            values = [float(v) for v in raw_values]
            for t, v in enumerate(values):
                if not lo <= v <= hi:
                    raise ValueError(
                        f"{self.name}.update: trajectory {i} value {v} at token {t} is outside "
                        f"reward_range {self.reward_range}; the critic predicts through bound(): "
                        "values=[m.bound(z) for z in head_outputs]"
                    )
            behavior = traj.get("behavior_logprobs")
            mu_lp = lp if behavior is None else [float(x) for x in behavior]
            raw_mask = traj.get("action_mask")
            mask = [1] * n if raw_mask is None else [1 if m else 0 for m in raw_mask]
            reward = float(traj["reward"])
            length = sum(mask)
            targets.append([reward] * n)
            if length == 0:
                masked_out.append(i)
                admitted.append(False)
                coefficients.append([0.0] * n)
                advantages.append([0.0] * n)
                continue
            lam = 1 - 1 / (float(self.gae_alpha) * length)
            if lam < 0:
                lam = 0.0
                short.append(i)
            last_action = max(t for t in range(n) if mask[t])
            adv = [0.0] * n
            next_value = 0.0
            last_gae = 0.0
            for t in range(n - 1, -1, -1):
                if not mask[t]:
                    continue
                r_t = reward if t == last_action else 0.0
                delta = r_t + next_value - values[t]
                last_gae = delta + lam * last_gae
                next_value = values[t]
                adv[t] = last_gae
            coef = [0.0] * n
            cap = float(BPCO_LOG_RATIO_CAP)
            for t in range(n):
                if not mask[t]:
                    continue
                log_ratio = min(cap, max(-cap, lp[t] - mu_lp[t]))
                ratio = math.exp(log_ratio)
                mu = math.exp(mu_lp[t])
                half = float(self.clip) / mu if mu > 0 else math.inf
                a = adv[t]
                if a > 0:
                    active = ratio <= 1 + half
                elif a < 0:
                    active = ratio >= 1 - half
                else:
                    active = True
                coef[t] = ratio * a if active else 0.0
                n_action += 1
                n_clipped += 0 if active else 1
                sum_ratio += ratio
                sum_adv += a
                sum_range += half
                pairs.append((reward, values[t]))
            sum_lambda += lam
            coefficients.append(coef)
            advantages.append(adv)
            admitted.append(True)
        n_admitted = sum(1 for a in admitted if a)
        stats: dict[str, float] = {"admitted_share": n_admitted / len(batch)}
        notes: list[str] = []
        if n_action:
            stats["clipped_token_share"] = n_clipped / n_action
            stats["mean_ratio"] = sum_ratio / n_action
            stats["mean_advantage"] = sum_adv / n_action
            stats["mean_clip_range"] = sum_range / n_action
            stats["mean_lambda"] = sum_lambda / n_admitted
        ev = _explained_variance(pairs)
        if ev is None:
            notes.append(
                "explained variance not computed: it needs rewards that differ across the batch "
                "(equation 10 divides by their variance); pass a batch with both outcomes"
            )
        else:
            stats["explained_variance"] = ev
        if masked_out:
            notes.append(
                f"{len(masked_out)} of {len(batch)} trajectories have every token masked "
                f"(action_mask all 0) and were not admitted: {masked_out[:MESSAGE_EXAMPLES]}"
            )
        if short:
            floor = 1 / float(self.gae_alpha)
            notes.append(
                f"{len(short)} trajectories are shorter than 1/gae_alpha = {floor:.3g} tokens, "
                "where lambda = 1 - 1/(gae_alpha * L) goes negative; clamped at 0 (one-step "
                "TD), which the paper leaves unspecified"
            )
        return Update(
            method=self.name,
            coefficients=coefficients,
            advantages=advantages,
            admitted=admitted,
            value_targets=targets,
            stats=stats,
            notes=notes,
        )

    def __str__(self) -> str:
        return (
            f"BPCO(clip={self.clip}, gae_alpha={self.gae_alpha}, reward_range={self.reward_range}, "
            f"critic_warmup={self.critic_warmup}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, lr={self.default_learning_rate(lora=False)}, "
            f"critic_lr={self.critic_learning_rate})"
        )


def _explained_variance(pairs: Sequence[tuple[float, float]]) -> float | None:
    """``1 - Var(R - V) / Var(R)`` over tokens (arXiv:2608.23566, equation 10);
    None when the rewards do not vary."""
    if not pairs:
        return None
    n = len(pairs)
    mean_r = sum(r for r, _ in pairs) / n
    var_r = sum((r - mean_r) ** 2 for r, _ in pairs) / n
    if var_r <= 0:
        return None
    errs = [r - v for r, v in pairs]
    mean_e = sum(errs) / n
    var_e = sum((e - mean_e) ** 2 for e in errs) / n
    return 1 - var_e / var_r


SingleRollout = FlashReinforce | SAO | BPCO
Method = OPD | OPSD | Async | SingleRollout | str


# --------------------------------------------------------------------------
# prime-rl
# --------------------------------------------------------------------------


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        return text if any(c in text for c in ".eE") else text + ".0"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_literal(v) for v in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} into TOML")


def _flatten(table: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for k, v in table.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, Mapping):
            out.extend(_flatten(v, key))
        else:
            out.append((key, v))
    return out


def _emit(
    table: Mapping[str, Any], prefix: str, out: list[str], comments: Mapping[str, str]
) -> None:
    scalars = {
        k: v for k, v in table.items() if not isinstance(v, Mapping) and not _is_table_array(v)
    }
    tables = {k: v for k, v in table.items() if isinstance(v, Mapping)}
    arrays = {k: v for k, v in table.items() if _is_table_array(v)}
    if scalars:
        if prefix:
            out.append("")
            out.append(f"[{prefix}]")
        for k, v in scalars.items():
            note = comments.get(f"{prefix}.{k}" if prefix else k)
            out.append(f"{k} = {_toml_literal(v)}" + (f"  # {note}" if note else ""))
    for k, v in tables.items():
        _emit(v, f"{prefix}.{k}" if prefix else k, out, comments)
    for k, items in arrays.items():
        for item in items:
            out.append("")
            out.append(f"[[{prefix}.{k}]]" if prefix else f"[[{k}]]")
            for key, v in _flatten(item):
                out.append(f"{key} = {_toml_literal(v)}")


def _is_table_array(v: Any) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(x, Mapping) for x in v)


def _set_dotted(table: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = table
    for p in parts[:-1]:
        nxt = node.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            node[p] = nxt
        node = nxt
    node[parts[-1]] = value


def _source(name: str, taskset: str | None = None) -> dict[str, Any]:
    """One train or eval source. ``harness.id = "null"`` and ``runtime.type =
    "subprocess"`` are what every prime-rl example config and our own runs set
    for a taskset with no agent harness of its own. Which rows a source reads
    is the taskset's own config field (verifiers v1 gives each taskset its own,
    ``dataset_split`` on the bundled ones), so it is not written here; set it
    with a ``train_source.env.taskset.<field>`` / ``eval_source.env.taskset.<field>``
    override, or ``source.<key>`` for both."""
    return {
        "name": name,
        "env": {
            "taskset": {"id": taskset or name},
            "agent": {"harness": {"id": "null"}, "runtime": {"type": "subprocess"}},
        },
    }


#: what prime-rl (main, 2026-09-21) has where a single-rollout method needs
#: something else, with the file that says so. ``orchestrator.algo``
#: (``packages/prime-rl-configs/src/prime_rl/configs/algorithm.py``,
#: ``AlgoConfig``) is grpo, echo, max_rl, rae, hierarchical_grpo, opd,
#: opsd, sft, debug; ``trainer.loss`` (``configs/trainer.py``,
#: ``LossConfig``) is ipo, icepop, custom.
_PRIME_RL_NO_BATCH_MEAN = (
    "prime-rl's reward advantages are group relative (orchestrator.algo grpo and max_rl: reward "
    "minus the group mean, src/prime_rl/orchestrator/algo/grpo.py), identically zero over a "
    "group of one, or an EMA of the agent's past rewards (rae); a batch-mean baseline is not "
    "offered"
)
_PRIME_RL_NO_CRITIC = (
    "prime-rl only ever hosts the trainable policy (prime_rl/configs/algorithm.py, "
    "FrozenModelConfig): no value network, so no GAE advantage"
)
_PRIME_RL_TOKEN_LOSSES = (
    "its losses mask per token (trainer.loss ipo on the probability difference, icepop on the "
    "ratio band; src/prime_rl/trainer/rl/loss.py) and normalize by the global token count "
    "(rl_scale in src/prime_rl/trainer/rl/train.py)"
)
_RUN_IT_YOURSELF = (
    "apply method.update(batch) inside your own trainer loop (the coefficients multiply each "
    "token's log-probability gradient; the batch contract is above Update in whileai/methods.py)"
)


def _refuse_single_rollout(method: SingleRollout) -> NoReturn:
    """Say what prime-rl lacks for this method and what to do instead.

    Raised from ``prime_rl_config`` rather than writing the nearest config,
    because a TOML headed ``flash_reinforce`` that ran ``rae`` with a
    token-level mask would train a different method under this one's name.
    """
    name = type(method).__name__
    if isinstance(method, FlashReinforce):
        raise ValueError(
            f"prime_rl_config: wai.{name} cannot run on prime-rl as written. "
            f"{_PRIME_RL_NO_BATCH_MEAN}; {_PRIME_RL_TOKEN_LOSSES}, so there is no sequence trust "
            f"region for trust={method.trust} and no 1/T weight per trajectory. What you can do: "
            f"{_RUN_IT_YOURSELF}; or run the nearest prime-rl algorithm, 'rae' (REINFORCE against "
            "an EMA of past rewards, SPIRAL, arXiv:2506.24119; group_size 1 allowed), as "
            f"wai.prime_rl_config(env, wai.Async('rae', off_policy_steps={method.off_policy_steps}, "
            "correction='icepop'), model=..., **{'orchestrator.group_size': 1}), which is a "
            "different baseline and a token-level mask, named as such; or wait for the trainer."
        )
    if isinstance(method, SAO):
        lo, hi = method.ratio
        raise ValueError(
            f"prime_rl_config: wai.{name} needs a value critic, and {_PRIME_RL_NO_CRITIC}. The "
            f"half prime-rl has is the token band: its icepop loss masks a token whose "
            f"trainer/inference ratio leaves (ratio_low, ratio_high), which is SAO's direct "
            f"double-sided importance sampling, and wai.Async(correction='icepop', ratio=({lo}, "
            f"{hi})) writes it around a group-mean or EMA baseline, which is a different method. "
            f"What you can do: {_RUN_IT_YOURSELF}, with your critic's 'values' on each "
            "trajectory; or wait for the trainer."
        )
    raise ValueError(
        f"prime_rl_config: wai.{name} needs a value critic, and {_PRIME_RL_NO_CRITIC}; and "
        f"prime-rl has no clip at all: {_PRIME_RL_TOKEN_LOSSES}, never clipping the ratio, so "
        f"the DPPO range clip/mu (clip={method.clip}) has no home there. What you can do: "
        f"{_RUN_IT_YOURSELF}, with your critic's 'values' on each trajectory; or wait for the "
        "trainer."
    )


def _taskset_id(env: Any) -> tuple[str, list[str]]:
    """The taskset id prime-rl addresses, and any warning about where it came from."""
    warnings: list[str] = []
    if isinstance(env, Mapping) and env.get("path"):
        env = env["path"]
    text = str(env)
    path = Path(text)
    if path.is_dir() and (path / "pyproject.toml").exists():
        pyproject = (path / "pyproject.toml").read_text(encoding="utf-8")
        m = re.search(r'^name\s*=\s*"([^"]+)"', pyproject, re.M)
        name = m.group(1) if m else path.name
        warnings.append(
            f"{path} is a whileai export in the verifiers load_environment shape, which vf-eval "
            f"reads and prime-rl does not: verifiers v1 resolves a taskset id by importing the "
            f"module and reading its __all__ for a Taskset subclass, and this package declares "
            f"no __all__ and no Taskset. `pip install {path}` then `vf-eval {name}` works; "
            f"`uv run rl` on this config will not find the taskset. Point prime-rl at an "
            "installed v1 taskset id instead, or write the Taskset subclass yourself (#841)."
        )
        return name, warnings
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", text):
        raise ValueError(
            f"env: {text!r} is neither an installed taskset id nor an exported environment "
            "directory. Pass the id prime-rl installs the taskset under, or the directory "
            "wai.export_environment wrote."
        )
    return text, warnings


@dataclass
class PrimeRLConfig:
    """What ``prime_rl_config`` wrote, and what the trainer will and will not read."""

    path: str | None
    text: str
    config: dict[str, Any]
    method: str
    model: str
    taskset: str
    gpus: tuple[int, int]
    honored: dict[str, str] = field(default_factory=dict)
    ignored: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def command(self) -> str:
        target = self.path or "<config.toml>"
        return f"uv run rl @ {target}"

    def __str__(self) -> str:
        orch = self.config.get("orchestrator", {})
        lines = [
            f"prime-rl config: {self.path or '(not written)'}",
            f"  method {self.method}, model {self.model}, taskset {self.taskset}, "
            f"{sum(self.gpus)} GPUs ({self.gpus[0]} inference, {self.gpus[1]} trainer)",
            f"  {self.config.get('max_steps')} steps x {orch.get('batch_size')} prompts x "
            f"{orch.get('group_size')} rollouts, max_off_policy_steps {orch.get('max_off_policy_steps')}",
        ]
        if self.honored:
            lines.append("  reads: " + "; ".join(f"{k} -> {v}" for k, v in self.honored.items()))
        if self.ignored:
            lines.append("  ignores: " + "; ".join(f"{k}: {v}" for k, v in self.ignored.items()))
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        lines.append(f"  run: {self.command}")
        return "\n".join(lines)


def prime_rl_config(
    env: Any,
    method: Method = "grpo",
    *,
    model: str,
    out: str | Path | None = None,
    gpus: int = PRIME_RL_GPUS,
    steps: int = PRIME_RL_STEPS,
    batch: int = PRIME_RL_BATCH,
    lora: bool = True,
    **overrides: Any,
) -> PrimeRLConfig:
    """Write the TOML prime-rl runs a method with, and say what it will read.

    ``env`` is the taskset: the id an installed verifiers v1 taskset is
    registered under, or the directory ``wai.export_environment`` wrote
    (its package name is used, with a warning that the export is the
    ``load_environment`` shape prime-rl main does not address by id).
    ``method`` is ``"grpo"``, ``"max_rl"``, ``"rae"`` (reward minus an
    EMA of past rewards, the one prime-rl baseline that stands at
    ``group_size = 1``), a ``OPD``, an ``OPSD`` or an ``Async`` around one
    of those. A single-rollout method (``FlashReinforce``, ``SAO``,
    ``BPCO``) is refused with a ``ValueError`` that says what prime-rl
    lacks (a batch-mean baseline, a value model, a sequence trust region,
    a clip) and what to do instead (``method.update(batch)`` in your own
    trainer loop, or ``"rae"``), because a TOML labelled with the method's
    name that trained something else would be worse than none. ``model`` is
    the policy to train; ``gpus`` are split half to the inference engine
    and half to the trainer (``PRIME_RL_GPUS``, 2, is the floor: prime-rl
    runs them on separate devices); ``steps`` and ``batch`` are optimizer
    steps and prompts per step; ``lora`` writes a rank-16 adapter
    (``TRAINING_LORA_RANK``) or trains full weights. Any other prime-rl key
    is an override in dotted form, ``prime_rl_config(..., **{"trainer.optim.lr": 2e-5})``,
    and lands verbatim; ``source.<key>`` lands on the train and the eval source,
    ``train_source.<key>`` and ``eval_source.<key>`` on one of them (which rows a
    taskset reads is its own field, ``dataset_split`` on the bundled ones).

    The result prints what was written, which of the method's knobs the
    trainer reads and where, which it ignores and why (prime-rl's ``opd``
    is reverse KL over the full vocabulary, its ``opsd`` scores against
    the live policy, its corrections are ``ipo`` and ``icepop``), and the
    launch line. A knob prime-rl cannot honor at all (``tis``) is refused
    here rather than silently dropped there.

    ```python
    cfg = wai.prime_rl_config("refunds-v1", wai.OPSD(), model="Qwen/Qwen3-8B", out="opsd.toml")
    print(cfg)
    ```
    """
    if not model:
        raise ValueError("model: the Hugging Face id of the policy to train, e.g. 'Qwen/Qwen3-4B'")
    if int(gpus) < PRIME_RL_GPUS:
        raise ValueError(
            f"gpus must be at least {PRIME_RL_GPUS}: prime-rl runs the inference engine and the "
            f"trainer on separate devices; got {gpus}"
        )
    if int(steps) < 1 or int(batch) < 1:
        raise ValueError(f"steps and batch must be positive; got steps={steps}, batch={batch}")

    taskset, warnings = _taskset_id(env)
    honored: dict[str, str] = {}
    ignored: dict[str, str] = {}
    comments: dict[str, str] = {}

    outer = method
    inner: OPD | OPSD | SingleRollout | str = method.method if isinstance(method, Async) else method
    if isinstance(inner, (FlashReinforce, SAO, BPCO)):
        _refuse_single_rollout(inner)
    if isinstance(inner, str):
        inner = _check_choice("method", inner, PRIME_RL_ALGORITHMS)
    elif not isinstance(inner, (OPD, OPSD)):
        raise TypeError(
            f"method must be one of {', '.join(PRIME_RL_ALGORITHMS)}, wai.OPD, wai.OPSD or "
            f"wai.Async; got {type(inner).__name__}"
        )

    infer = int(gpus) // 2
    train_gpus = int(gpus) - infer
    lr: float
    if isinstance(inner, str):
        samples = RL_ROLLOUTS_PER_PROMPT
        temperature = OPD_TEMPERATURE
        max_tokens = OPSD_MAX_TOKENS
        lr = PRIME_RL_LEARNING_RATE_LORA if lora else PRIME_RL_LEARNING_RATE_FULL
        algo: dict[str, Any] = {"type": inner}
        method_name = inner
        if inner == "rae":
            warnings.append(
                "rae's baseline is an EMA of the agent's own past rewards (orchestrator.algo.decay, "
                "prime-rl's 0.95, about twenty traces), not a group mean or a batch mean; on a "
                "single-agent taskset it is REINFORCE with that baseline, and group_size 1 is "
                "allowed (prime-rl docs/algorithms.md, RAEAlgoConfig). Override "
                "orchestrator.group_size to run it one rollout per prompt."
            )
    else:
        samples = int(inner.samples)
        temperature = float(inner.temperature)
        max_tokens = int(inner.max_tokens)
        lr = inner.default_learning_rate(lora)
        method_name = inner.name
        if isinstance(inner, OPD):
            teacher_model, teacher_url = inner.teacher_ref
            algo = {"type": "opd", "teacher": {"name": teacher_model, "base_url": teacher_url}}
            honored["teacher"] = "orchestrator.algo.teacher (name, base_url; key from VLLM_API_KEY)"
            if inner.divergence != "reverse_kl":
                ignored[f"divergence={inner.divergence}"] = "prime-rl opd is the reverse KL"
            ignored[f"top_k={inner.top_k}"] = (
                "prime-rl scores the teacher's full-vocabulary prefill; the top-k support is not a knob there"
            )
        else:
            algo = {"type": "opsd", "demo_key": inner.privileged, "template": inner.template}
            honored["privileged"] = (
                f"orchestrator.algo.demo_key = {inner.privileged!r} (read from the task's info, "
                "then its top-level fields)"
            )
            honored["template"] = "orchestrator.algo.template"
            kind, _ = inner.anchor_parts()
            if kind != "live":
                ignored[f"anchor={inner.anchor}"] = (
                    "prime-rl opsd scores against the live policy; an EMA or initial-weights "
                    "teacher (SDFT, SDPO) is not offered there"
                )
            if inner.divergence != "reverse_kl":
                ignored[f"divergence={inner.divergence}"] = "prime-rl opsd is the reverse KL"
            warnings.append(
                "OPSD costs points on thinking models (Kaur et al. 2026, arXiv:2607.05184) and "
                "needs in-context learning strong enough to use the hint (about 7B up); run a "
                "GRPO arm on the same holdout before believing a number."
            )
        honored["samples"] = "orchestrator.group_size"
        honored["temperature"] = "orchestrator.train.sampling.temperature"
        honored["max_tokens"] = "orchestrator.train.sampling.max_completion_tokens"
        honored["learning_rate"] = f"trainer.optim.lr = {lr}"

    off_policy = ASYNC_OFF_POLICY_STEPS
    loss: dict[str, Any] | None = None
    if isinstance(outer, Async):
        off_policy = int(outer.off_policy_steps)
        honored["off_policy_steps"] = "orchestrator.max_off_policy_steps"
        if outer.correction == "ipo":
            loss = {"type": "ipo", "eps": float(outer.eps)}
            honored["correction"] = f"trainer.loss = ipo, eps {outer.eps}"
        elif outer.correction == "icepop":
            lo, hi = outer.ratio
            loss = {"type": "icepop", "ratio_low": lo, "ratio_high": hi}
            honored["correction"] = f"trainer.loss = icepop, ratio {lo} to {hi}"
        else:
            raise ValueError(
                "correction='tis' is verl's truncated importance sampling; prime-rl offers "
                "'ipo' (mask on probability difference) and 'icepop' (mask outside a ratio band). "
                "Pick one of those for prime-rl."
            )
        method_name = f"async {method_name}"
    else:
        comments["orchestrator.max_off_policy_steps"] = (
            "prime-rl default; the trainer runs one step ahead of the sampler by design. "
            "wai.Async(off_policy_steps=) moves it"
        )

    trainer: dict[str, Any] = {"optim": {"lr": lr}}
    config: dict[str, Any] = {
        "max_steps": int(steps),
        "seq_len": PRIME_RL_SEQ_LEN,
        "deployment": {
            "gpus_per_node": int(gpus),
            "num_infer_gpus": infer,
            "num_train_gpus": train_gpus,
        },
        "model": {"name": str(model)},
        "trainer": trainer,
        "orchestrator": {
            "batch_size": int(batch),
            "group_size": samples,
            "max_off_policy_steps": off_policy,
            "algo": algo,
            "train": {
                "sampling": {"max_completion_tokens": max_tokens, "temperature": temperature},
                "source": [_source(taskset)],
            },
            "eval": {
                "interval": max(1, int(steps) // 4),
                "num_examples": PRIME_RL_EVAL_EXAMPLES,
                "group_size": PRIME_RL_EVAL_GROUP,
                "source": [_source(f"{taskset}-eval", taskset)],
            },
        },
    }
    if lora:
        config["trainer"]["model"] = {
            "lora": {"rank": TRAINING_LORA_RANK, "alpha": TRAINING_LORA_ALPHA}
        }
    if loss is not None:
        config["trainer"]["loss"] = loss
    comments["seq_len"] = "prompt plus response cap; raise it for long tool output"
    for key, value in overrides.items():
        blocks = {
            "source.": ("train", "eval"),
            "train_source.": ("train",),
            "eval_source.": ("eval",),
        }
        prefix = next((p for p in blocks if key.startswith(p)), None)
        if prefix:
            orchestrator: dict[str, Any] = config["orchestrator"]
            for block in blocks[prefix]:
                sources: list[dict[str, Any]] = orchestrator[block]["source"]
                for src in sources:
                    _set_dotted(src, key[len(prefix) :], value)
            honored[key] = f"override, written on the {' and '.join(blocks[prefix])} source"
            continue
        _set_dotted(config, key, value)
        honored[key] = "override, written as given"

    lines = [
        f"# whileai {method_name}: written by wai.prime_rl_config; every default is named in",
        "# whileai/simulations/defaults.py with its source. Edit freely; prime-rl reads this file.",
    ]
    _emit(config, "", lines, comments)
    text = "\n".join(lines) + "\n"
    try:  # pragma: no cover - tomllib is 3.11+
        import tomllib

        tomllib.loads(text)
    except ImportError:
        pass

    path: str | None = None
    if out is not None:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
        path = str(target)
    return PrimeRLConfig(
        path=path,
        text=text,
        config=config,
        method=method_name,
        model=str(model),
        taskset=taskset,
        gpus=(infer, train_gpus),
        honored=honored,
        ignored=ignored,
        warnings=warnings,
    )


__all__ = [
    "ANCHORS",
    "BPCO",
    "CORRECTIONS",
    "DIVERGENCES",
    "OPD",
    "OPSD",
    "PRIME_RL_ALGORITHMS",
    "PRIVILEGED",
    "SAO",
    "Async",
    "FlashReinforce",
    "GroupwiseGrading",
    "GroupwiseStats",
    "Method",
    "PrimeRLConfig",
    "SingleRollout",
    "SpreadReport",
    "Update",
    "factors_from_ranking",
    "prime_rl_config",
    "redistribute",
    "spread",
]
