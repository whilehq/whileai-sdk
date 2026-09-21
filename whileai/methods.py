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
* ``prime_rl_config``, the TOML prime-rl reads, from a method object
  and a taskset.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, NoReturn

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
    BPCO_MAX_TOKENS,
    BPCO_REWARD_RANGE,
    BPCO_TEMPERATURE,
    FLASH_REINFORCE_LEARNING_RATE,
    FLASH_REINFORCE_MAX_TOKENS,
    FLASH_REINFORCE_OFF_POLICY_STEPS,
    FLASH_REINFORCE_TEMPERATURE,
    FLASH_REINFORCE_TRUST,
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
    SAO_GAE_ALPHA,
    SAO_LEARNING_RATE,
    SAO_MAX_TOKENS,
    SAO_RATIO,
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


@dataclass(frozen=True)
class FlashReinforce:
    """Critic-free single-rollout REINFORCE with a batch-mean baseline (Hu et al. 2026).

    One rollout per prompt. The advantage is the reward minus the batch
    mean; each token is corrected by the ratio of the trained policy to
    the policy that sampled it; a trajectory whose mean sampled-action KL
    to that sampler is over ``trust`` is masked whole (the sequence trust
    region); each admitted trajectory gets the same outer weight
    regardless of length (sample-mean optimization, ``1/T``).
    ``off_policy_steps`` is the lag the method is built to absorb.
    """

    trust: float = FLASH_REINFORCE_TRUST
    off_policy_steps: int = FLASH_REINFORCE_OFF_POLICY_STEPS
    temperature: float = FLASH_REINFORCE_TEMPERATURE
    max_tokens: int = FLASH_REINFORCE_MAX_TOKENS
    learning_rate: float | None = None

    name: ClassVar[str] = "flash_reinforce"
    samples: ClassVar[int] = 1

    def __post_init__(self) -> None:
        raise NotImplementedError("FlashReinforce: filled in by the flash-reinforce agent")

    def default_learning_rate(self, lora: bool) -> float:
        return FLASH_REINFORCE_LEARNING_RATE

    def update(self, batch: Sequence[Mapping[str, Any]]) -> Update:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class SAO:
    """Single-rollout asynchronous optimization: a critic and a token band (Hou et al. 2026).

    One rollout per prompt. A value network gives each token an advantage
    (length-adaptive GAE, ``lambda = 1 - 1/(gae_alpha * L)``, skipping
    tokens the environment wrote); a token whose current/rollout ratio
    leaves ``ratio`` is masked, not clipped (direct double-sided
    importance sampling); the critic takes ``critic_steps`` updates per
    policy update. arXiv:2607.07508.
    """

    ratio: tuple[float, float] = SAO_RATIO
    gae_alpha: float = SAO_GAE_ALPHA
    critic_steps: int = SAO_CRITIC_STEPS
    temperature: float = SAO_TEMPERATURE
    max_tokens: int = SAO_MAX_TOKENS
    learning_rate: float | None = None
    critic_learning_rate: float = SAO_CRITIC_LEARNING_RATE

    name: ClassVar[str] = "sao"
    samples: ClassVar[int] = 1

    def __post_init__(self) -> None:
        raise NotImplementedError("SAO: filled in by the sao agent")

    def default_learning_rate(self, lora: bool) -> float:
        return SAO_LEARNING_RATE

    def update(self, batch: Sequence[Mapping[str, Any]]) -> Update:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class BPCO:
    """Best practice critic optimization: a bounded critic, one response (Qi et al. 2026).

    One rollout per prompt. The critic predicts inside ``reward_range``
    through a scaled arctangent and trains toward the Monte Carlo return;
    the policy advantage is length-adaptive GAE (``lambda = 1 - 1/(gae_alpha
    * L)``), unnormalized; the surrogate is DPPO, a clip range of
    ``clip / mu`` that widens for a rare token; the critic trains alone for
    ``critic_warmup`` updates first. arXiv:2608.23566.
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
        raise NotImplementedError("BPCO: filled in by the bpco agent")

    def default_learning_rate(self, lora: bool) -> float:
        return BPCO_LEARNING_RATE

    def update(self, batch: Sequence[Mapping[str, Any]]) -> Update:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


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
            f"{path} is a whileai export in the verifiers load_environment shape; prime-rl main "
            f"addresses installed verifiers v1 tasksets by id, so `pip install {path}` and, if "
            "the trainer cannot find it, port the package to a v1 Taskset (tracked in issue 564)."
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
    "Method",
    "PrimeRLConfig",
    "SingleRollout",
    "Update",
    "prime_rl_config",
]
