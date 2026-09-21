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
  read what it kept and why.
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
from typing import Any, ClassVar

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
    FLASH_REINFORCE_LEARNING_RATE,
    FLASH_REINFORCE_MAX_TOKENS,
    FLASH_REINFORCE_OFF_POLICY_STEPS,
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
#: of ``method=`` and inside ``Async``
PRIME_RL_ALGORITHMS = ("grpo", "max_rl")

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

    ``method`` is what to train (``"grpo"``, ``"max_rl"``, an ``OPD``
    or an ``OPSD``). ``off_policy_steps`` is the bound: a rollout
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
    ``method`` is ``"grpo"``, ``"max_rl"``, a ``OPD``, an
    ``OPSD`` or an ``Async`` around one of those. ``model`` is
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
        raise NotImplementedError(
            f"prime_rl_config for wai.{type(inner).__name__}: the prime-rl mapping is written "
            "by the integration agent"
        )
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
