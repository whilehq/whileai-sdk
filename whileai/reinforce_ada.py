"""Reinforce-Ada: keep sampling a prompt until its group can teach something.

    import whileai as wai

    ada = wai.methods.ReinforceAda()          # keep 4, rounds of 8, at most 32 draws
    result = ada(draw, prompts)               # draw(prompts, k) -> k rewards per prompt
    print(result)                             # rounds, draws per prompt, groups with no gradient
    result.advantages                         # one list of `keep` advantages per prompt

    # or swap it into TRL's GRPO generation step (trl==0.19.x):
    from trl import GRPOTrainer
    trainer = ada.trainer(GRPOTrainer)(model=..., reward_funcs=[...], args=cfg)

GRPO scores each rollout against its group's mean reward, so a group
whose rollouts all scored the same (all right or all wrong) has a zero
advantage everywhere and contributes no gradient. Xiong et al. 2025
(arXiv:2510.04996) call that undersampling: a prompt solved one time in
ten comes back all-wrong from four draws two times in three, and from
twenty draws almost never. Reinforce-Ada-Seq samples in rounds, retires a
prompt once its pool can make a balanced group, trains on the same
``keep`` rollouts per prompt GRPO would, and measures each against the
pass rate of everything drawn for that prompt.

The authors' defaults below are read from their verl implementation
(github.com/RLHFlow/Reinforce-Ada, ``verl/trainer/config/algorithm.py``
and ``scripts/run_reinforce_ada.sh``). Replicated in
``recipes/papers/reinforce-ada``: on GSM8K with Qwen2.5-1.5B the groups
with no gradient fell from 0.52-0.62 to 0.25-0.33 of prompts at 2.5 times
the GPU minutes, and pass@1 moved +0.047 [-0.047, +0.140] across two
training seeds per arm, flat. The prompts it could not rescue were the
ones the model always solves; it pays most where pass rates are low.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Rollouts per prompt the update trains on (the reference's ``n = 4``).
#: The same size as the GRPO group it replaces, so the backward pass does
#: not grow; only which rollouts it sees does.
KEEP = 4
#: Rollouts drawn per still-active prompt per round (``round_repeat = 8``).
ROUND_SIZE = 8
#: The round cap (``max_rounds = 4``): at most 32 draws for a prompt that
#: never splits. The paper's Table setting is N_max = 32 as well.
MAX_ROUNDS = 4
#: A rollout counts as right above this reward (``positive_threshold = 0.7``).
#: Any cut in (0, 1) is the same on a binary reward; 0.7 is theirs.
THRESHOLD = 0.7
#: ``balanced`` retires a prompt at keep // 2 right and the rest wrong, the
#: paper's headline variant; ``positive`` retires it at one right answer
#: (the paper's positive-focused exit), which spends nothing trying to find
#: a wrong answer to a prompt the model always solves.
EXITS = ("balanced", "positive")

Draw = Callable[[list[Any], int], Sequence[Sequence[Any]]]


def _reward(sample: Any) -> float:
    """A sample is a float reward or a mapping with ``reward``."""
    if isinstance(sample, Mapping):
        if "reward" not in sample:
            raise ValueError(f"a drawn sample is a reward or a dict with 'reward'; got {sample!r}")
        return float(sample["reward"])
    return float(sample)


def balanced_pick(pos: Sequence[Any], neg: Sequence[Any], keep: int) -> tuple[list, list]:
    """keep // 2 right and the rest wrong, topped up from whichever side has
    spare when the other runs short; first drawn, first kept (the
    reference's ``downsample_cache``)."""
    take_pos = min(keep // 2, len(pos))
    take_neg = min(keep - take_pos, len(neg))
    short = keep - take_pos - take_neg
    if short and len(pos) > take_pos:
        take_pos += min(len(pos) - take_pos, short)
    elif short and len(neg) > take_neg:
        take_neg += min(len(neg) - take_neg, short)
    return list(pos[:take_pos]), list(neg[:take_neg])


@dataclass
class AdaGroup:
    """One prompt after sampling: the ``keep`` samples the update trains on,
    their advantages, and the pool they were picked from."""

    samples: list[Any]
    rewards: list[float]
    advantages: list[float]
    pass_rate: float
    drawn: int
    retired: bool

    @property
    def has_gradient(self) -> bool:
        return any(a != 0 for a in self.advantages)


@dataclass
class AdaResult:
    """What one Reinforce-Ada sampling pass did to a batch of prompts."""

    groups: list[AdaGroup]
    rounds: int
    keep: int
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def advantages(self) -> list[list[float]]:
        return [g.advantages for g in self.groups]

    @property
    def samples(self) -> list[list[Any]]:
        return [g.samples for g in self.groups]

    def __str__(self) -> str:
        s = self.stats
        return "\n".join(
            [
                f"reinforce-ada: {len(self.groups)} prompts, {self.rounds} round(s), "
                f"{self.keep} kept per prompt",
                f"  drawn per prompt: {s['drawn_per_prompt']:.1f}",
                f"  retired early: {s['retired']:.2f} of prompts",
                f"  no gradient: {s['no_gradient']:.2f} of prompts "
                f"(GRPO at {self.keep} would expect {s['grpo_no_gradient']:.2f})",
                f"  pass rate over everything drawn: {s['pass_rate']:.2f}",
            ]
        )

    def _repr_html_(self) -> str:
        return "<pre>" + str(self).replace("<", "&lt;") + "</pre>"


@dataclass(frozen=True)
class ReinforceAda:
    """Reinforce-Ada-Seq (Xiong et al. 2025, arXiv:2510.04996): adaptive
    sampling for group-relative RL.

    For a batch of prompts: draw ``round_size`` rollouts for every prompt
    still active; retire a prompt once it has ``keep // 2`` right and
    ``keep - keep // 2`` wrong (``exit="balanced"``) or one right
    (``exit="positive"``); repeat up to ``max_rounds``. Then keep ``keep``
    rollouts per prompt, right and wrong in balance where the pool has
    both, and give rollout ``i`` of prompt ``x`` the advantage

        A_i = r_i - p_hat(x),    p_hat(x) = right(x) / drawn(x)

    measured over everything drawn for ``x``, not over the ``keep`` kept
    (those are balanced by construction, so their own mean says nothing
    about difficulty), and not divided by a standard deviation (the
    reference run's ``norm_adv_by_std_in_grpo=False``). A prompt that never
    splits keeps what it has at zero advantage, as it would under GRPO.

    Why hard prompts get more: the paper optimizes E_x[log p(x)] instead of
    E_x[p(x)], whose gradient weights prompt x by 1/p(x). Spending more
    draws on a low-p prompt is that weight paid in samples rather than in
    the gradient.

    ``__call__(draw, prompts)`` is the sampler in plain Python: ``draw``
    takes a list of prompts and ``k`` and returns ``k`` samples per prompt,
    each a float reward or a dict with ``reward``. ``trainer(GRPOTrainer)``
    returns a TRL subclass whose generation step is this sampler; the loss
    is TRL's own.
    """

    keep: int = KEEP
    round_size: int = ROUND_SIZE
    max_rounds: int = MAX_ROUNDS
    exit: str = "balanced"
    threshold: float = THRESHOLD

    name = "reinforce_ada"

    def __post_init__(self) -> None:
        if int(self.keep) < 2:  # noqa: PLR2004  # a group of one has no baseline to split
            raise ValueError(f"keep must be at least 2 (KEEP is {KEEP}); got {self.keep}")
        if int(self.round_size) < int(self.keep):
            raise ValueError(
                f"round_size must be at least keep ({self.keep}), so one round can fill a "
                f"group (ROUND_SIZE is {ROUND_SIZE}); got {self.round_size}"
            )
        if int(self.max_rounds) < 1:
            raise ValueError(
                f"max_rounds must be at least 1 (MAX_ROUNDS is {MAX_ROUNDS}); got {self.max_rounds}"
            )
        if self.exit not in EXITS:
            raise ValueError(f"exit must be one of {EXITS}; got {self.exit!r}")

    @property
    def max_draws(self) -> int:
        return self.round_size * self.max_rounds

    def _done(self, pos: int, neg: int) -> bool:
        if self.exit == "positive":
            return pos >= 1
        return pos >= self.keep // 2 and neg >= self.keep - self.keep // 2

    def __call__(self, draw: Draw, prompts: Sequence[Any]) -> AdaResult:
        prompts = list(prompts)
        if not prompts:
            raise ValueError("ReinforceAda: no prompts to sample")
        pos: list[list[tuple[float, Any]]] = [[] for _ in prompts]
        neg: list[list[tuple[float, Any]]] = [[] for _ in prompts]
        active = list(range(len(prompts)))
        rounds = 0
        while active and rounds < self.max_rounds:
            rounds += 1
            drawn = draw([prompts[p] for p in active], self.round_size)
            if len(drawn) != len(active):
                raise ValueError(
                    f"draw returned {len(drawn)} sample lists for {len(active)} prompts"
                )
            for p, samples in zip(active, drawn):
                for sample in samples:
                    r = _reward(sample)
                    (pos if r > self.threshold else neg)[p].append((r, sample))
            active = [p for p in active if not self._done(len(pos[p]), len(neg[p]))]

        groups = []
        for p in range(len(prompts)):
            n = len(pos[p]) + len(neg[p])
            if n < self.keep:
                raise ValueError(
                    f"prompt {p} drew {n} samples, fewer than keep ({self.keep}); "
                    "draw must return round_size samples per prompt"
                )
            p_hat = len(pos[p]) / n
            kept_pos, kept_neg = balanced_pick(pos[p], neg[p], self.keep)
            kept = kept_pos + kept_neg
            groups.append(
                AdaGroup(
                    samples=[s for _, s in kept],
                    rewards=[r for r, _ in kept],
                    advantages=[r - p_hat for r, _ in kept],
                    pass_rate=p_hat,
                    drawn=n,
                    retired=self._done(len(pos[p]), len(neg[p])),
                )
            )
        # What plain GRPO at `keep` draws would have left flat, from each
        # prompt's measured pass rate: P(all right) + P(all wrong). A plug-in
        # estimate: early stopping biases each pool's rate toward 0.5, so
        # this reads low (0.60 against a true 0.66 at p = 0.1).
        grpo_flat = statistics.fmean(
            g.pass_rate**self.keep + (1 - g.pass_rate) ** self.keep for g in groups
        )
        stats = {
            "drawn_per_prompt": statistics.fmean(g.drawn for g in groups),
            "retired": statistics.fmean(g.retired for g in groups),
            "no_gradient": statistics.fmean(not g.has_gradient for g in groups),
            "grpo_no_gradient": grpo_flat,
            "pass_rate": statistics.fmean(g.pass_rate for g in groups),
        }
        return AdaResult(groups=groups, rounds=rounds, keep=self.keep, stats=stats)

    def trainer(self, base: type) -> type:
        """A subclass of TRL's ``GRPOTrainer`` (pass the class; TRL is not a
        dependency of this package) whose generation step is this sampler.

        Set ``num_generations`` to ``keep``, ``num_iterations=1`` and
        ``beta=0``: the batch is rebuilt after generation and carries no
        old or reference log-probabilities. Reward functions must be
        callables (their outputs are read back to sort right from wrong).
        Written against trl 0.19.x, whose ``_generate_and_score_completions``
        returns the padded prompt and completion ids this override
        re-pads; ``recipes/papers/reinforce-ada`` runs it end to end.
        """
        ada = self
        parent_cls: Any = base

        class ReinforceAdaTrainer(parent_cls):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                if self.num_generations != ada.keep:
                    raise ValueError(
                        f"num_generations ({self.num_generations}) must equal keep ({ada.keep})"
                    )
                if self.num_iterations != 1 or self.beta != 0.0:
                    raise ValueError(
                        "ReinforceAda rebuilds the batch after generation and carries no old "
                        "or reference log-probs: set num_iterations=1 and beta=0"
                    )
                funcs: list[Any] = list(getattr(self, "reward_funcs"))  # noqa: B009  # set by the parent
                self._ada_last: list[list[float]] = [[] for _ in funcs]
                wrapped = []
                for i, func in enumerate(funcs):
                    if not callable(func) or hasattr(func, "config"):
                        raise TypeError(
                            "ReinforceAda reads rewards back from callable reward functions; "
                            f"reward function {i} is a model"
                        )
                    wrapped.append(self._ada_capture(i, func))
                self.reward_funcs = wrapped

            def _ada_capture(self, i: int, func: Callable) -> Callable:
                def capture(*args: Any, **kwargs: Any) -> Any:
                    out = func(*args, **kwargs)
                    self._ada_last[i] = [float("nan") if r is None else float(r) for r in out]
                    return out

                capture.__name__ = getattr(func, "__name__", f"reward_{i}")
                return capture

            def _ada_rewards(self) -> list[float]:
                weights = [float(w) for w in self.reward_weights.tolist()]
                cols = self._ada_last
                return [
                    sum(w * c[j] for w, c in zip(weights, cols) if c[j] == c[j])
                    for j in range(len(cols[0]))
                ]

            def _generate_and_score_completions(self, inputs: list[dict]) -> dict:
                import torch

                parent = super()._generate_and_score_completions
                unique = inputs[:: self.num_generations]

                def draw(batch: list[dict], k: int) -> list[list[dict]]:
                    rows = [x for x in batch for _ in range(k)]
                    out = parent(rows)
                    rewards = self._ada_rewards()
                    per: list[list[dict]] = []
                    for n in range(len(batch)):
                        group = []
                        for j in range(n * k, (n + 1) * k):
                            keep_mask = out["prompt_mask"][j].bool()
                            group.append(
                                {
                                    "reward": rewards[j],
                                    "prompt_ids": out["prompt_ids"][j][keep_mask],
                                    "completion_ids": out["completion_ids"][j],
                                    "completion_mask": out["completion_mask"][j],
                                }
                            )
                        per.append(group)
                    return per

                result = ada(draw, unique)
                items = [s for g in result.samples for s in g]
                device = self.accelerator.device
                pad = self.processing_class.pad_token_id
                p_len = max(int(s["prompt_ids"].numel()) for s in items)
                c_len = max(int(s["completion_ids"].numel()) for s in items)
                size = len(items)
                prompt_ids = torch.full((size, p_len), pad, dtype=torch.long, device=device)
                prompt_mask = torch.zeros((size, p_len), dtype=torch.long, device=device)
                completion_ids = torch.full((size, c_len), pad, dtype=torch.long, device=device)
                completion_mask = torch.zeros((size, c_len), dtype=torch.long, device=device)
                for i, s in enumerate(items):
                    p, c, m = s["prompt_ids"], s["completion_ids"], s["completion_mask"]
                    prompt_ids[i, p_len - p.numel() :] = p  # prompts left-padded
                    prompt_mask[i, p_len - p.numel() :] = 1
                    completion_ids[i, : c.numel()] = c  # completions right-padded
                    completion_mask[i, : m.numel()] = m
                metrics = self._metrics["train"]
                for key, value in result.stats.items():
                    metrics[f"ada/{key}"].append(value)
                metrics["ada/rounds"].append(float(result.rounds))
                advantages = [a for g in result.advantages for a in g]
                return {
                    "prompt_ids": prompt_ids,
                    "prompt_mask": prompt_mask,
                    "completion_ids": completion_ids,
                    "completion_mask": completion_mask,
                    "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
                    "old_per_token_logps": None,
                    "ref_per_token_logps": None,
                }

        ReinforceAdaTrainer.__name__ = f"ReinforceAda{base.__name__}"
        return ReinforceAdaTrainer

    def __str__(self) -> str:
        return (
            f"ReinforceAda(keep={self.keep}, round_size={self.round_size}, "
            f"max_rounds={self.max_rounds}, exit={self.exit!r}): at most {self.max_draws} "
            f"draws per prompt, {self.keep} trained on"
        )
