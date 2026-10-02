"""Context files: train a model that keeps its own context as a file.

    import whileai as wai

    env = wai.methods.KVLog()                     # 5 chunks of 8 `set key = value` lines, 8 keys
    clm = wai.methods.ContextFile()               # w_eff 0.25, Eq. 6 on every success
    tasks = env.tasks(512, seed=0)
    episodes = clm.play(generate, tasks, env)     # generate(messages, max_tokens) -> replies
    print(clm.report(episodes))                     # pass rate, tokens a trajectory, file size

    # or swap it into TRL's GRPO generation step (trl==0.19.x):
    from trl import GRPOTrainer
    trainer = clm.trainer(GRPOTrainer)(model=..., reward_funcs=[...], args=cfg, env=env)

Context Language Models (arXiv:2609.37725) give the model its context as
a file it can rewrite, rather than a transcript that only grows. Here an
episode is ``env.edits(task)`` rewrites and one answer: at every edit the
model sees the file and the next input, then the input is gone and only
the file is left; at the last step it sees the file and the question.

Trained with stepwise GRPO, every step of a trajectory gets the outcome
advantage ``r_i - mean(r)``. The paper adds a success-gated efficiency
advantage on the edit steps (Eq. 6):

    A_eff_i = clip((c_bar - c_i) / c_bar, -1, 1)   for i among the group's successes
            = 0                                     otherwise, and for all when < 2 succeed

where ``c_i`` is the trajectory's prefix-reuse cost and ``c_bar`` the mean
over the successes. The step advantage is ``A_out + w_eff * A_eff`` on
edits, ``A_out`` on the answer.

``gate="paper"`` (the default) ranks every success. ``gate="complete"``
ranks only successes whose last file holds the task's whole state
(``env.complete``); it was built against a shortcut one early seed found
(copy only the latest chunk) and lost to the paper's rule over four seeds.
Replicated in ``recipes/papers/context-lm``: on ``KVLog`` the harness takes
Qwen2.5-1.5B from 0.07 to 0.95 pass@1 under plain stepwise GRPO, and the
paper's Eq. 6 adds +0.02 [+0.01, +0.03] at 15% fewer tokens, four seeds an
arm.
"""

from __future__ import annotations

import random
import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

#: Weight on Eq. 6, the paper's BrowseComp-Plus run
#: (``POLAR_DUAL_CHANNEL_W_EFF=0.25``; the code default is 1.0).
W_EFF = 0.25
#: Which successes Eq. 6 ranks. ``paper``: every success. ``complete``:
#: successes whose last file holds the task's whole state. ``off``: no
#: efficiency term, plain stepwise GRPO.
GATES = ("paper", "complete", "off")
#: Cap on one rewrite of the file and on the answer, in tokens.
FILE_TOKENS = 256
ANSWER_TOKENS = 48
#: Successes a group needs before Eq. 6 has something to rank against.
MIN_WINS = 2

SYSTEM = (
    "You keep notes in a file called context.md. Each turn you see the file and one "
    "new chunk of input; after the turn the chunk is gone and only context.md is left. "
    "At the end you will be asked a question about the input. Reply with the complete "
    "new contents of context.md and nothing else. Anything you leave out is lost."
)


class Env(Protocol):
    """What ``ContextFile`` needs from a task family."""

    def edits(self, task: dict) -> int: ...
    def messages(self, task: dict, step: int, file: str) -> list[dict]: ...
    def reward(self, task: dict, answer: str) -> float: ...
    def complete(self, task: dict, file: str) -> bool: ...


@dataclass
class Reply:
    """One generated step: the text, and the token ids the cost is counted
    from. ``row`` is whatever the caller needs back (a trainer keeps its
    padded tensors there)."""

    text: str
    prompt_ids: list[int]
    reply_ids: list[int]
    row: Any = None


Generate = Callable[[list[list[dict]], int], Sequence[Reply]]


@dataclass
class Episode:
    task: dict
    files: list[str] = field(default_factory=list)
    replies: list[Reply] = field(default_factory=list)
    answer: str = ""
    reward: float = 0.0
    cost: int = 0
    complete: bool = False


_FENCE = re.compile(r"^```[a-z]*\n?|\n?```$")


def as_file(reply: str) -> str:
    """A reply is the new file; one wrapped in a code fence means the inside."""
    return _FENCE.sub("", reply.strip()).strip()


def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = text.find("}", i)
    return text[i + 7 : j].replace(",", "").strip() if j > 0 else None


@dataclass(frozen=True)
class KVLog:
    """A seeded key-value log, ContextBench's KV Store cut down
    (arXiv:2609.37725, Appendix D): ``chunks`` chunks of ``updates``
    ``set key = value`` lines over ``keys`` keys, then one question, the
    final value of a key set at least twice. The answer is boxed; the
    reward is exact match, a program."""

    chunks: int = 5
    updates: int = 8
    keys: int = 8

    NAMES: ClassVar[tuple[str, ...]] = (
        "river", "stone", "maple", "cloud", "ember", "frost", "orbit", "pearl",
        "cedar", "delta", "flint", "grove", "haven", "ivory", "jade", "lumen",
    )  # fmt: skip

    def __post_init__(self) -> None:
        if not 1 <= self.keys <= len(self.NAMES):
            raise ValueError(f"keys must be 1 to {len(self.NAMES)}; got {self.keys}")
        if self.chunks < 2 or self.updates < 1:  # noqa: PLR2004  # the asked key is set in two chunks
            raise ValueError(
                f"need chunks >= 2 and updates >= 1; got {self.chunks}, {self.updates}"
            )

    def task(self, seed: int) -> dict:
        rng = random.Random(seed)
        names = rng.sample(self.NAMES, self.keys)
        ask = names[0]
        lines = [
            (rng.choice(names), rng.randrange(100, 1000)) for _ in range(self.chunks * self.updates)
        ]
        # The asked key: once in the first chunk, once in a later one, so
        # the first value seen is never the answer.
        lines[rng.randrange(self.updates)] = (ask, rng.randrange(100, 1000))
        lines[rng.randrange(self.updates, len(lines))] = (ask, rng.randrange(100, 1000))
        state: dict[str, int] = {}
        for k, v in lines:
            state[k] = v
        return {
            "scenario_id": f"kv-{seed}",
            "chunks": [
                [f"set {k} = {v}" for k, v in lines[i * self.updates : (i + 1) * self.updates]]
                for i in range(self.chunks)
            ],
            "ask": ask,
            "gold": str(state[ask]),
            "state": {k: str(v) for k, v in state.items()},
            "question": f"final value of {ask}?",
        }

    def tasks(self, n: int, seed: int = 0, *, split: str = "train") -> list[dict]:
        """``n`` tasks; ``split="holdout"`` draws from a disjoint seed range."""
        offsets = {"train": 0, "holdout": 1_000_000}
        if split not in offsets:
            raise ValueError(f"split is 'train' or 'holdout'; got {split!r}")
        base = offsets[split] + seed * 10_000
        out = [self.task(base + i) for i in range(n)]
        for t in out:
            t["scenario_id"] = f"{split}-{t['scenario_id']}"
        return out

    def edits(self, task: dict) -> int:
        return len(task["chunks"])

    def messages(self, task: dict, step: int, file: str) -> list[dict]:
        shown = file.strip() or "(empty)"
        n = self.edits(task)
        if step < n:
            body = (
                f"context.md:\n```\n{shown}\n```\n\nLog chunk {step + 1} of {n}:\n"
                + "\n".join(task["chunks"][step])
                + "\n\nWrite the new context.md."
            )
        else:
            body = (
                f"context.md:\n```\n{shown}\n```\n\nQuestion: what is the final value of "
                f"{task['ask']}? Answer with the number only, as \\boxed{{n}}."
            )
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]

    def reward(self, task: dict, answer: str) -> float:
        return 1.0 if boxed(answer) == task["gold"] else 0.0

    @staticmethod
    def final_state(task: dict) -> dict[str, str]:
        """Each key's last value, read from the log lines. Not from
        ``task["state"]``: a ``datasets.Dataset`` unifies dict schemas across
        rows and gives every task the keys it never set, as ``None``."""
        state: dict[str, str] = {}
        for chunk in task["chunks"]:
            for line in chunk:
                key, value = line.removeprefix("set ").split(" = ")
                state[key] = value
        return state

    def complete(self, task: dict, file: str) -> bool:
        """Every key's final value sits next to its name in the file."""
        return all(
            re.search(rf"\b{k}\b\W{{0,8}}{v}\b", file) is not None
            for k, v in self.final_state(task).items()
        )


@dataclass
class ContextReport:
    """What a batch of episodes did."""

    n: int
    pass_rate: float
    cost: float
    file_chars: float
    complete: float
    shortcut: float

    def __str__(self) -> str:
        return "\n".join(
            [
                f"context file: {self.n} episodes",
                f"  answered right: {self.pass_rate:.2f}",
                f"  tokens a trajectory (prefix-reuse): {self.cost:.0f}",
                f"  last file: {self.file_chars:.0f} chars, holds the whole state {self.complete:.2f}",
                f"  right with an incomplete file (shortcut): {self.shortcut:.2f}",
            ]
        )

    def _repr_html_(self) -> str:
        return "<pre>" + str(self).replace("<", "&lt;") + "</pre>"


@dataclass(frozen=True)
class ContextFile:
    """The context-as-file harness and the paper's credit
    (Context Language Models, arXiv:2609.37725, Section 4.2).

    ``w_eff`` weights Eq. 6 on the edit steps; ``gate`` picks which
    successes Eq. 6 ranks (``GATES``); ``file_tokens`` and
    ``answer_tokens`` cap each step. ``play`` runs episodes with any
    generator, ``credit`` gives every step its advantage, and
    ``trainer(GRPOTrainer)`` is a TRL subclass that does both."""

    w_eff: float = W_EFF
    gate: str = "paper"
    file_tokens: int = FILE_TOKENS
    answer_tokens: int = ANSWER_TOKENS

    name = "context_file"

    def __post_init__(self) -> None:
        if self.gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}; got {self.gate!r}")
        if not 0.0 <= float(self.w_eff) <= 1.0:
            # Above 1, the dearest success can rank under a failure.
            raise ValueError(f"w_eff must be in [0, 1] (W_EFF is {W_EFF}); got {self.w_eff}")
        if self.file_tokens < 1 or self.answer_tokens < 1:
            raise ValueError("file_tokens and answer_tokens must be positive")

    # -- the cost and the credit, plain Python ------------------------------

    @staticmethod
    def cost(prompts: Sequence[Sequence[int]], replies: Sequence[Sequence[int]]) -> int:
        """Prefix-reuse cost in tokens: each step pays for its prompt past
        the longest prefix it shares with the previous step's prompt plus
        reply (those sit in the KV cache), and for its reply. The paper's
        ``kv_cache_flops.py`` turns the same counts into FLOPs (2N per
        token plus an attention term); for one model the linear term is a
        constant factor."""
        total = 0
        prev: list[int] = []
        for p, r in zip(prompts, replies):
            total += len(p) - _common_prefix(p, prev) + len(r)
            prev = [*p, *r]
        return int(total)

    def efficiency(
        self,
        rewards: Sequence[float],
        costs: Sequence[float],
        complete: Sequence[bool] | None = None,
    ) -> list[float]:
        """Eq. 6 for one group. Under ``gate="complete"`` a success counts
        only when ``complete[i]``."""
        if self.gate == "off":
            return [0.0] * len(rewards)
        ok = [r >= 1.0 for r in rewards]
        if self.gate == "complete":
            if complete is None:
                raise ValueError("gate='complete' needs complete=, one bool per trajectory")
            ok = [a and bool(b) for a, b in zip(ok, complete)]
        wins = [c for o, c in zip(ok, costs) if o]
        if len(wins) < MIN_WINS or statistics.fmean(wins) <= 0:
            return [0.0] * len(rewards)
        bar = statistics.fmean(wins)
        return [max(-1.0, min(1.0, (bar - c) / bar)) if o else 0.0 for o, c in zip(ok, costs)]

    def credit(
        self,
        rewards: Sequence[float],
        costs: Sequence[float],
        edits: int,
        complete: Sequence[bool] | None = None,
    ) -> list[list[float]]:
        """Every step's advantage, ``[trajectory][step]``: ``edits`` edit
        steps carry ``A_out + w_eff * A_eff``, the answer step ``A_out``."""
        mean = statistics.fmean(rewards)
        eff = self.efficiency(rewards, costs, complete)
        return [[r - mean + self.w_eff * e] * edits + [r - mean] for r, e in zip(rewards, eff)]

    # -- the episode ----------------------------------------------------------

    def play(self, generate: Generate, tasks: Sequence[dict], env: Env) -> list[Episode]:
        """One episode per task. ``generate(messages_list, max_tokens)``
        returns one ``Reply`` per message list; every task must have the
        same number of edits (one batch per step)."""
        tasks = list(tasks)
        if not tasks:
            return []
        n = {env.edits(t) for t in tasks}
        if len(n) != 1:
            raise ValueError(f"every task in one call needs the same number of edits; got {n}")
        edits = n.pop()
        eps = [Episode(task=t) for t in tasks]
        for step in range(edits + 1):
            last = step == edits
            msgs = [env.messages(e.task, step, e.files[-1] if e.files else "") for e in eps]
            replies = list(generate(msgs, self.answer_tokens if last else self.file_tokens))
            if len(replies) != len(eps):
                raise ValueError(f"generate returned {len(replies)} replies for {len(eps)} prompts")
            for e, r in zip(eps, replies):
                e.replies.append(r)
                if last:
                    e.answer = r.text
                else:
                    e.files.append(as_file(r.text))
        for e in eps:
            e.reward = float(env.reward(e.task, e.answer))
            e.cost = self.cost([r.prompt_ids for r in e.replies], [r.reply_ids for r in e.replies])
            e.complete = bool(e.files) and bool(env.complete(e.task, e.files[-1]))
        return eps

    def report(self, episodes: Sequence[Episode]) -> ContextReport:
        eps = list(episodes)
        if not eps:
            raise ValueError("no episodes to report")
        return ContextReport(
            n=len(eps),
            pass_rate=statistics.fmean(e.reward for e in eps),
            cost=statistics.fmean(e.cost for e in eps),
            file_chars=statistics.fmean(len(e.files[-1]) if e.files else 0 for e in eps),
            complete=statistics.fmean(e.complete for e in eps),
            shortcut=statistics.fmean(e.reward >= 1.0 and not e.complete for e in eps),
        )

    def rows(self, episodes: Sequence[Episode]) -> list[dict]:
        """Eval rows, one per episode, grouped by task: what ``wai.pass_at``
        and ``wai.compare`` read, with the cost and the file riding along."""
        out: list[dict] = []
        seen: dict[str, int] = {}
        for e in episodes:
            sid = e.task["scenario_id"]
            out.append(
                {
                    "prompt": e.task.get("question", ""),
                    "final_text": e.answer,
                    "reward": e.reward,
                    "scenario_id": sid,
                    "rollout_index": seen.get(sid, 0),
                    "cost_tokens": e.cost,
                    "file_chars": len(e.files[-1]) if e.files else 0,
                    "complete": e.complete,
                }
            )
            seen[sid] = seen.get(sid, 0) + 1
        return out

    # -- TRL ------------------------------------------------------------------

    def trainer(self, base: type) -> type:
        """A subclass of TRL's ``GRPOTrainer`` (pass the class; TRL is not a
        dependency of this package) whose generation step plays the
        episode with TRL's own generation and gives each step its credit.

        Construct it with ``env=``. Set ``num_iterations=1`` and ``beta=0``:
        the batch is rebuilt after generation and carries no old or
        reference log-probabilities. Every dataset row needs the env's task
        fields; the reward functions TRL calls each step only log. Written
        against trl 0.19.x; ``recipes/papers/context-lm`` runs it end to end.
        """
        clm = self
        parent_cls: Any = base

        class ContextFileTrainer(parent_cls):
            def __init__(self, *args: Any, env: Env, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                if self.num_iterations != 1 or self.beta != 0.0:
                    raise ValueError(
                        "ContextFile rebuilds the batch after generation and carries no old "
                        "or reference log-probs: set num_iterations=1 and beta=0"
                    )
                self.clm_env = env

            def _clm_generate(self, messages: list[list[dict]], max_new: int) -> list[Reply]:
                rows = [{**self._clm_inputs[i], "prompt": m} for i, m in enumerate(messages)]
                # trl 0.19 generates with generation_config and clips with
                # max_completion_length: both carry the step's cap.
                gen = self.generation_config
                keep = (getattr(self, "max_completion_length"), gen.max_new_tokens)  # noqa: B009  # set by the parent
                setattr(self, "max_completion_length", max_new)  # noqa: B010
                gen.max_new_tokens = max_new
                try:
                    out = super()._generate_and_score_completions(rows)
                finally:
                    setattr(self, "max_completion_length", keep[0])  # noqa: B010
                    gen.max_new_tokens = keep[1]
                texts = self.processing_class.batch_decode(
                    out["completion_ids"], skip_special_tokens=True
                )
                replies = []
                for j, text in enumerate(texts):
                    p = out["prompt_ids"][j][out["prompt_mask"][j].bool()]
                    c, m = out["completion_ids"][j], out["completion_mask"][j]
                    replies.append(Reply(text, p.tolist(), c[m.bool()].tolist(), (p, c, m)))
                return replies

            def _generate_and_score_completions(self, inputs: list[dict]) -> dict:
                import torch

                self._clm_inputs = inputs
                eps = clm.play(self._clm_generate, inputs, self.clm_env)
                g = self.num_generations
                rows, advantages, effs = [], [], []
                for start in range(0, len(eps), g):
                    group = eps[start : start + g]
                    rewards = [e.reward for e in group]
                    costs = [float(e.cost) for e in group]
                    complete = [e.complete for e in group]
                    edits = len(group[0].files)
                    effs += clm.efficiency(rewards, costs, complete)
                    for e, steps in zip(group, clm.credit(rewards, costs, edits, complete)):
                        for reply, a in zip(e.replies, steps):
                            rows.append(reply.row)
                            advantages.append(a)
                rep = clm.report(eps)
                metrics = self._metrics["train"]
                metrics["context/reward"].append(rep.pass_rate)
                metrics["context/cost"].append(rep.cost)
                metrics["context/file_chars"].append(rep.file_chars)
                metrics["context/complete"].append(rep.complete)
                metrics["context/shortcut"].append(rep.shortcut)
                metrics["context/eff_abs"].append(statistics.fmean(abs(x) for x in effs))

                device = self.accelerator.device
                pad = self.processing_class.pad_token_id
                p_len = max(int(p.numel()) for p, _, _ in rows)
                c_len = max(int(c.numel()) for _, c, _ in rows)
                size = len(rows)
                prompt_ids = torch.full((size, p_len), pad, dtype=torch.long, device=device)
                prompt_mask = torch.zeros((size, p_len), dtype=torch.long, device=device)
                completion_ids = torch.full((size, c_len), pad, dtype=torch.long, device=device)
                completion_mask = torch.zeros((size, c_len), dtype=torch.long, device=device)
                for i, (p, c, m) in enumerate(rows):
                    prompt_ids[i, p_len - p.numel() :] = p  # prompts left-padded
                    prompt_mask[i, p_len - p.numel() :] = 1
                    completion_ids[i, : c.numel()] = c  # completions right-padded
                    completion_mask[i, : m.numel()] = m
                return {
                    "prompt_ids": prompt_ids,
                    "prompt_mask": prompt_mask,
                    "completion_ids": completion_ids,
                    "completion_mask": completion_mask,
                    "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
                    "old_per_token_logps": None,
                    "ref_per_token_logps": None,
                }

        ContextFileTrainer.__name__ = f"ContextFile{base.__name__}"
        return ContextFileTrainer

    def __str__(self) -> str:
        return (
            f"ContextFile(w_eff={self.w_eff}, gate={self.gate!r}): stepwise GRPO, "
            f"Eq. 6 on edits, {self.file_tokens} tokens a rewrite"
        )
