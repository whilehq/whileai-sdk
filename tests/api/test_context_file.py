"""wai.methods.ContextFile and KVLog: the harness, the cost, Eq. 6, the gate."""

from __future__ import annotations

import collections

import pytest

import whileai as wai
from whileai.context_file import Reply, as_file


def test_reachable_one_dot_down():
    assert wai.methods.ContextFile().gate == "paper"
    assert wai.methods.KVLog().chunks == 5


def test_bad_configuration_is_refused_on_construction():
    with pytest.raises(ValueError, match="gate must be one of"):
        wai.methods.ContextFile(gate="sometimes")
    with pytest.raises(ValueError, match="w_eff must be in"):
        wai.methods.ContextFile(w_eff=1.5)
    with pytest.raises(ValueError, match="keys must be"):
        wai.methods.KVLog(keys=99)


def test_kvlog_is_seeded_and_the_splits_never_share_a_log():
    env = wai.methods.KVLog()
    assert env.task(7) == env.task(7)
    t = env.task(7)
    lines = [ln for c in t["chunks"] for ln in c]
    asked = [ln for ln in lines if ln.startswith(f"set {t['ask']} ")]
    assert len(asked) >= 2 and asked[-1] == f"set {t['ask']} = {t['gold']}"
    assert any(ln.startswith(f"set {t['ask']} ") for ln in t["chunks"][0])
    train = {x["scenario_id"] for x in env.tasks(50)}
    hold = {x["scenario_id"] for x in env.tasks(50, split="holdout")}
    assert not train & hold


def test_complete_reads_the_whole_state_not_the_last_chunk():
    env = wai.methods.KVLog()
    t = env.task(3)
    table = "\n".join(f"{k}: {v}" for k, v in t["state"].items())
    assert env.complete(t, table)
    assert env.complete(t, "; ".join(f"set {k} = {v}" for k, v in t["state"].items()))
    assert not env.complete(t, "\n".join(t["chunks"][-1]))  # the seed-17 shortcut
    assert env.reward(t, f"\\boxed{{{t['gold']}}}") == 1.0 and env.reward(t, t["gold"]) == 0.0


def test_cost_counts_only_what_misses_the_cache():
    cost = wai.methods.ContextFile.cost
    assert cost([[1, 2, 3], [1, 2, 3, 9, 5, 6]], [[9], [7]]) == 3 + 1 + 2 + 1
    assert cost([[1, 2], [3, 4]], [[5], [6]]) == 6


def test_eq6_ranks_successes_by_cost_and_leaves_failures_alone():
    paper = wai.methods.ContextFile(gate="paper")
    assert paper.efficiency([1, 1, 0, 0], [300, 500, 100, 900]) == [0.25, -0.25, 0.0, 0.0]
    assert paper.efficiency([1, 0], [100, 50]) == [0.0, 0.0]  # one success: nothing to rank
    assert paper.efficiency([1, 1, 1], [10, 10, 1000])[2] == -1.0  # clipped
    assert wai.methods.ContextFile(gate="off").efficiency([1, 1], [1, 9]) == [0.0, 0.0]


def test_the_complete_gate_stops_paying_a_cheap_incomplete_success():
    gated = wai.methods.ContextFile(gate="complete")
    # The cheap success (cost 100) kept an incomplete file: under the paper
    # gate it is the best-ranked; under the complete gate it is not ranked.
    assert wai.methods.ContextFile(gate="paper").efficiency([1, 1, 1], [100, 300, 500])[0] > 0
    eff = gated.efficiency([1, 1, 1], [100, 300, 500], complete=[False, True, True])
    assert eff == [0.0, 0.25, -0.25]
    with pytest.raises(ValueError, match="needs complete="):
        gated.efficiency([1, 1], [1, 2])


def test_credit_puts_eq6_on_edits_only():
    clm = wai.methods.ContextFile(w_eff=0.5, gate="paper")
    adv = clm.credit([1, 1, 0, 0], [300, 500, 100, 900], edits=3)
    assert adv[0] == [0.625, 0.625, 0.625, 0.5] and adv[1] == [0.375, 0.375, 0.375, 0.5]
    assert adv[2] == [-0.5] * 4
    worst = wai.methods.ContextFile(w_eff=1.0, gate="paper").credit([1, 1, 0], [1, 1000, 1], 2)
    assert min(worst[1]) > max(worst[2])  # a dear success still outranks a failure


def _echo_generate(messages, max_tokens):
    """A stand-in model that folds every chunk into a key: value table."""
    out = []
    for m in messages:
        body = m[-1]["content"]
        file = body.split("```")[1].strip()
        state = dict(ln.split(": ") for ln in file.splitlines() if ": " in ln and file != "(empty)")
        if "Question" in body:
            key = body.split("final value of ")[1].split("?")[0]
            text = f"\\boxed{{{state.get(key, '0')}}}"
        else:
            for ln in body.split("Log chunk")[1].splitlines():
                if ln.startswith("set "):
                    k, v = ln[4:].split(" = ")
                    state[k] = v
            text = "```\n" + "\n".join(f"{k}: {v}" for k, v in state.items()) + "\n```"
        out.append(Reply(text, list(range(len(body) // 4)), list(range(len(text) // 4))))
    return out


def test_play_runs_episodes_and_the_report_reads_them():
    env, clm = wai.methods.KVLog(), wai.methods.ContextFile()
    eps = clm.play(_echo_generate, env.tasks(6), env)
    assert all(e.reward == 1.0 and e.complete for e in eps)
    assert all(len(e.files) == env.chunks and len(e.replies) == env.chunks + 1 for e in eps)
    rep = clm.report(eps)
    assert rep.pass_rate == 1.0 and rep.complete == 1.0 and rep.shortcut == 0.0
    assert "answered right: 1.00" in str(rep)
    rows = clm.rows(eps)
    assert len(rows) == 6 and {"cost_tokens", "complete"} <= set(rows[0])
    assert as_file("```\na: 1\n```") == "a: 1"


def test_trainer_hook_rebuilds_one_row_per_step():
    torch = pytest.importorskip("torch")

    class FakeGRPO:
        """The slice of trl 0.19 GRPOTrainer the hook touches."""

        num_generations = 4
        num_iterations = 1
        beta = 0.0
        max_completion_length = 8

        def __init__(self, reward_funcs):
            self.reward_funcs = reward_funcs
            self.generation_config = type("G", (), {"max_new_tokens": 8})()
            self.accelerator = type("A", (), {"device": "cpu"})()
            self.processing_class = type(
                "T",
                (),
                {"pad_token_id": 0, "batch_decode": lambda _, ids, **k: ["\\boxed{1}"] * len(ids)},
            )()
            self._metrics = {"train": collections.defaultdict(list)}

        def _generate_and_score_completions(self, inputs):
            n = len(inputs)
            return {
                "prompt_ids": torch.tensor([[0, 5, 6]] * n),
                "prompt_mask": torch.tensor([[0, 1, 1]] * n),
                "completion_ids": torch.tensor([[7, 8]] * n),
                "completion_mask": torch.tensor([[1, 1]] * n),
            }

    env = wai.methods.KVLog(chunks=2, updates=2, keys=2)
    tasks = env.tasks(2)
    trainer = wai.methods.ContextFile().trainer(FakeGRPO)(reward_funcs=[], env=env)
    inputs = [t for t in tasks for _ in range(4)]
    out = trainer._generate_and_score_completions(inputs)
    assert out["advantages"].shape == (8 * 3,)  # 8 trajectories x (2 edits + 1 answer)
    assert out["prompt_ids"].shape == (24, 2)
    assert trainer._metrics["train"]["context/cost"]


def test_complete_survives_a_dataset_round_trip():
    """datasets unifies dict schemas: a task's `state` comes back with every
    other task's keys as None. complete() reads the log, not that dict."""
    env = wai.methods.KVLog()
    t = env.task(5)
    table = "\n".join(f"{k}: {v}" for k, v in t["state"].items())
    padded = {**t, "state": {**{n: None for n in env.NAMES}, **t["state"]}}
    assert env.complete(padded, table)
    assert env.final_state(t) == t["state"]
