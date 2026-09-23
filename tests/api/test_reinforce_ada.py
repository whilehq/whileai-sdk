"""wai.methods.ReinforceAda: the rounds, the exit, the pick, the advantage, the TRL hook."""

from __future__ import annotations

import random

import pytest

import whileai as wai
from whileai.reinforce_ada import KEEP, MAX_ROUNDS, ROUND_SIZE, ReinforceAda, balanced_pick


def _coins(rates, seed=0):
    rng = random.Random(seed)
    lookup = dict(enumerate(rates))

    def draw(prompts, k):
        return [[1.0 if rng.random() < lookup[p] else 0.0 for _ in range(k)] for p in prompts]

    return draw


def test_reachable_one_dot_down():
    assert wai.methods.ReinforceAda is ReinforceAda
    assert "32 draws per prompt, 4 trained on" in str(ReinforceAda())


def test_bad_configuration_is_refused_on_construction():
    with pytest.raises(ValueError, match="keep must be at least 2"):
        ReinforceAda(keep=1)
    with pytest.raises(ValueError, match="round_size must be at least keep"):
        ReinforceAda(keep=8, round_size=4)
    with pytest.raises(ValueError, match="max_rounds"):
        ReinforceAda(max_rounds=0)
    with pytest.raises(ValueError, match="exit must be one of"):
        ReinforceAda(exit="greedy")


def test_balanced_pick_tops_up_from_the_side_with_spare():
    assert balanced_pick("ab", "xyz", 4) == (["a", "b"], ["x", "y"])
    assert balanced_pick("a", "wxyz", 4) == (["a"], ["w", "x", "y"])
    assert balanced_pick("abcd", "", 4) == (["a", "b", "c", "d"], [])


def test_retired_prompt_keeps_two_of_each_against_the_pool_rate():
    result = ReinforceAda()(lambda ps, k: [[1.0] * 3 + [0.0] * (k - 3) for _ in ps], ["x"])
    g = result.groups[0]
    assert result.rounds == 1 and g.drawn == ROUND_SIZE and g.retired
    assert g.rewards == [1.0, 1.0, 0.0, 0.0]
    # the kept four average 0.5; the pool's rate is 3/8, and that is the baseline
    assert g.advantages == [1 - 3 / 8, 1 - 3 / 8, -3 / 8, -3 / 8]


def test_prompt_that_never_splits_runs_every_round_at_zero_advantage():
    result = ReinforceAda()(lambda ps, k: [[0.0] * k for _ in ps], ["x"])
    g = result.groups[0]
    assert result.rounds == MAX_ROUNDS and g.drawn == ROUND_SIZE * MAX_ROUNDS
    assert g.advantages == [0.0] * KEEP and not g.retired and not g.has_gradient


def test_dict_samples_come_back_as_drawn():
    def draw(ps, k):
        return [[{"reward": float(i % 2), "text": f"{p}-{i}"} for i in range(k)] for p in ps]

    result = ReinforceAda()(draw, ["a", "b"])
    assert [s["text"] for s in result.samples[1]] == ["b-1", "b-3", "b-0", "b-2"]


def test_positive_exit_stops_at_one_right_answer():
    balanced = ReinforceAda()(_coins([0.97] * 40), list(range(40)))
    positive = ReinforceAda(exit="positive")(_coins([0.97] * 40), list(range(40)))
    assert positive.rounds == 1 and positive.stats["drawn_per_prompt"] == ROUND_SIZE
    assert balanced.stats["drawn_per_prompt"] > 2 * ROUND_SIZE


def test_hard_prompts_recover_the_signal_grpo_loses():
    """The paper's claim on coins: p = 0.1 splits a group of 4 a third of the time."""
    rates = [0.1] * 400
    result = ReinforceAda()(_coins(rates, seed=1), list(range(len(rates))))
    assert result.stats["grpo_no_gradient"] == pytest.approx(0.9**4 + 0.1**4, abs=0.1)
    assert result.stats["no_gradient"] < 0.1
    assert "no gradient:" in str(result)


def test_draw_must_answer_every_prompt():
    with pytest.raises(ValueError, match="sample lists for 2 prompts"):
        ReinforceAda()(lambda ps, k: [[1.0] * k], ["a", "b"])
    with pytest.raises(ValueError, match="fewer than keep"):
        ReinforceAda(max_rounds=1)(lambda ps, k: [[1.0] for _ in ps], ["a"])


def test_trainer_hook_rebuilds_a_keep_sized_batch():
    torch = pytest.importorskip("torch")

    class FakeGRPO:
        """The slice of trl 0.19 GRPOTrainer the hook touches."""

        num_generations = 4
        num_iterations = 1
        beta = 0.0

        def __init__(self, reward_funcs):
            self.reward_funcs = reward_funcs
            self.reward_weights = torch.ones(len(reward_funcs))
            self.accelerator = type("A", (), {"device": "cpu"})()
            self.processing_class = type("T", (), {"pad_token_id": 0})()
            self._metrics = {"train": __import__("collections").defaultdict(list)}

        def _generate_and_score_completions(self, inputs):
            n = len(inputs)
            completions = [x["answer"] for x in inputs]
            for f in self.reward_funcs:
                f(completions=completions, prompts=[x["prompt"] for x in inputs])
            return {
                "prompt_ids": torch.tensor([[0, 5, 6]] * n),
                "prompt_mask": torch.tensor([[0, 1, 1]] * n),
                "completion_ids": torch.tensor([[7, 8]] * n),
                "completion_mask": torch.tensor([[1, 1]] * n),
            }

    rng = random.Random(3)

    def reward(completions, prompts, **_):
        return [1.0 if rng.random() < 0.3 else 0.0 for _ in completions]

    trainer = ReinforceAda().trainer(FakeGRPO)(reward_funcs=[reward])
    inputs = [{"prompt": f"q{i}", "answer": "x"} for i in range(3) for _ in range(4)]
    out = trainer._generate_and_score_completions(inputs)
    assert out["prompt_ids"].shape == (12, 2) and out["completion_ids"].shape == (12, 2)
    assert out["advantages"].shape == (12,)
    assert trainer._metrics["train"]["ada/drawn_per_prompt"]
    assert trainer.reward_funcs[0].__name__ == "reward"
