"""wai.FlashReinforce: the critic-free single-rollout update of Hu et al. 2026,
in plain Python, checked by hand and by a tabular policy that has to learn.

Nothing here touches a model, a GPU or the network. The hand-computed cases
pin the formulas (batch-centered advantage, per-token ratio, the sequence
trust gate on the mean Bernoulli KL proxy, the 1/T_i and 1/B factors); the
toy at the end is the proof that the coefficients, applied as a policy
gradient, raise the reward from chance to near one, on-policy and with a
sampler eight updates stale."""

from __future__ import annotations

import inspect
import math
import random
from collections import deque

import pytest

import whileai as wai
from whileai.simulations import defaults

FLASH_CONSTANTS = (
    "FLASH_REINFORCE_TRUST",
    "FLASH_REINFORCE_OFF_POLICY_STEPS",
    "FLASH_REINFORCE_BATCH",
    "FLASH_REINFORCE_LEARNING_RATE",
    "FLASH_REINFORCE_LEARNING_RATE_LORA",
    "FLASH_REINFORCE_TEMPERATURE",
    "FLASH_REINFORCE_MAX_TOKENS",
    "FLASH_REINFORCE_LOG_RATIO_CLAMP",
    "FLASH_REINFORCE_PROBABILITY_FLOOR",
)


def bernoulli_kl(p: float, q: float) -> float:
    """Eq. (6) of the paper, written out independently of the implementation."""
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


# --- the object ----------------------------------------------------------


def test_front_door_and_defaults_are_the_named_constants():
    assert wai.FlashReinforce is wai.methods.FlashReinforce
    m = wai.FlashReinforce()
    assert m.name == "flash_reinforce" and m.samples == 1
    assert m.trust == defaults.FLASH_REINFORCE_TRUST == 0.003
    assert m.off_policy_steps == defaults.FLASH_REINFORCE_OFF_POLICY_STEPS == 8
    assert m.temperature == defaults.FLASH_REINFORCE_TEMPERATURE == 1.0
    assert m.max_tokens == defaults.FLASH_REINFORCE_MAX_TOKENS
    assert m.batch == defaults.FLASH_REINFORCE_BATCH == 128
    assert m.learning_rate is None
    assert m.default_learning_rate(lora=False) == defaults.FLASH_REINFORCE_LEARNING_RATE == 1e-6
    assert m.default_learning_rate(lora=True) == defaults.FLASH_REINFORCE_LEARNING_RATE_LORA
    assert wai.FlashReinforce(learning_rate=3e-6).default_learning_rate(lora=True) == 3e-6
    assert (
        str(m)
        == "FlashReinforce(trust=0.003, off_policy_steps=8, temperature=1.0, max_tokens=8192)"
    )
    assert wai.FlashReinforce(trust=math.inf).trust == math.inf  # the no-trust ablation


def test_every_default_is_cited_in_defaults_source():
    src = inspect.getsource(defaults)
    for name in FLASH_CONSTANTS:
        assert f"# {name} = " in src or f"/ {name} = " in src, name
    block = src[src.index("# --- FlashReinforce") : src.index("# --- SAO")]
    assert "FlashREINFORCE.pdf" in block  # no arXiv id as of the day this was written
    assert "Hu" in block and "2026" in block
    for table in ("Table 9", "Table 13", "Sec. 3.2", "Appendix A"):
        assert table in block, table
    # the one number the paper does not give says so
    lora = block[block.index("# FLASH_REINFORCE_LEARNING_RATE_LORA") :]
    assert "(convention, untested" in lora.split("\nFLASH_REINFORCE_LEARNING_RATE_LORA = ")[0]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"trust": 0}, "FLASH_REINFORCE_TRUST"),
        ({"trust": -0.1}, "trust must be above 0"),
        ({"trust": math.nan}, "trust must be above 0"),
        ({"off_policy_steps": -1}, "FLASH_REINFORCE_OFF_POLICY_STEPS"),
        ({"temperature": 0}, "FLASH_REINFORCE_TEMPERATURE"),
        ({"temperature": 3}, "temperature must be in"),
        ({"max_tokens": 0}, "FLASH_REINFORCE_MAX_TOKENS"),
        ({"learning_rate": 0}, "FLASH_REINFORCE_LEARNING_RATE"),
        ({"learning_rate": -1e-6}, "learning_rate must be positive"),
    ],
)
def test_refusals_name_the_constant_and_the_fix(kwargs, match):
    with pytest.raises(ValueError, match=match):
        wai.FlashReinforce(**kwargs)


# --- the update, by hand -------------------------------------------------


def test_on_policy_batch_of_three_centers_rewards_and_normalizes_by_length():
    m = wai.FlashReinforce()
    batch = [
        {"reward": 1.0, "logprobs": [-0.5, -1.2, -0.3]},
        {"reward": 0.0, "logprobs": [-0.9, -0.4]},
        {"reward": 1.0, "logprobs": [-0.1] * 20},
    ]
    u = m.update(batch)
    assert u.method == "flash_reinforce" and u.n == 3 and u.n_admitted == 3
    assert u.admitted == [True, True, True] and u.value_targets is None
    expected = [1 / 3, -2 / 3, 1 / 3]
    for i, a in enumerate(expected):
        assert u.advantages[i] == pytest.approx([a] * len(batch[i]["logprobs"]))
    assert sum(row[0] for row in u.advantages) == pytest.approx(0.0)
    for i, a in enumerate(expected):  # every token: A_i / (T_i * B), ratio 1 on-policy
        t = len(batch[i]["logprobs"])
        assert u.coefficients[i] == pytest.approx([a / (t * 3)] * t)
        assert sum(u.coefficients[i]) == pytest.approx(a / 3)  # 1/B per trajectory, any length
    assert u.stats["batch_mean_reward"] == pytest.approx(2 / 3)
    assert u.stats["admitted_share"] == 1.0
    assert u.stats["mean_sequence_kl"] == 0.0 and u.stats["max_sequence_kl"] == 0.0
    assert u.stats["mean_ratio"] == pytest.approx(1.0)
    assert any("taken as on-policy" in n for n in u.notes)


def test_total_weight_is_length_independent():
    m = wai.FlashReinforce()
    u = m.update(
        [
            {"reward": 1.0, "logprobs": [-0.7, -0.2]},
            {"reward": 1.0, "logprobs": [-0.05] * 20},
            {"reward": 0.0, "logprobs": [-1.0] * 7},
        ]
    )
    assert sum(u.coefficients[0]) == pytest.approx(sum(u.coefficients[1]))
    assert sum(u.coefficients[0]) == pytest.approx((1 - 2 / 3) / 3)
    assert sum(u.coefficients[2]) == pytest.approx(-(2 / 3) / 3)


def test_stale_token_is_corrected_by_the_ratio_and_the_gate_reads_the_proxy():
    p, q = 0.2, 0.4  # sampler and learner probability of the sampled token
    d = bernoulli_kl(p, q)
    row = {"reward": 1.0, "logprobs": [math.log(q)] * 2, "behavior_logprobs": [math.log(p)] * 2}
    other = {"reward": 0.0, "logprobs": [-1.0]}
    open_gate = wai.FlashReinforce(trust=math.inf).update([row, other])
    assert open_gate.admitted == [True, True]
    assert open_gate.stats["max_sequence_kl"] == pytest.approx(d)
    assert open_gate.coefficients[0] == pytest.approx([0.5 * (q / p) / (2 * 2)] * 2)
    assert open_gate.stats["mean_ratio"] == pytest.approx((2 * (q / p) + 1) / 3)
    # the gate is inclusive at the threshold and the mean proxy is what it reads
    at_threshold = wai.FlashReinforce(trust=d).update([row, other])
    assert at_threshold.admitted == [True, True]
    below = wai.FlashReinforce(trust=d * 0.999).update([row, other])
    assert below.admitted == [False, True]


def test_drifted_trajectory_is_masked_whole_and_named():
    m = wai.FlashReinforce()
    batch = [
        {"reward": 1.0, "logprobs": [-1.0, -1.0], "behavior_logprobs": [-1.0, -1.0]},
        {"reward": 0.0, "logprobs": [-1.0, -1.0], "behavior_logprobs": [-4.0, -1.0]},
        {"reward": 1.0, "logprobs": [-0.5, -0.5, -0.5]},
    ]
    u = m.update(batch)
    assert u.admitted == [True, False, True]
    assert u.coefficients[1] == [0.0, 0.0]  # no token of a rejected trajectory trains
    assert u.advantages[1] == pytest.approx([-2 / 3, -2 / 3])  # the advantage is still reported
    assert u.coefficients[0] == pytest.approx([(1 / 3) / (2 * 3)] * 2)  # B stays 3
    assert u.stats["admitted_share"] == pytest.approx(2 / 3)
    assert u.stats["max_sequence_kl"] > m.trust
    note = next(n for n in u.notes if "masked whole" in n)
    assert note.startswith("1 trajectory (1) over trust 0.003 masked whole (max mean KL ")
    assert "off_policy_steps" in note and "trust" in note
    text = str(u)
    assert "2 of 3 trajectories admitted" in text and "masked whole" in text


def test_action_mask_excludes_tokens_from_length_and_from_drift():
    m = wai.FlashReinforce()
    batch = [
        {
            "reward": 1.0,
            "logprobs": [-0.5, float("nan"), -0.5],
            "behavior_logprobs": [-0.5, -9.0, -0.5],  # huge drift on the tool token
            "action_mask": [True, False, True],
        },
        {"reward": 0.0, "logprobs": [-1.0, -1.0, -1.0, -1.0]},
    ]
    u = m.update(batch)
    assert u.admitted == [True, True]  # the masked token is not in D_i
    assert u.stats["max_sequence_kl"] == 0.0
    assert u.coefficients[0] == pytest.approx([0.5 / (2 * 2), 0.0, 0.5 / (2 * 2)])  # T_0 is 2
    assert u.coefficients[1] == pytest.approx([-0.5 / (4 * 2)] * 4)


def test_all_equal_rewards_give_an_empty_update_and_say_what_to_do():
    u = wai.FlashReinforce().update(
        [{"reward": 1.0, "logprobs": [-1.0]}, {"reward": 1.0, "logprobs": [-2.0, -3.0]}]
    )
    assert u.admitted == [True, True]
    assert all(c == 0.0 for row in u.coefficients for c in row)
    assert all(a == 0.0 for row in u.advantages for a in row)
    note = next(n for n in u.notes if "every advantage is 0" in n)
    assert "every reward is 1" in note and "wai.select" in note


def test_log_ratio_is_clamped_not_clipped():
    m = wai.FlashReinforce(trust=math.inf)
    u = m.update(
        [
            {"reward": 1.0, "logprobs": [-0.1], "behavior_logprobs": [-40.1]},
            {"reward": 0.0, "logprobs": [-1.0]},
        ]
    )
    clamp = defaults.FLASH_REINFORCE_LOG_RATIO_CLAMP
    assert u.coefficients[0] == pytest.approx([0.5 * math.exp(clamp) / 2])
    assert any("log-ratio clamp" in n for n in u.notes)
    mild = m.update(
        [
            {"reward": 1.0, "logprobs": [-0.1], "behavior_logprobs": [-2.1]},
            {"reward": 0.0, "logprobs": [-1.0]},
        ]
    )
    assert mild.coefficients[0] == pytest.approx([0.5 * math.exp(2.0) / 2])  # no clipping
    assert not any("clamp" in n for n in mild.notes)


@pytest.mark.parametrize(
    ("batch", "match"),
    [
        ([], "batch is empty"),
        ([{"reward": 1.0}], "trajectory 0 needs 'logprobs'"),
        ([{"reward": 1.0, "logprobs": []}], "trajectory 0 needs 'logprobs'"),
        ([{"logprobs": [-1.0]}], "trajectory 0 has no 'reward'"),
        (
            [
                {"reward": 1.0, "logprobs": [-1.0]},
                {"reward": 0.0, "logprobs": [-1.0, -1.0], "behavior_logprobs": [-1.0]},
            ],
            "trajectory 1 has 1 behavior_logprobs for 2 logprobs",
        ),
        ([{"reward": 1.0, "logprobs": [-1.0], "action_mask": [False]}], "no action token"),
        ([{"reward": math.nan, "logprobs": [-1.0]}], "finite number"),
        ([{"reward": 1.0, "logprobs": [0.5]}], "token 0 has logprob 0.5"),
        ([{"reward": 1.0, "logprobs": [-math.inf]}], "log-probability is finite"),
    ],
)
def test_refuses_a_batch_it_cannot_read_and_names_the_field(batch, match):
    with pytest.raises(ValueError, match=match):
        wai.FlashReinforce().update(batch)


def test_print_contains_admitted():
    u = wai.FlashReinforce().update(
        [{"reward": 1.0, "logprobs": [-1.0]}, {"reward": 0.0, "logprobs": [-1.0]}]
    )
    assert "admitted" in str(u) and "admitted" in u._repr_html_()


# --- the proof that it learns --------------------------------------------
#
# A contextual sequence task: 4 prompts, each with a secret target of 3 tokens
# from a vocabulary of 4; the reward is 1 when the sampled sequence is the
# target, else 0. The policy is a table of logits per (prompt, position),
# softmax over the vocabulary, so the gradient of log pi(a | s) with respect
# to the logits of s is onehot(a) - pi(. | s). Chance is (1/4)^3 = 1/64.

PROMPTS, LENGTH, VOCAB, REPEATS = 4, 3, 4, 4  # batch = 16 rollouts, one per prompt copy
LR, STEPS, SEED = 5.0, 400, 0


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def train_toy(method: wai.FlashReinforce, lag: int = 0, absurd: bool = False) -> dict:
    """Run the toy; returns the reward curve and what the gate did."""
    rng = random.Random(SEED)
    targets = [[rng.randrange(VOCAB) for _ in range(LENGTH)] for _ in range(PROMPTS)]
    logits = [[[0.0] * VOCAB for _ in range(LENGTH)] for _ in range(PROMPTS)]
    snapshots: deque = deque(maxlen=lag + 1)
    curve: list[float] = []
    admitted_shares: list[float] = []
    ratios: list[float] = []
    for _ in range(STEPS):
        snapshots.append([[row[:] for row in p] for p in logits])
        sampler = snapshots[0]  # the sampling policy lags the trained one by `lag` updates
        batch, rows = [], []
        for prompt in range(PROMPTS):
            for _ in range(REPEATS):
                tokens, behavior = [], []
                for pos in range(LENGTH):
                    probs = softmax(sampler[prompt][pos])
                    a = rng.choices(range(VOCAB), weights=probs)[0]
                    tokens.append(a)
                    behavior.append(math.log(probs[a]))
                current = [
                    math.log(softmax(logits[prompt][pos])[a]) for pos, a in enumerate(tokens)
                ]
                traj = {"reward": float(tokens == targets[prompt]), "logprobs": current}
                if lag:
                    traj["behavior_logprobs"] = behavior
                if absurd:
                    traj["behavior_logprobs"] = [x - 5.0 for x in current]
                batch.append(traj)
                rows.append((prompt, tokens))
        update = method.update(batch)
        curve.append(update.stats["batch_mean_reward"])
        admitted_shares.append(update.stats["admitted_share"])
        ratios.append(update.stats["mean_ratio"])
        if absurd:
            return {"curve": curve, "update": update}
        for (prompt, tokens), coefs in zip(rows, update.coefficients):
            for pos, (a, c) in enumerate(zip(tokens, coefs)):
                if c == 0.0:
                    continue
                probs = softmax(logits[prompt][pos])
                for v in range(VOCAB):
                    logits[prompt][pos][v] += LR * c * ((1.0 if v == a else 0.0) - probs[v])
    return {"curve": curve, "admitted": admitted_shares, "ratios": ratios}


def test_toy_policy_learns_from_chance_on_policy():
    run = train_toy(wai.FlashReinforce())
    curve = run["curve"]
    assert pytest.approx(1 / 64) == (1 / VOCAB) ** LENGTH  # the uniform start
    assert sum(curve[:10]) / 10 < 0.2  # about chance at the start
    assert sum(curve[-20:]) / 20 > 0.8  # near one at the end
    assert all(share == 1.0 for share in run["admitted"])  # on-policy: nothing to gate
    assert all(r == pytest.approx(1.0) for r in run["ratios"])


def test_toy_policy_learns_with_a_sampler_eight_updates_stale():
    method = wai.FlashReinforce()  # the paper's trust, 0.003
    run = train_toy(method, lag=method.off_policy_steps)
    curve = run["curve"]
    assert sum(curve[:10]) / 10 < 0.2
    assert sum(curve[-20:]) / 20 > 0.8
    assert any(abs(r - 1.0) > 1e-6 for r in run["ratios"])  # the ratio did real work
    assert min(run["admitted"]) < 1.0  # and the gate bit at least once


def test_toy_absurd_behavior_logprobs_are_fully_masked():
    run = train_toy(wai.FlashReinforce(), absurd=True)
    update = run["update"]
    assert update.n_admitted == 0 and update.n == PROMPTS * REPEATS
    assert all(c == 0.0 for row in update.coefficients for c in row)
    assert update.stats["admitted_share"] == 0.0
    assert any("16 trajectories" in n and "masked whole" in n for n in update.notes)
    assert any("moves nothing" in n for n in update.notes)
