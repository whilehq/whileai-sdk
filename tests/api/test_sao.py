"""wai.SAO: single-rollout asynchronous optimization (Hou et al. 2026, arXiv:2607.07508).

The object refuses a bad knob naming the constant; ``update`` computes
skip-observation length-adaptive GAE over a critic, the direct
double-sided importance-sampling band, and the token-mean coefficient;
and a seeded tabular toy learns from it, on-policy and under a lagged
sampler. Nothing here touches a model or the network."""

from __future__ import annotations

import copy
import inspect
import math
import random

import pytest

import whileai as wai
from whileai.simulations import defaults

ALPHA = defaults.SAO_GAE_ALPHA


def _traj(reward, values, logprobs=None, behavior=None, mask=None):
    t = {"reward": reward, "values": list(values)}
    t["logprobs"] = list(logprobs) if logprobs is not None else [-1.0] * len(values)
    if behavior is not None:
        t["behavior_logprobs"] = list(behavior)
    if mask is not None:
        t["action_mask"] = list(mask)
    return t


# --- the object -----------------------------------------------------------


def test_front_door_and_defaults_are_the_named_constants():
    assert wai.SAO is wai.methods.SAO
    m = wai.SAO()
    assert m.name == "sao" and m.samples == 1
    assert m.ratio == defaults.SAO_RATIO == (0.7, 6.0)  # eps_low 0.3, eps_high 5.0
    assert defaults.SAO_RATIO_CODING == (0.2, 4.0)  # eps_low 0.8, eps_high 3.0
    assert m.gae_alpha == defaults.SAO_GAE_ALPHA == 1.5
    assert m.critic_steps == defaults.SAO_CRITIC_STEPS == 2
    assert m.critic_warmup == defaults.SAO_CRITIC_WARMUP == 10
    assert m.temperature == defaults.SAO_TEMPERATURE
    assert m.max_tokens == defaults.SAO_MAX_TOKENS == 128 * 1024
    assert m.critic_learning_rate == defaults.SAO_CRITIC_LEARNING_RATE == 5e-6
    assert m.default_learning_rate(lora=True) == defaults.SAO_LEARNING_RATE == 1e-6
    assert m.default_learning_rate(lora=False) == defaults.SAO_LEARNING_RATE
    assert wai.SAO(learning_rate=3e-6).default_learning_rate(lora=False) == 3e-6
    assert defaults.SAO_GAMMA == 1.0 and defaults.SAO_BATCH == 128
    text = str(m)
    assert text.count("\n") == 0
    assert "ratio=(0.7, 6.0)" in text and "critic_steps=2" in text and "gae_alpha=1.5" in text


def test_every_sao_default_cites_the_paper():
    src = inspect.getsource(defaults)
    start = src.index("# --- SAO, single-rollout asynchronous optimization")
    end = src.index("# --- BPCO")
    block = src[start:end]
    for name in (
        "SAO_RATIO",
        "SAO_RATIO_CODING",
        "SAO_GAE_ALPHA",
        "SAO_GAMMA",
        "SAO_CRITIC_STEPS",
        "SAO_CRITIC_WARMUP",
        "SAO_LEARNING_RATE",
        "SAO_CRITIC_LEARNING_RATE",
        "SAO_BATCH",
        "SAO_TEMPERATURE",
        "SAO_MAX_TOKENS",
    ):
        assert f"# {name} = " in block or f"/ {name} = " in block, name
        assert f"\n{name} = " in block, name
    assert block.count("2607.07508") >= 9
    # the two numbers the paper does not give say so in the exact words the checker reads
    for name in ("SAO_GAMMA", "SAO_TEMPERATURE"):
        comment = block[block.index(f"# {name} = ") : block.index(f"\n{name} = ")]
        assert "(convention, untested)" in comment, name


def test_refusals_name_the_constant_and_the_fix():
    with pytest.raises(ValueError, match=r"bracket 1.*SAO_RATIO"):
        wai.SAO(ratio=(1.2, 6.0))
    with pytest.raises(ValueError, match="bracket 1"):
        wai.SAO(ratio=(0.0, 6.0))
    with pytest.raises(ValueError, match="bracket 1"):
        wai.SAO(ratio=(0.7, 1.0))
    with pytest.raises(ValueError, match="SAO_RATIO"):
        wai.SAO(ratio=(0.7,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="SAO_GAE_ALPHA"):
        wai.SAO(gae_alpha=0)
    with pytest.raises(ValueError, match="SAO_GAE_ALPHA"):
        wai.SAO(gae_alpha=-1.5)
    with pytest.raises(ValueError, match="SAO_CRITIC_STEPS"):
        wai.SAO(critic_steps=0)
    with pytest.raises(ValueError, match="SAO_CRITIC_WARMUP"):
        wai.SAO(critic_warmup=-1)
    with pytest.raises(ValueError, match="SAO_TEMPERATURE"):
        wai.SAO(temperature=0)
    with pytest.raises(ValueError, match="SAO_MAX_TOKENS"):
        wai.SAO(max_tokens=0)
    with pytest.raises(ValueError, match="SAO_LEARNING_RATE"):
        wai.SAO(learning_rate=0.0)
    with pytest.raises(ValueError, match="SAO_CRITIC_LEARNING_RATE"):
        wai.SAO(critic_learning_rate=-5e-6)
    assert wai.SAO(ratio=defaults.SAO_RATIO_CODING).ratio == (0.2, 4.0)


def test_update_refuses_a_batch_it_cannot_read():
    m = wai.SAO()
    with pytest.raises(ValueError, match="empty"):
        m.update([])
    with pytest.raises(ValueError, match=r"trajectory 0 has no 'values'.*FlashReinforce"):
        m.update([{"reward": 1.0, "logprobs": [-1.0, -1.0]}])
    with pytest.raises(ValueError, match="trajectory 1 has no 'values'"):
        m.update([_traj(1.0, [0.5, 0.5]), {"reward": 0.0, "logprobs": [-1.0]}])
    with pytest.raises(ValueError, match="one per token"):
        m.update([{"reward": 1.0, "logprobs": [-1.0, -1.0], "values": [0.5]}])
    with pytest.raises(ValueError, match="no 'reward'"):
        m.update([{"logprobs": [-1.0], "values": [0.5]}])


# --- hand-computed cases ----------------------------------------------------


def test_three_token_gae_by_hand():
    # values V = [0.2, 0.5, 0.6], reward 1 on the last token, gamma 1, L = 3
    # lambda = 1 - 1/(1.5 * 3) = 1 - 1/4.5 = 7/9
    # delta_2 = 1 + 0      - 0.6 = 0.4
    # delta_1 = 0 + V(2)   - 0.5 = 0.6 - 0.5 = 0.1
    # delta_0 = 0 + V(1)   - 0.2 = 0.5 - 0.2 = 0.3
    # A_2 = 0.4
    # A_1 = 0.1 + (7/9) * 0.4        = 0.41111...
    # A_0 = 0.3 + (7/9) * 0.41111... = 0.61975...
    lam = 7 / 9
    a2 = 0.4
    a1 = 0.1 + lam * a2
    a0 = 0.3 + lam * a1
    update = wai.SAO().update([_traj(1.0, [0.2, 0.5, 0.6])])
    assert update.advantages[0] == pytest.approx([a0, a1, a2])
    assert update.stats["lambda_mean"] == pytest.approx(lam)
    # on-policy, ratio 1, token mean over the 3 action tokens in the batch
    assert update.coefficients[0] == pytest.approx([a0 / 3, a1 / 3, a2 / 3])
    # lambda_critic = 1: the Monte Carlo return, the reward at every token
    assert update.value_targets == [[1.0, 1.0, 1.0]]
    assert update.admitted == [True]
    assert update.stats["mean_advantage"] == pytest.approx((a0 + a1 + a2) / 3)
    assert update.stats["action_tokens"] == 3
    # a failing two-token trajectory in the same batch: delta_1 = 0 - 0.3, delta_0 = 0.3 - 0.4
    # lambda = 1 - 1/3 = 2/3, A_1 = -0.3, A_0 = -0.1 + (2/3)(-0.3) = -0.3
    both = wai.SAO().update([_traj(1.0, [0.2, 0.5, 0.6]), _traj(0.0, [0.4, 0.3])])
    assert both.advantages[1] == pytest.approx([-0.3, -0.3])
    assert both.value_targets[1] == [0.0, 0.0]
    assert both.coefficients[0] == pytest.approx([a0 / 5, a1 / 5, a2 / 5])  # N is now 5
    assert both.stats["lambda_mean"] == pytest.approx((7 / 9 + 2 / 3) / 2)


def test_length_adaptive_lambda_keeps_the_terminal_weight_near_exp_minus_one_over_alpha():
    m = wai.SAO()
    assert m.gae_lambda(4) == pytest.approx(1 - 1 / (ALPHA * 4))  # 5/6
    assert m.gae_lambda(40) == pytest.approx(1 - 1 / (ALPHA * 40))  # 59/60
    target = math.exp(-1 / ALPHA)  # 0.5134
    for length in (4, 40, 400, 4000):
        weight = m.gae_lambda(length) ** (length - 1)
        assert abs(weight - target) < 0.07, (length, weight)
    # the same lambda comes out of an update of that many tokens
    update = m.update([_traj(1.0, [0.0] * 40)])
    assert update.stats["lambda_mean"] == pytest.approx(59 / 60)
    # the reward reaches the first token with weight lambda ** (L - 1) when V is 0 everywhere
    assert update.advantages[0][0] == pytest.approx((59 / 60) ** 39)
    assert wai.SAO(gae_alpha=0.1).gae_lambda(2) == 0.0  # floored, never negative
    assert m.gae_lambda(0) == 0.0


def test_band_masks_a_token_outside_it_and_passes_the_ratio_inside():
    # three tokens, V = 0 everywhere, reward 1: A = [lam^2, lam, 1] with lam = 7/9
    lam = 7 / 9
    lp = [-1.0, -1.0, -1.0]
    # token 0 ratio 10 (above 6), token 1 ratio 0.5 (below 0.7), token 2 ratio 2 (inside)
    behavior = [-1.0 - math.log(10), -1.0 - math.log(0.5), -1.0 - math.log(2)]
    update = wai.SAO().update([_traj(1.0, [0.0, 0.0, 0.0], lp, behavior)])
    assert update.coefficients[0][0] == 0.0 and update.coefficients[0][1] == 0.0
    assert update.coefficients[0][2] == pytest.approx(2 * 1.0 / 3)  # r * A / N
    assert update.advantages[0] == pytest.approx([lam**2, lam, 1.0])  # A is before the mask
    assert update.stats["masked_token_share"] == pytest.approx(2 / 3)
    assert update.stats["mean_ratio"] == pytest.approx(2.0)  # over the tokens in the band
    assert update.admitted == [True]
    assert any("masked 2 of 3" in n for n in update.notes)
    # the coding band (0.2, 4.0) admits the 0.5 token and still drops the 10
    coding = wai.SAO(ratio=defaults.SAO_RATIO_CODING).update(
        [_traj(1.0, [0.0, 0.0, 0.0], lp, behavior)]
    )
    assert coding.coefficients[0][0] == 0.0
    assert coding.coefficients[0][1] == pytest.approx(0.5 * lam / 3)
    assert coding.stats["masked_token_share"] == pytest.approx(1 / 3)
    # the band is open: a ratio exactly on the edge is out, whichever sign the advantage has
    edge = wai.SAO().update(
        [_traj(0.0, [0.5, 0.5], [-1.0, -1.0], [-1.0 - math.log(6.0), -1.0 - math.log(0.7)])]
    )
    assert edge.coefficients == [[0.0, 0.0]] and edge.admitted == [False]
    assert any("outside the band" in n for n in edge.notes)
    # an overflowing ratio is an excursion, not an exception
    huge = wai.SAO().update([_traj(1.0, [0.0], [0.0], [-1000.0])])
    assert huge.coefficients == [[0.0]] and huge.admitted == [False]


def test_on_policy_ratio_is_exactly_one_so_the_coefficient_is_the_advantage():
    update = wai.SAO().update([_traj(1.0, [0.1, 0.4]), _traj(0.0, [0.3, 0.2, 0.5])])
    n = update.stats["action_tokens"]
    assert n == 5
    for coef, adv in zip(update.coefficients, update.advantages, strict=True):
        for c, a in zip(coef, adv, strict=True):
            assert c * n == pytest.approx(a)
    assert update.stats["mean_ratio"] == 1.0
    assert update.stats["masked_token_share"] == 0.0
    assert any("on-policy" in n for n in update.notes)
    # behavior_logprobs equal to logprobs is the same thing, without the note
    same = wai.SAO().update([_traj(1.0, [0.1, 0.4], [-1.0, -2.0], [-1.0, -2.0])])
    assert same.stats["mean_ratio"] == 1.0 and not any("on-policy" in n for n in same.notes)


def test_observation_tokens_get_no_gradient_and_the_bootstrap_skips_over_them():
    # the same three actions, with a tool observation between the second and the third
    plain = wai.SAO().update([_traj(1.0, [0.2, 0.5, 0.6])])
    agentic = wai.SAO().update([_traj(1.0, [0.2, 0.5, 0.9, 0.6], mask=[True, True, False, True])])
    a = agentic.advantages[0]
    assert [a[0], a[1], a[3]] == pytest.approx(plain.advantages[0])
    assert a[2] == 0.0  # the observation has no advantage
    assert agentic.coefficients[0][2] == 0.0
    # the same L = 3 action tokens, so the same lambda and the same 1/N
    assert agentic.stats["lambda_mean"] == pytest.approx(plain.stats["lambda_mean"])
    assert agentic.stats["action_tokens"] == 3
    c = agentic.coefficients[0]
    assert [c[0], c[1], c[3]] == pytest.approx(plain.coefficients[0])
    # the critic target on the observation is the return of the next action token
    assert agentic.value_targets[0] == [1.0, 1.0, 1.0, 1.0]
    # a wildly wrong V on the observation changes nothing: it is never read
    wrong = wai.SAO().update([_traj(1.0, [0.2, 0.5, -50.0, 0.6], mask=[True, True, False, True])])
    assert wrong.advantages[0] == pytest.approx(a)
    # a trailing observation after the last action carries the trajectory's return
    tail = wai.SAO().update([_traj(1.0, [0.2, 0.6, 0.0], mask=[True, True, False])])
    assert tail.value_targets[0] == [1.0, 1.0, 1.0]
    assert tail.advantages[0][1] == pytest.approx(0.4) and tail.advantages[0][2] == 0.0


def test_a_fully_masked_trajectory_is_not_admitted():
    update = wai.SAO().update(
        [
            _traj(1.0, [0.2, 0.5], mask=[False, False]),
            _traj(1.0, [0.2, 0.5, 0.6]),
        ]
    )
    assert update.admitted == [False, True]
    assert update.n_admitted == 1 and update.n == 2
    assert update.coefficients[0] == [0.0, 0.0]
    assert update.value_targets[0] == [1.0, 1.0]
    assert update.stats["admitted_share"] == 0.5
    assert update.stats["action_tokens"] == 3  # the masked one adds nothing to N
    assert any("no action token" in n for n in update.notes)
    only = wai.SAO().update([_traj(1.0, [0.2, 0.5], mask=[False, False])])
    assert only.admitted == [False] and only.stats["action_tokens"] == 0
    assert only.stats["lambda_mean"] == 0.0


def test_update_prints_itself():
    update = wai.SAO().update([_traj(1.0, [0.2, 0.5, 0.6]), _traj(0.0, [0.4, 0.3])])
    text = str(update)
    assert text.startswith("sao update: 2 of 2 trajectories admitted")
    assert "masked token share" in text and "lambda mean" in text
    assert "admitted" in update._repr_html_()


# --- the proof that it learns ---------------------------------------------
#
# A contextual sequence task: 4 prompts, each with a target sequence of 3
# tokens from a vocabulary of 4; reward 1 when the sampled sequence is the
# target, else 0 (chance 1/64). The policy is tabular softmax logits per
# state (prompt, prefix), the critic a tabular V[state] trained toward
# ``value_targets`` by plain SGD for ``critic_steps`` inner steps per policy
# step. The policy step is the update rule as a trainer applies it: the
# gradient of log softmax at the sampled action is onehot(a) - pi(.|s), so
# logits[s] += lr * coefficient * (onehot(a) - pi(.|s)).

VOCAB = 4
LENGTH = 3
PROMPTS = 4
REPEATS = 4  # batch of 16, one rollout per prompt occurrence
STEPS = 150
LR = 20.0  # the coefficients carry 1/N for N = 48 tokens, so the step is about 0.4 per token
CRITIC_LR = 0.5


def _softmax(logits):
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    z = sum(e)
    return [x / z for x in e]


def _train(seed: int, lag: int) -> tuple[list[float], list[float]]:
    """Mean batch reward and masked-token share per step. ``lag`` is how
    many policy updates behind the sampler runs (0: on-policy)."""
    rng = random.Random(seed)
    method = wai.SAO()
    targets = [tuple(rng.randrange(VOCAB) for _ in range(LENGTH)) for _ in range(PROMPTS)]
    logits: dict[tuple, list[float]] = {}
    values: dict[tuple, float] = {}
    history: list[dict[tuple, list[float]]] = []
    rewards: list[float] = []
    masked: list[float] = []
    for _step in range(STEPS):
        history.append(copy.deepcopy(logits))
        behavior = history[max(0, len(history) - 1 - lag)]
        batch, states = [], []
        for p in list(range(PROMPTS)) * REPEATS:
            prefix: tuple[int, ...] = ()
            traj_states, actions, blp, lp, vs = [], [], [], [], []
            for _ in range(LENGTH):
                s = (p, prefix)
                pb = _softmax(behavior.get(s, [0.0] * VOCAB))
                a = rng.choices(range(VOCAB), weights=pb)[0]
                pc = _softmax(logits.setdefault(s, [0.0] * VOCAB))
                traj_states.append(s)
                actions.append(a)
                blp.append(math.log(pb[a]))
                lp.append(math.log(pc[a]))
                vs.append(values.get(s, 0.0))
                prefix = (*prefix, a)
            reward = 1.0 if prefix == targets[p] else 0.0
            batch.append({"reward": reward, "logprobs": lp, "behavior_logprobs": blp, "values": vs})
            states.append((traj_states, actions))
        update = method.update(batch)
        assert update.value_targets is not None
        rewards.append(sum(t["reward"] for t in batch) / len(batch))
        masked.append(update.stats["masked_token_share"])
        for _ in range(method.critic_steps):  # K critic updates per policy update
            for (traj_states, _actions), tgt in zip(states, update.value_targets, strict=True):
                for s, y in zip(traj_states, tgt, strict=True):
                    v = values.get(s, 0.0)
                    values[s] = v - CRITIC_LR * (v - y)  # SGD on the squared error
        for (traj_states, actions), coefs in zip(states, update.coefficients, strict=True):
            for s, a, c in zip(traj_states, actions, coefs, strict=True):
                if c == 0.0:
                    continue
                row = logits[s]
                pi = _softmax(row)
                for k in range(VOCAB):
                    row[k] += LR * c * ((1.0 if k == a else 0.0) - pi[k])
    return rewards, masked


def test_toy_policy_learns_on_policy():
    rewards, masked = _train(seed=0, lag=0)
    assert rewards[0] < 0.2  # about chance, 1/64
    assert sum(rewards[-10:]) / 10 > 0.8, rewards[-10:]
    assert max(masked) == 0.0  # on-policy: every ratio is 1, nothing to mask


def test_toy_policy_still_learns_when_the_sampler_lags_three_updates():
    rewards, masked = _train(seed=0, lag=3)
    assert rewards[0] < 0.2
    assert sum(rewards[-10:]) / 10 > 0.8, rewards[-10:]
    assert max(masked) > 0.0  # DIS dropped the tokens the policy had moved away from
