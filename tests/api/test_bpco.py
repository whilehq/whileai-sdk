"""wai.BPCO: best practice critic optimization (Qi, Zhou and Lee 2026,
arXiv:2608.23566) as an object with cited defaults, refused on a bad value,
whose ``update(batch)`` is the paper's update rule in plain Python.

The hand-computed cases write the paper's formulas out next to the number
they check. The toy at the end is the proof that the rule learns: a tabular
softmax policy on a contextual sequence task, a tabular critic bounded
through ``bound()`` and trained on ``value_targets``, a critic warm-up, and
a second run whose behaviour policy lags the trained one. No network, no
numpy, no torch; the whole file runs in a few seconds.
"""

from __future__ import annotations

import inspect
import math
import random

import pytest

import whileai as wai
from whileai.simulations import defaults

ALPHA = defaults.BPCO_GAE_ALPHA  # 0.4: lambda(L) = 1 - 1/(ALPHA * L)
CLIP = defaults.BPCO_CLIP  # 0.2: |pi - mu| <= CLIP, a ratio range of 1 -/+ CLIP/mu


def _traj(reward, values, logprobs=None, behavior=None, mask=None):
    row = {"reward": reward, "values": list(values)}
    row["logprobs"] = list(logprobs) if logprobs is not None else [-1.0] * len(values)
    if behavior is not None:
        row["behavior_logprobs"] = list(behavior)
    if mask is not None:
        row["action_mask"] = list(mask)
    return row


# --- the front door and the cited defaults --------------------------------


def test_front_door_and_defaults_are_the_named_constants():
    assert wai.BPCO is wai.methods.BPCO
    m = wai.BPCO()
    assert m.name == "bpco" and m.samples == 1
    assert m.clip == defaults.BPCO_CLIP
    assert m.gae_alpha == defaults.BPCO_GAE_ALPHA
    assert m.reward_range == defaults.BPCO_REWARD_RANGE == (0.0, 1.0)
    assert m.critic_warmup == defaults.BPCO_CRITIC_WARMUP
    assert m.temperature == defaults.BPCO_TEMPERATURE
    assert m.max_tokens == defaults.BPCO_MAX_TOKENS
    assert m.critic_learning_rate == defaults.BPCO_CRITIC_LEARNING_RATE
    assert m.learning_rate is None
    assert m.default_learning_rate(lora=True) == defaults.BPCO_LEARNING_RATE
    assert m.default_learning_rate(lora=False) == defaults.BPCO_LEARNING_RATE
    assert wai.BPCO(learning_rate=3e-6).default_learning_rate(lora=False) == 3e-6
    text = str(m)
    assert "\n" not in text and text.startswith("BPCO(") and "clip=0.2" in text
    assert "gae_alpha=0.4" in text and "critic_warmup=15" in text


def test_every_default_is_cited_to_the_paper_in_defaults_source():
    src = inspect.getsource(defaults)
    for name in (
        "BPCO_CLIP",
        "BPCO_GAE_ALPHA",
        "BPCO_REWARD_RANGE",
        "BPCO_CRITIC_WARMUP",
        "BPCO_LEARNING_RATE",
        "BPCO_CRITIC_LEARNING_RATE",
        "BPCO_TEMPERATURE",
        "BPCO_MAX_TOKENS",
        "BPCO_LOG_RATIO_CAP",
    ):
        assert f"# {name} = " in src or f"/ {name} = " in src, name
    block = src[src.index("# --- BPCO") : src.index("# PRIME_RL_GPUS")]
    # the header and every knob the paper prints (clip, alpha, range, warm-up, the two
    # rates, max_tokens) name it; temperature and the exp guard say the paper does not
    assert block.count("2608.23566") >= 7
    assert block.count("convention, untested") == 2


# --- refusals --------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"clip": 0}, "BPCO_CLIP"),
        ({"clip": -0.1}, "BPCO_CLIP"),
        ({"clip": 1.5}, r"in \(0, 1\]"),
        ({"gae_alpha": 0}, "BPCO_GAE_ALPHA"),
        ({"gae_alpha": -1}, "gae_alpha must be positive"),
        ({"reward_range": (1.0, 1.0)}, "low < high"),
        ({"reward_range": (2.0, 1.0)}, "BPCO_REWARD_RANGE"),
        ({"reward_range": (0.0, math.inf)}, "finite"),
        ({"reward_range": (0.0,)}, "pair"),
        ({"critic_warmup": -1}, "BPCO_CRITIC_WARMUP"),
        ({"temperature": 0}, "temperature"),
        ({"temperature": 3}, r"\(0, 2\]"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"learning_rate": 0}, "BPCO_LEARNING_RATE"),
        ({"critic_learning_rate": -1e-5}, "BPCO_CRITIC_LEARNING_RATE"),
    ],
)
def test_constructor_refuses_a_bad_knob_and_names_the_constant(kwargs, match):
    with pytest.raises(ValueError, match=match):
        wai.BPCO(**kwargs)


def test_update_refuses_a_batch_it_cannot_read():
    m = wai.BPCO()
    with pytest.raises(ValueError, match="empty"):
        m.update([])
    with pytest.raises(ValueError, match="no 'reward'"):
        m.update([{"logprobs": [-1.0], "values": [0.5]}])
    with pytest.raises(ValueError, match="needs 'logprobs'"):
        m.update([{"reward": 1.0, "values": [0.5]}])
    with pytest.raises(ValueError, match=r"no 'values'.*bound\(z\)"):
        m.update([{"reward": 1.0, "logprobs": [-1.0, -1.0]}])
    with pytest.raises(ValueError, match="2 values for 3 logprobs"):
        m.update([_traj(1.0, [0.5, 0.5], logprobs=[-1.0, -1.0, -1.0])])
    with pytest.raises(ValueError, match=r"1\.2 at token 1 is outside reward_range.*bound\(\)"):
        m.update([_traj(1.0, [0.5, 1.2, 0.5])])
    with pytest.raises(ValueError, match=r"-0\.5 at token 0 is outside"):
        m.update([_traj(0.0, [-0.5, 0.5, 0.5])])
    with pytest.raises(ValueError, match="open interval"):
        m.unbound(1.0)


# --- bound(): equation 9 -----------------------------------------------------


def test_bound_is_the_scaled_arctangent_into_the_reward_range():
    m = wai.BPCO()
    assert m.bound(0.0) == 0.5  # the midpoint
    assert 0.999 < m.bound(1e6) < 1.0  # approaches the top, never reaches it
    assert 0.0 < m.bound(-1e6) < 0.001
    grid = [m.bound(z) for z in range(-20, 21)]
    assert grid == sorted(grid) and len(set(grid)) == len(grid)  # strictly monotone
    # V = R_min + (R_max - R_min)(1/2 + atan(z)/pi), written out for z = 1
    assert m.bound(1.0) == pytest.approx(0.5 + math.atan(1.0) / math.pi)
    wide = wai.BPCO(reward_range=(-1.0, 3.0))
    assert wide.bound(0.0) == 1.0
    assert wide.bound(1.0) == pytest.approx(-1.0 + 4.0 * (0.5 + math.atan(1.0) / math.pi))
    for z in (-3.0, -0.2, 0.0, 0.7, 5.0):
        assert wide.unbound(wide.bound(z)) == pytest.approx(z)
    assert m.unbound(0.5) == 0.0


# --- the advantage: equations 3, 4 and 14 -------------------------------------


def test_three_token_gae_by_hand_with_length_adaptive_lambda():
    m = wai.BPCO()
    values = [0.2, 0.5, 0.8]
    reward = 1.0
    # lambda(3) = 1 - 1/(0.4 * 3) = 1/6
    lam = 1 - 1 / (ALPHA * 3)
    assert lam == pytest.approx(1 / 6)
    # delta_t = r_t + V(s_{t+1}) - V(s_t), r = 0 before the last token, R at it, V past the end 0
    d3 = reward + 0.0 - 0.8  # 0.2
    d2 = 0.0 + 0.8 - 0.5  # 0.3
    d1 = 0.0 + 0.5 - 0.2  # 0.3
    a3 = d3  # 0.2
    a2 = d2 + lam * a3  # 0.3 + 0.2/6 = 0.3333
    a1 = d1 + lam * a2  # 0.3 + 0.3333/6 = 0.3556
    u = m.update([_traj(reward, values)])
    assert u.advantages[0] == pytest.approx([a1, a2, a3])
    assert u.advantages[0] == pytest.approx([0.35556, 0.33333, 0.2], abs=1e-4)
    assert u.stats["mean_lambda"] == pytest.approx(1 / 6)
    assert u.value_targets == [[1.0, 1.0, 1.0]]
    assert u.admitted == [True]


def test_lambda_tracks_length_so_the_terminal_reward_weight_holds():
    m = wai.BPCO()
    assert 1 - 1 / (ALPHA * 4) == 0.375
    assert 1 - 1 / (ALPHA * 40) == 0.9375
    assert m.update([_traj(1.0, [0.0] * 4)]).stats["mean_lambda"] == pytest.approx(0.375)
    assert m.update([_traj(1.0, [0.0] * 40)]).stats["mean_lambda"] == pytest.approx(0.9375)
    # with V = 0 everywhere only the terminal residual is nonzero, so the first
    # token's advantage is lambda^(L-1) * R: the weight the paper says stays
    # near exp(-1/alpha) = exp(-2.5) = 0.082 whatever the length (section 3.6)
    target = math.exp(-1 / ALPHA)
    for length in (40, 400):
        first = m.update([_traj(1.0, [0.0] * length)]).advantages[0][0]
        assert first == pytest.approx((1 - 1 / (ALPHA * length)) ** (length - 1))
        assert abs(first - target) / target < 0.05
    # a fixed lambda = 0.99 would give the first of 400 tokens 0.99^399 = 0.018 instead
    assert target / 4 > 0.99**399


def test_short_trajectories_clamp_lambda_at_zero_and_say_so():
    u = wai.BPCO().update([_traj(1.0, [0.5, 0.5])])  # L = 2 < 1/alpha = 2.5
    assert u.stats["mean_lambda"] == 0.0
    assert u.advantages[0] == pytest.approx([0.0, 0.5])  # one-step TD residuals
    assert any("clamped at 0" in n and "unspecified" in n for n in u.notes)


def test_advantages_are_not_normalized():
    m = wai.BPCO(reward_range=(0.0, 3.0))
    base = [_traj(1.0, [0.0] * 3), _traj(1.0, [0.0] * 3), _traj(0.0, [0.0] * 3)]
    tripled = [_traj(3.0, [0.0] * 3), _traj(3.0, [0.0] * 3), _traj(0.0, [0.0] * 3)]
    a1 = m.update(base).advantages
    a3 = m.update(tripled).advantages
    for row1, row3 in zip(a1, a3):
        assert row3 == pytest.approx([3 * x for x in row1])
    flat = [x for row in a1 for x in row]
    mean = sum(flat) / len(flat)
    assert mean != pytest.approx(0.0)  # no batch mean subtracted
    assert max(flat) == pytest.approx(1.0)  # the terminal residual is the raw reward


def test_value_targets_are_the_monte_carlo_reward_at_every_token():
    m = wai.BPCO()
    u = m.update(
        [
            _traj(1.0, [0.1, 0.9, 0.4, 0.2]),
            _traj(0.0, [0.7, 0.7, 0.7], mask=[1, 0, 1]),
        ]
    )
    assert u.value_targets == [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]


def test_environment_tokens_carry_no_value_residual_or_coefficient():
    m = wai.BPCO()
    # the middle token is the environment's: V skips it, L = 3 counts the policy's tokens
    u = m.update([_traj(1.0, [0.2, 0.9, 0.5, 0.8], mask=[1, 0, 1, 1])])
    lam = 1 - 1 / (ALPHA * 3)
    d4 = 1.0 - 0.8
    d3 = 0.8 - 0.5
    d1 = 0.5 - 0.2  # V(s_2) is the next policy token's value, 0.5, not 0.9
    a4 = d4
    a3 = d3 + lam * a4
    a1 = d1 + lam * a3
    assert u.advantages[0] == pytest.approx([a1, 0.0, a3, a4])
    assert u.coefficients[0][1] == 0.0
    assert u.stats["mean_lambda"] == pytest.approx(lam)
    silent = m.update([_traj(1.0, [0.5, 0.5]), _traj(1.0, [0.5, 0.5, 0.5], mask=[0, 0, 0])])
    assert silent.admitted == [True, False]
    assert silent.coefficients[1] == [0.0, 0.0, 0.0]
    assert silent.stats["admitted_share"] == 0.5
    assert any("every token masked" in n for n in silent.notes)


# --- the surrogate: DPPO, equation 2 ---------------------------------------


def test_on_policy_ratio_is_one_so_the_coefficient_is_the_advantage():
    m = wai.BPCO()
    u = m.update([_traj(1.0, [0.2, 0.5, 0.8]), _traj(0.0, [0.3, 0.6, 0.1])])
    assert u.coefficients == u.advantages
    assert u.stats["mean_ratio"] == 1.0
    assert u.stats["clipped_token_share"] == 0.0
    # behavior_logprobs equal to logprobs is the same thing, said explicitly
    lp = [-0.4, -1.3, -0.1]
    v = m.update([_traj(1.0, [0.2, 0.5, 0.8], logprobs=lp, behavior=lp)])
    assert v.coefficients == v.advantages


def test_dppo_range_widens_for_a_rare_token():
    m = wai.BPCO()
    # values 0.5 everywhere, reward 1: only the last token has a residual, A_3 = 0.5 > 0
    values = [0.5, 0.5, 0.5]
    on = [-1.0, -1.0]
    # the last token's ratio is 1.5: mu = 0.5, pi = 0.75
    common = _traj(1.0, values, logprobs=[*on, math.log(0.75)], behavior=[*on, math.log(0.5)])
    rare = _traj(1.0, values, logprobs=[*on, math.log(0.075)], behavior=[*on, math.log(0.05)])
    u = m.update([common, rare])
    # mu = 0.5: range 1 -/+ 0.2/0.5 = [0.6, 1.4]; 1.5 is past the top on the side the
    # advantage favors, so the clipped branch is the minimum and the gradient is 0
    assert u.coefficients[0][2] == 0.0
    # mu = 0.05: range 1 -/+ 0.2/0.05 = [-3, 5]; 1.5 is inside, coefficient rho * A = 1.5 * 0.5
    assert u.coefficients[1][2] == pytest.approx(0.75)
    assert u.advantages[0][2] == u.advantages[1][2] == pytest.approx(0.5)
    assert u.stats["clipped_token_share"] == pytest.approx(1 / 6)
    half_widths = [CLIP / math.exp(-1.0)] * 4 + [CLIP / 0.5, CLIP / 0.05]
    assert u.stats["mean_clip_range"] == pytest.approx(sum(half_widths) / 6)
    assert u.stats["mean_ratio"] == pytest.approx((4 * 1.0 + 1.5 + 1.5) / 6)


def test_dppo_clips_a_negative_advantage_only_below_its_range():
    m = wai.BPCO()
    values = [0.5, 0.5, 0.5]  # reward 0: A_3 = -0.5
    on = [-1.0, -1.0]
    # ratio 0.5 with mu = 0.5: below the range floor 0.6 on the side A < 0 favors: clipped
    down = _traj(0.0, values, logprobs=[*on, math.log(0.25)], behavior=[*on, math.log(0.5)])
    # ratio 0.5 with mu = 0.05: the floor is -3, so it is inside and keeps rho * A
    down_rare = _traj(0.0, values, logprobs=[*on, math.log(0.025)], behavior=[*on, math.log(0.05)])
    # ratio 1.5 with A < 0: past the top, but there the unclipped branch is the minimum
    up = _traj(0.0, values, logprobs=[*on, math.log(0.75)], behavior=[*on, math.log(0.5)])
    u = m.update([down, down_rare, up])
    assert u.coefficients[0][2] == 0.0
    assert u.coefficients[1][2] == pytest.approx(0.5 * -0.5)
    assert u.coefficients[2][2] == pytest.approx(1.5 * -0.5)


def test_ratio_overflow_is_capped():
    m = wai.BPCO()
    u = m.update(
        [_traj(1.0, [0.5, 0.5, 0.5], logprobs=[-1.0, -1.0, -1.0], behavior=[-1, -1, -900])]
    )
    assert math.isfinite(u.coefficients[0][2])
    assert u.stats["mean_ratio"] <= (2 + math.exp(defaults.BPCO_LOG_RATIO_CAP)) / 3


def test_update_prints_itself_and_reports_explained_variance():
    m = wai.BPCO()
    u = m.update([_traj(1.0, [0.9, 0.9, 0.9]), _traj(0.0, [0.1, 0.1, 0.1])])
    text = str(u)
    assert "bpco update: 2 of 2 trajectories admitted" in text and "admitted" in text
    assert "clipped token share" in text and "explained variance" in text
    # EV = 1 - Var(R - V) / Var(R): errors +0.1 and -0.1 (variance 0.01) over rewards 1 and 0
    # (variance 0.25), so EV = 1 - 0.04 = 0.96
    assert u.stats["explained_variance"] == pytest.approx(0.96)
    same = m.update([_traj(1.0, [0.5, 0.5, 0.5]), _traj(1.0, [0.5, 0.5, 0.5])])
    assert "explained_variance" not in same.stats
    assert any("explained variance not computed" in n for n in same.notes)
    assert "<pre>" in u._repr_html_()


# --- the proof that it learns ----------------------------------------------

PROMPTS = 4
VOCAB = 4
LENGTH = 3
TARGETS = [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]
CHANCE = 1 / VOCAB**LENGTH  # 1/64 under the uniform starting policy


def _softmax(logits):
    top = max(logits)
    e = [math.exp(x - top) for x in logits]
    z = sum(e)
    return [x / z for x in e]


def _train(method, *, seed, steps, rate, critic_rate, lag=0, batch=16):
    """A tabular softmax policy and a tabular critic trained by ``method.update``.

    State is (prompt, prefix). The critic keeps a raw output ``z[state]`` and
    predicts ``method.bound(z)``; it trains on ``value_targets`` by SGD on the
    squared error through the arctangent: ``dV/dz = (R_max - R_min) / (pi (1 + z^2))``.
    For ``method.critic_warmup`` steps only the critic moves. The policy step
    is ``logits[s] += rate * mean_i sum_t coef[i][t] * (onehot(a_t) - pi(.|s))``,
    the gradient of the DPPO surrogate through log-softmax. With ``lag`` the
    rollouts come from the logits ``lag`` updates ago, so the ratio and the
    DPPO range are live.
    """
    rng = random.Random(seed)
    logits: dict = {}
    z: dict = {}
    history: list = []
    lo, hi = method.reward_range

    def rollout(behavior):
        p = rng.randrange(PROMPTS)
        prefix = ()
        tokens, lps, blps, vals = [], [], [], []
        for _ in range(LENGTH):
            s = (p, prefix)
            pi_b = _softmax(behavior.get(s, [0.0] * VOCAB))
            a = rng.choices(range(VOCAB), weights=pi_b)[0]
            pi = _softmax(logits.get(s, [0.0] * VOCAB))
            tokens.append(a)
            lps.append(math.log(pi[a]))
            blps.append(math.log(pi_b[a]))
            vals.append(method.bound(z.setdefault(s, 0.0)))
            prefix = (*prefix, a)
        return {
            "prompt": p,
            "tokens": tokens,
            "reward": 1.0 if tuple(tokens) == TARGETS[p] else 0.0,
            "logprobs": lps,
            "behavior_logprobs": blps,
            "values": vals,
        }

    curve, clipped, policy_steps = [], 0.0, 0
    for step in range(steps):
        behavior = history[-lag - 1] if lag and len(history) > lag else logits
        rows = [rollout(behavior) for _ in range(batch)]
        curve.append(sum(r["reward"] for r in rows) / batch)
        upd = method.update(rows)
        for row, targets in zip(rows, upd.value_targets):
            prefix = ()
            for t, a in enumerate(row["tokens"]):
                s = (row["prompt"], prefix)
                zz = z[s]
                grad = 2 * (method.bound(zz) - targets[t]) * (hi - lo) / (math.pi * (1 + zz * zz))
                z[s] = zz - critic_rate * grad
                prefix = (*prefix, a)
        if step < method.critic_warmup:
            continue
        policy_steps += 1
        clipped += upd.stats["clipped_token_share"]
        grads: dict = {}
        for row, coefs in zip(rows, upd.coefficients):
            prefix = ()
            for t, a in enumerate(row["tokens"]):
                s = (row["prompt"], prefix)
                pi = _softmax(logits.get(s, [0.0] * VOCAB))
                g = grads.setdefault(s, [0.0] * VOCAB)
                for k in range(VOCAB):
                    g[k] += coefs[t] * ((1.0 if k == a else 0.0) - pi[k]) / batch
                prefix = (*prefix, a)
        history.append({s: list(v) for s, v in logits.items()})
        for s, g in grads.items():
            old = logits.get(s, [0.0] * VOCAB)
            logits[s] = [x + rate * gg for x, gg in zip(old, g)]
    return curve, clipped / max(policy_steps, 1), logits, z


def test_toy_policy_learns_from_one_rollout_per_prompt_with_a_bounded_critic():
    m = wai.BPCO()  # the paper's knobs: clip 0.2, alpha 0.4, range (0, 1), warm-up 15
    curve, clipped, _logits, z = _train(m, seed=0, steps=300, rate=3.0, critic_rate=1.0)
    early = sum(curve[:10]) / 10
    late = sum(curve[-20:]) / 20
    assert early < 4 * CHANCE  # about chance at the start (0.019 at this seed)
    assert late > 0.8  # 0.98 at this seed
    assert clipped == 0.0  # on-policy: the ratio is 1, nothing to clip
    # the critic learned: a prefix two tokens into a target is worth nearly 1, the empty prefix
    # about the policy's own success rate
    for p, target in enumerate(TARGETS):
        assert m.bound(z[(p, target[:2])]) > 0.8
    assert all(0.0 <= m.bound(v) <= 1.0 for v in z.values())


def test_toy_critic_warmup_leaves_the_policy_untouched():
    m = wai.BPCO(critic_warmup=15)
    _, _, logits, z = _train(m, seed=0, steps=15, rate=3.0, critic_rate=1.0)
    assert logits == {}  # no policy step in the first critic_warmup updates
    assert z and any(v != 0.0 for v in z.values())  # the critic moved


def test_toy_still_learns_when_the_behavior_policy_lags_and_the_range_clips():
    m = wai.BPCO()
    curve, clipped, _, _ = _train(m, seed=0, steps=300, rate=8.0, critic_rate=1.0, lag=5)
    assert sum(curve[:10]) / 10 < 4 * CHANCE
    assert sum(curve[-20:]) / 20 > 0.8  # 1.0 at this seed
    assert clipped > 0.0  # some tokens moved more than clip = 0.2 in probability (0.2% here)
