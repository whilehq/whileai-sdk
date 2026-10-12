"""``Selection.trim``: the order, the curve fit and the stopping rule."""

from __future__ import annotations

import random

import pytest

from whileai.selection import Selection
from whileai.trim import _n_star, fit_saturation, order, trim_sizes

# a known curve: the untuned model at 0.30, rising to 0.70 (rectified form)
TRUE = (0.30, 0.70, 1.0, 200.0)


def _true_p(n: float) -> float:
    base, a, c, d = TRUE
    return base + (a - base) * n**c / (d + n**c)


def _simulate(sizes, *, items=500, seeds=2, rng_seed=0):
    """Per-item 0/1 scores per seed; item i passes when its fixed
    difficulty is under the model's pass rate (items are paired)."""
    rng = random.Random(rng_seed)
    difficulty = [rng.random() for _ in range(items)]
    curve = {}
    for n in sizes:
        runs = []
        for _ in range(seeds):
            p = _true_p(n) + rng.gauss(0, 0.005)
            runs.append([1.0 if u < p else 0.0 for u in difficulty])
        curve[n] = runs
    base = [[1.0 if u < TRUE[0] else 0.0 for u in difficulty]]
    return curve, base


def _rows(prompts, responses=None, rewards=None):
    responses = responses or ["ok"] * len(prompts)
    rewards = rewards or [1.0] * len(prompts)
    return Selection(
        [
            {"prompt": p, "response": r, "reward": w, "id": i}
            for i, (p, r, w) in enumerate(zip(prompts, responses, rewards, strict=True))
        ],
        mode="sft",
    )


SIZES = [64 * 2**i for i in range(9)]  # 64 .. 16,384


# -- order ----------------------------------------------------------------


def test_random_order_nests_and_is_seeded():
    rows = _rows([f"question {i} about item {i * 3}" for i in range(50)])
    small, big = rows.trim(10), rows.trim(20)
    assert [r["id"] for r in small] == [r["id"] for r in big][:10]
    assert [r["id"] for r in rows.trim(10, seed=1)] != [r["id"] for r in small]
    assert small.report["n_before_trim"] == 50
    assert "random order" in str(small)


def test_duplicates_go_last():
    rows = _rows(
        ["sum sales by region", "Sum sales, by region!", "list overdue invoices", "plot revenue"]
    )
    for by in ("random", "bigram"):
        assert order(rows, by)[-1] == 1


def test_near_duplicates_by_embedding_go_last():
    vec = {"a": [1.0, 0.0], "b": [0.99, 0.01], "c": [0.0, 1.0]}
    rows = _rows(["a", "b", "c"])
    idx = order(rows, "random", embed=lambda texts: [vec[t.split()[0]] for t in texts])
    assert idx[-1] == 1


def test_diverse_round_robins_clusters():
    prompts = [f"x{i}" for i in range(6)] + [f"y{i}" for i in range(6)]

    def embed(texts):
        return [
            [1.0, 0.1 * i] if t.startswith("x") else [-1.0, 0.1 * i] for i, t in enumerate(texts)
        ]

    idx = order(_rows(prompts), "diverse", embed=embed)
    first_two = {prompts[i][0] for i in idx[:2]}
    assert first_two == {"x", "y"}  # one from each cluster before a second from either


def test_target_ranks_closest_to_dev_first():
    vec = {"tax": [1.0, 0.0], "cooking": [0.0, 1.0], "audit": [0.6, 0.4]}
    rows = _rows(["cooking", "audit", "tax"])
    idx = order(rows, "target", embed=lambda ts: [vec[t.split()[0]] for t in ts], dev=["tax"])
    assert [rows[i]["prompt"] for i in idx] == ["tax", "audit", "cooking"]


def test_rankers_that_need_embeddings_say_so():
    with pytest.raises(ValueError, match="embed="):
        order(_rows(["a b"]), "diverse")
    with pytest.raises(ValueError, match="dev="):
        order(_rows(["a b"]), "target", embed=lambda ts: [[1.0] for _ in ts])
    with pytest.raises(ValueError, match="one of"):
        order(_rows(["a b"]), "quality")


def test_trim_sizes_double_to_len():
    assert trim_sizes(1000) == [64, 128, 256, 512, 1000]
    assert trim_sizes(50) == [50]


# -- curve and stopping rule ------------------------------------------------


def test_recovers_known_saturation_point():
    truth = _n_star("rectified", TRUE, 0.01, 1)  # about 4,000 rows
    curve, base = _simulate(SIZES)
    fit = fit_saturation(curve, base=base, draws=40)
    assert fit.verdict == "saturated"
    assert truth / 1.5 < fit.n_saturated < truth * 1.5
    assert fit.interval is not None and fit.interval[0] <= fit.n_saturated <= fit.interval[1]
    assert "N* 90% range" in str(fit)


def test_n_star_does_not_grow_with_test_size():
    # the first version's threshold was the test's error bar, so N* grew
    # with test_n on an identical curve; gain per doubling must not
    small = fit_saturation(
        _simulate(SIZES, items=200)[0], base=_simulate(SIZES, items=200)[1], draws=0
    )
    large = fit_saturation(
        _simulate(SIZES, items=2000)[0], base=_simulate(SIZES, items=2000)[1], draws=0
    )
    assert 0.6 < small.n_saturated / large.n_saturated < 1.6


def test_curve_still_climbing_is_a_forecast():
    curve, base = _simulate(SIZES[:5])  # stops at 1,024; truth is about 4,000
    fit = fit_saturation(curve, base=base, draws=40)
    assert fit.verdict in ("forecast", "not saturated")
    assert not fit.saturated


def test_flat_curve():
    fit = fit_saturation({64: 0.60, 128: 0.59, 256: 0.60, 512: 0.58, 1024: 0.59}, base=0.6)
    assert fit.saturated and fit.n_saturated <= 128


def test_one_score_per_size_says_no_interval():
    fit = fit_saturation({n: _true_p(n) for n in SIZES}, base=TRUE[0])
    assert fit.interval is None
    assert any("no range on N*" in n for n in fit.notes)


def test_undetectable_min_gain_is_flagged():
    curve, base = _simulate(SIZES[:6], items=100)  # still climbing at 2,048
    fit = fit_saturation(curve, base=base, min_gain=0.001, draws=0)
    assert fit.mde is not None and fit.mde > 0.001
    assert any("under what this test can detect" in n for n in fit.notes)


def test_trim_with_curve_keeps_n_star_rows():
    rows = _rows([f"ask {i} on table t{i}" for i in range(6000)])
    curve, base = _simulate([*SIZES[:7], 6000])
    kept = rows.trim(curve, base=base)
    fit = kept.report["trim"]
    if fit.saturated:
        assert len(kept) == min(int(fit.n_saturated), 6000)
    else:
        assert len(kept) == 6000
    assert "verdict:" in str(kept)


def test_refusals():
    with pytest.raises(ValueError, match="at least 5"):
        fit_saturation({64: 0.4, 128: 0.5, 256: 0.55, 512: 0.56})
    with pytest.raises(ValueError, match="base="):
        fit_saturation({0: 0.3, 64: 0.4, 128: 0.5, 256: 0.55, 512: 0.56})
    with pytest.raises(ValueError, match="row count"):
        _rows(["a b"]).trim(0)
