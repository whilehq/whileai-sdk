"""Case study (gentlyventures.com/casestudies/whileai, whileai 0.126): a
holdout size read off a small pilot is right on average but noisy (the
same goal gave 10 to 52 tasks), and decontamination misses reworded
copies without saying so at run time. Fixed seeds, no network."""

from __future__ import annotations

import random
import warnings

from whileai.simulations.score import stats
from whileai.simulations.score.stats import decontaminate, holdout_size

EFFECT = 0.1
K = 4
PILOT_TASKS = 12
PILOTS = 40


def _pilot(n_tasks, rng):
    """Before and after arms on the same tasks from one fixed distribution:
    per-task base rate uniform on [0.2, 0.8]; a third of tasks gain 0.3,
    the rest do not move. Bernoulli draws, k rollouts per side."""
    before, after = [], []
    for t in range(n_tasks):
        p = rng.uniform(0.2, 0.8)
        q = min(1.0, p + 0.3) if rng.random() < 1 / 3 else p
        for arm, rate in ((before, p), (after, q)):
            for i in range(K):
                arm.append(
                    {
                        "scenario_id": f"t{t}",
                        "rollout_index": i,
                        "prompt": f"p{t}",
                        "reward": 1 if rng.random() < rate else 0,
                    }
                )
    return before, after


def _true_n():
    """The answer a very large pilot from the same distribution gives."""
    before, after = _pilot(5000, random.Random(12345))
    return holdout_size(EFFECT, before=before, after=after)["n_tasks"]


def _small_pilots():
    return [
        holdout_size(EFFECT, before=b, after=a)
        for b, a in (_pilot(PILOT_TASKS, random.Random(seed)) for seed in range(PILOTS))
    ]


def test_a_small_pilot_gives_a_noisy_point_estimate():
    """The reproduction: same goal, same distribution, very different n."""
    ns = [r["n_tasks"] for r in _small_pilots()]
    assert max(ns) > 2 * min(ns)


def test_the_report_carries_a_range_that_covers_the_true_n():
    truth = _true_n()
    reports = _small_pilots()
    covered = 0
    for r in reports:
        assert r["n_tasks_low"] is not None and r["n_tasks_high"] is not None
        assert r["n_tasks_low"] <= r["n_tasks"] <= r["n_tasks_high"]
        assert "bootstrap" in r["n_tasks_range_method"]
        covered += r["n_tasks_low"] <= truth <= r["n_tasks_high"]
    assert covered / len(reports) >= 0.7, (covered, truth)  # 29 of 40 on these seeds
    assert "range" in str(reports[0])


def test_the_range_is_deterministic_and_absent_where_nothing_was_measured():
    b, a = _pilot(PILOT_TASKS, random.Random(3))
    one, two = holdout_size(EFFECT, before=b, after=a), holdout_size(EFFECT, before=b, after=a)
    assert (one["n_tasks_low"], one["n_tasks_high"]) == (two["n_tasks_low"], two["n_tasks_high"])
    # before= alone: the range comes from resampling the base rate
    alone = holdout_size(EFFECT, before=b)
    assert alone["n_tasks_low"] <= alone["n_tasks"] <= alone["n_tasks_high"]
    # nothing measured: no range to report
    for r in (holdout_size(EFFECT), holdout_size(EFFECT, task_std=0.3)):
        assert r["n_tasks_low"] is None and r["n_tasks_high"] is None
        assert r["n_tasks_range_method"] is None


# ------------------------------------------------------------------ decontamination

EVAL = [{"prompt": "What is the refund window for an unopened blender bought in March?"}]
TRAIN = [
    {"prompt": "Where is order 4473?"},
    # a reworded copy of the eval question: no 8-gram in common
    {"prompt": "If I purchased a blender in March and never opened it, how long can I return it?"},
]


def _fake_embedder(texts):
    """Blender questions point one way, everything else another."""
    return [[1.0, 0.0] if "blender" in t.lower() else [0.0, 1.0] for t in texts]


def _decon(**kw):
    stats._DECONTAM_PARAPHRASE_WARNED = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clean, report = decontaminate(TRAIN, against=[EVAL], **kw)
    return clean, report, [w for w in caught if "reworded" in str(w.message)]


def test_a_reworded_copy_slips_through_the_text_rules():
    clean, report, _ = _decon()
    assert report["n_contaminated"] == 0 and len(clean) == 2


def test_without_an_embedder_a_warning_names_the_option():
    _, report, caught = _decon()
    assert len(caught) == 1
    assert "embedder=" in str(caught[0].message)
    assert any("reworded" in n and "embedder=" in n for n in report["notes"])


def test_the_warning_fires_once_per_process():
    _decon()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decontaminate(TRAIN, against=[EVAL])
    assert not [w for w in caught if "reworded" in str(w.message)]


def test_with_an_embedder_the_reworded_copy_is_caught_and_no_warning():
    clean, report, caught = _decon(embedder=_fake_embedder)
    assert report["n_semantic"] == 1 and len(clean) == 1
    assert not caught
