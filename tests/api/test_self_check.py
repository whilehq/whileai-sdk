"""``wai.self_check``: the statistics checked by simulation, reproducibly.

The fast run (40 trials per check, seed 0) is the one a user gets from
``wai self-check``. Every measured rate has to land within three Monte
Carlo errors of its target, and the same seed has to print the same
report, through the Python call and through the command.
"""

from __future__ import annotations

import pytest

import whileai as wai
from whileai.cli import main
from whileai.simulations.defaults import SELF_CHECK_SEED, SELF_CHECK_TRIALS

# The fast run takes about 13 s. Two tests read it, so under xdist the
# suite costs two fast runs on one worker at most (the second is the
# command's own run).


@pytest.fixture(scope="module")
def fast() -> dict:
    return wai.self_check()


def test_every_check_lands_on_its_target(fast) -> None:
    assert fast["trials"] == SELF_CHECK_TRIALS
    assert fast["seed"] == SELF_CHECK_SEED
    assert len(fast["checks"]) == 8
    for c in fast["checks"]:
        assert c["trials"] == SELF_CHECK_TRIALS
        assert c["mc_error"] > 0, c
        # two-sided, tighter than the report's own one-sided flag
        assert abs(c["measured"] - c["target"]) <= 3 * c["mc_error"], c
        assert c["ok"], c
    assert fast["ok"] is True
    assert fast["same_path"] is True

    # the targets are the named ones
    by = {c["check"]: c for c in fast["checks"]}
    assert by["false_alarm"]["target"] == pytest.approx(0.05)
    assert by["near_duplicates"]["target"] == pytest.approx(0.05)
    assert by["coverage_10"]["target"] == pytest.approx(0.95)
    assert by["coverage_20"]["target"] == pytest.approx(0.95)
    # the power holdout_size sizes with, one try a side, base 0.6
    assert by["power_30_10"]["target"] == pytest.approx(0.126, abs=0.001)
    assert by["power_60_16"]["target"] == pytest.approx(0.479, abs=0.001)

    # the report prints itself
    text = str(fast)
    assert text.splitlines()[0].startswith("self-check:")
    assert "MC error" in text
    assert repr(fast) == f"SelfCheckReport(ok=True, trials={SELF_CHECK_TRIALS}, seed=0)"
    assert fast._repr_html_().startswith("<pre>")


def test_same_seed_prints_the_same_report(fast, capsys) -> None:
    """The command, run again from scratch, prints the Python call's report byte for byte."""
    code = main(["self-check"])
    out = capsys.readouterr().out
    assert code == 0
    assert out == str(fast) + "\n"
    assert f"seed {SELF_CHECK_SEED}, {SELF_CHECK_TRIALS} trials per check" in out
    assert "all 8 checks on target." in out


@pytest.mark.parametrize(
    ("kwargs", "word"),
    [({"trials": 0}, "trials"), ({"alpha": 1.0}, "alpha"), ({"flag_at": 0}, "flag_at")],
)
def test_bad_values_name_the_argument(kwargs, word) -> None:
    with pytest.raises(ValueError, match=word):
        wai.self_check(**kwargs)
