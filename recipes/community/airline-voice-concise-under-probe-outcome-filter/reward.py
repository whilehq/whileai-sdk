"""The reward, and the one line the two arms disagree about.

Reward per rollout, both arms:

    outcome - LENGTH_WEIGHT * min(words / LENGTH_CAP, 1)

``outcome`` is 1 when the reply names every reservation code the caller asked
about, else 0. It is a program, not a judge. The length term is the shaping:
among replies that answer, the shorter one scores better.

The two arms differ only in which number decides that a group of rollouts is
flat and should teach nothing:

* ``score``   - the group is flat when every *shaped* score is equal. This is
  stock GRPO: TRL divides the group's advantages by their standard deviation,
  so an all-equal group contributes nothing on its own. A group of eight
  replies that all failed to answer but differ in length is *not* all-equal,
  so it survives and its length crumbs become full-size advantages.
* ``outcome`` - the group is flat when every *binary outcome* is equal, which
  is what the paper argues for and what DAPO's dynamic sampling means.

Dropping is done by masking the advantage to zero, so the batch keeps its
shape and the two arms take the same number of optimizer steps.
"""

from __future__ import annotations

LENGTH_WEIGHT = 0.30
LENGTH_CAP = 120.0  # words; the shaping saturates here
CONCISE_WORDS = 120  # the eval threshold, same number, see README


def word_count(reply: str) -> int:
    return len((reply or "").split())


def covered(reply: str, required) -> bool:
    """Every reservation code the caller asked about is named in the reply."""
    low = (reply or "").lower()
    return all(str(c).lower() in low for c in (required or []))


def outcome_of(reply: str, required) -> float:
    return 1.0 if covered(reply, required) else 0.0


def shaped_reward(reply: str, required) -> float:
    """outcome minus the length shaping."""
    o = outcome_of(reply, required)
    return o - LENGTH_WEIGHT * min(word_count(reply) / LENGTH_CAP, 1.0)


def concise_and_covered(reply: str, required, *, truncated: bool = False) -> float:
    """The target metric: answered the question, and stayed short.

    A truncated reply is not concise, it is cut off, so it does not count.
    """
    if truncated:
        return 0.0
    return 1.0 if (covered(reply, required) and word_count(reply) <= CONCISE_WORDS) else 0.0


def group_is_flat(outcomes, scores, *, metric: str) -> bool:
    """Should this group of rollouts teach nothing?

    ``metric`` is ``"score"`` (stock GRPO) or ``"outcome"`` (the paper).
    """
    if metric == "outcome":
        vals = list(outcomes)
    elif metric == "score":
        vals = list(scores)
    else:
        raise ValueError(f"filter metric must be 'score' or 'outcome', got {metric!r}")
    if not vals:
        return True
    first = vals[0]
    return all(abs(v - first) < 1e-9 for v in vals)


def messages_for(row: dict) -> list[dict]:
    return [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["prompt"]},
    ]


def selftest() -> None:
    """The reward and the filter on hand-written replies. No GPU, no key."""
    req = ["AHSPQL", "AZ46IR"]
    short_ok = "AHSPQL is economy with 2 bags. AZ46IR is basic economy, no free bags."
    long_ok = short_ok + " " + ("Additionally, please note the following details. " * 30)
    miss = "Your first reservation is economy and the other is basic economy."

    assert covered(short_ok, req) and covered(long_ok, req)
    assert not covered(miss, req)
    assert outcome_of(short_ok, req) == 1.0 and outcome_of(miss, req) == 0.0

    # Short and answered beats long and answered.
    assert shaped_reward(short_ok, req) > shaped_reward(long_ok, req)
    # Answering at any length beats not answering at a short length.
    assert shaped_reward(long_ok, req) > shaped_reward(miss, req)

    assert concise_and_covered(short_ok, req) == 1.0
    assert concise_and_covered(long_ok, req) == 0.0, "long reply is not concise"
    assert concise_and_covered(miss, req) == 0.0, "missing a code is not covered"
    assert concise_and_covered(short_ok, req, truncated=True) == 0.0, "cut off is not concise"

    # The filter. Eight replies that all miss a code but differ in length:
    # the shaped scores differ, the outcomes do not.
    outcomes = [0.0] * 8
    scores = [-0.30 * (i + 1) / 8 for i in range(8)]
    assert group_is_flat(outcomes, scores, metric="outcome") is True, "all-wrong is flat"
    assert group_is_flat(outcomes, scores, metric="score") is False, (
        "this is the phantom-advantage group: stock GRPO keeps it"
    )

    # A group that actually disagrees about the outcome survives both filters.
    mixed_o = [1.0, 0.0] * 4
    mixed_s = [0.8, -0.1] * 4
    assert group_is_flat(mixed_o, mixed_s, metric="outcome") is False
    assert group_is_flat(mixed_o, mixed_s, metric="score") is False

    # A saturated group: every rollout answered, lengths differ.
    sat_o = [1.0] * 8
    sat_s = [1.0 - 0.30 * (i + 1) / 16 for i in range(8)]
    assert group_is_flat(sat_o, sat_s, metric="outcome") is True, (
        "outcome filter drops it: this is the production case the paper never hits"
    )
    assert group_is_flat(sat_o, sat_s, metric="score") is False, (
        "stock GRPO keeps it, and its length spread is the only signal for conciseness"
    )

    try:
        group_is_flat([1.0], [1.0], metric="shaped")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("an unknown filter metric must raise")

    print("reward selftest: ok")


if __name__ == "__main__":
    selftest()
