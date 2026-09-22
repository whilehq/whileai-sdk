"""A cached row is a rollout, and a verdict is not a rollout (#737).

``recipes/papers/*/.cache/<arm>.json`` holds each arm's graded rows so
``--reuse`` can rebuild the delta without paying for the rollouts again.
The rows carry ``reward``, so the cache carried the grader's verdict too,
and nothing recorded which grader wrote it: 788c553 replaced ``MathEqual``'s
string-and-number rule with Math-Verify and ``--reuse`` went on reproducing
the old rule's numbers exactly. #737 re-graded the six cached
``zero-rl-format-reward`` arms (5,760 rollouts) and 402 verdicts moved,
every arm by 2 to 4 points.

These tests pin both directions. A cache written by grader A and read under
grader B must not come back as A's verdicts -- and a cache written by the
grader that is still in the tree must still be reused, or ``--reuse`` is
dead and every run pays for its rollouts twice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PAPERS = REPO / "recipes" / "papers"
sys.path.insert(0, str(PAPERS))


@pytest.fixture(name="cache_stamp")
def _cache_stamp():
    """The recipe tree's cache helper. Its absence is the bug, not a broken
    test, so the failure says which."""
    try:
        import cache_stamp as module
    except ImportError:
        pytest.fail(
            "recipes/papers/cache_stamp.py is missing: nothing records which grader "
            "wrote a cached verdict, so --reuse rebuilds a delta from a rule that may "
            "no longer be in the tree (#737)"
        )
    return module


ARM = {
    "after_rows": [
        {
            "prompt": "what is one half",
            "final_text": "Therefore the answer is 1/2",
            # what the retired string-and-number rule said about a correct
            # answer written outside a \boxed{}
            "reward": 0.0,
            "markers": {"strict_reward": 0.0},
            "scenario_id": "s0",
            "rollout_index": 0,
            "privileged": {"reference": "\\frac{1}{2}"},
        }
    ],
    "base_runs": [],
    "gpu_minutes": 12.0,
    "steps": 30,
}


def _flip(payload: dict) -> dict:
    """A stand-in grader that reverses every verdict, so a re-grade is
    unmistakable in the numbers rather than in a log line."""
    rows = [{**r, "reward": 1.0 - float(r.get("reward") or 0.0)} for r in payload["after_rows"]]
    return {**payload, "after_rows": rows}


def test_a_cache_from_another_grader_is_not_read_back_as_its_verdicts(
    cache_stamp, tmp_path: Path
) -> None:
    path = tmp_path / "recipe.json"
    cache_stamp.write(path, ARM, grader="string-and-number-rule")
    with pytest.raises(cache_stamp.StaleCache) as excinfo:
        cache_stamp.read(path, grader="math-verify", regrade=None)
    message = str(excinfo.value)
    assert "string-and-number-rule" in message and "math-verify" in message, message
    assert "a verdict is not a rollout" in message, message


def test_a_cache_from_another_grader_is_regraded_not_believed(cache_stamp, tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    cache_stamp.write(path, ARM, grader="string-and-number-rule")
    out, note = cache_stamp.read(path, grader="math-verify", regrade=_flip)
    assert out["after_rows"][0]["reward"] == 1.0, (
        "the stored reward came straight back: --reuse rebuilt the delta from a "
        "grader that is not in the tree (#737)"
    )
    assert note, "a re-graded reuse must say so, not pass for an ordinary one"
    # and the rollouts, the expensive part, are untouched
    assert out["after_rows"][0]["final_text"] == ARM["after_rows"][0]["final_text"]
    assert out["gpu_minutes"] == ARM["gpu_minutes"]
    # the file now carries the new stamp, so a second reuse is free
    on_disk = json.loads(path.read_text())
    assert on_disk[cache_stamp.STAMP_KEY]["grader"] == "math-verify"


def test_an_unchanged_grader_still_reuses_the_cache(cache_stamp, tmp_path: Path) -> None:
    """The negative. If every reuse re-grades, --reuse is dead and everyone
    pays for the rollouts again."""
    path = tmp_path / "recipe.json"
    cache_stamp.write(path, ARM, grader="math-verify")

    def must_not_run(payload: dict) -> dict:  # pragma: no cover - the assert is the test
        raise AssertionError("an unchanged grader must not re-grade: the cache is still good")

    out, note = cache_stamp.read(path, grader="math-verify", regrade=must_not_run)
    assert note == ""
    assert out["after_rows"][0]["reward"] == 0.0
    assert out["after_rows"] == ARM["after_rows"]


def test_a_cache_written_before_stamping_is_refused(cache_stamp, tmp_path: Path) -> None:
    """Every .cache file on disk today was written without a stamp. It is not
    known which grader decided it, so it is not trusted."""
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(ARM))
    with pytest.raises(cache_stamp.StaleCache) as excinfo:
        cache_stamp.read(path, grader="math-verify", regrade=None)
    assert "no grader stamp" in str(excinfo.value)


def test_the_whileai_version_is_part_of_the_stamp(cache_stamp, tmp_path: Path) -> None:
    """The grader's name alone is not the grader: 788c553 kept the name
    ``MathEqual`` and replaced the rule inside it. Belief 1 wants the
    versions that produced the number."""
    path = tmp_path / "recipe.json"
    cache_stamp.write(path, ARM, grader="math-verify")
    payload = json.loads(path.read_text())
    payload[cache_stamp.STAMP_KEY]["whileai"] = "0.83"
    path.write_text(json.dumps(payload))
    with pytest.raises(cache_stamp.StaleCache) as excinfo:
        cache_stamp.read(path, grader="math-verify", regrade=None)
    assert "whileai '0.83'" in str(excinfo.value)


def test_the_recipe_regrades_a_cache_the_retired_rule_wrote() -> None:
    """End to end on the recipe #737 measured: a cached arm whose verdicts
    came from the retired rule, re-graded by the one in the tree, straight
    from the rollouts on disk. No GPU, no key, no sampling."""
    recipe = _zero_rl()
    payload = {"after_rows": list(ARM["after_rows"]), "base_runs": [list(ARM["after_rows"])]}
    fresh = recipe.regrade(payload)
    row = fresh["after_rows"][0]
    assert row["reward"] == 1.0, (
        f"a correct answer written outside a box still scores {row['reward']}: the "
        "recipe reused the retired rule's verdict instead of re-grading the rollout"
    )
    # the proxy is a verdict too, and it moved with the target (#737)
    assert row["markers"]["strict_reward"] == recipe.NO_BOX_PENALTY
    assert fresh["base_runs"][0][0]["reward"] == 1.0, "the base re-runs are graded rows too"


def test_the_recipe_stamps_its_cache_and_its_results(cache_stamp) -> None:
    recipe = _zero_rl()
    stamp = cache_stamp.stamp(recipe.GRADER)
    assert stamp["grader"] == recipe.GRADER
    assert stamp["whileai"], "the whileai version belongs in the stamp"
    source = (PAPERS / "zero-rl-format-reward" / "recipe.py").read_text(encoding="utf-8")
    assert "cache_stamp.write(cached, out, grader=GRADER)" in source
    assert "json.loads(cached.read_text())" not in source, (
        "the reuse path still reads the cache straight off disk, so a stored "
        "verdict is believed whatever wrote it (#737)"
    )


def test_the_papers_contract_states_the_rule() -> None:
    contract = (PAPERS / "README.md").read_text(encoding="utf-8")
    assert "a cached row is a rollout, and a verdict is not a rollout" in contract.lower()


def test_the_shipped_results_are_stamped_and_marked_void() -> None:
    """#737's three arms are still in the tree with the retired rule's
    numbers. They must say so rather than read as measured."""
    results = json.loads(
        (PAPERS / "zero-rl-format-reward" / "results.json").read_text(encoding="utf-8")
    )
    grader = results.get("grader")
    assert isinstance(grader, dict), "results.json carries no grader stamp"
    assert grader["status"] == "void"
    assert grader["whileai"] == "0.83", "the stamp names the whileai that produced the numbers"
    assert "788c553" in grader["replaced_by"]
    assert "737" in grader["direction_measured_in"]["issue"]
    # the numbers in the file are the ones that were published, not re-graded
    # ones nobody produced
    assert results["arms"]["base"]["score"] == 0.509375
    assert results["arms"]["recipe"]["score"] == 0.7203125


def _zero_rl():
    """``zero-rl-format-reward/recipe.py`` with a stub ``modal``: the module
    body builds an image, the reward and cache functions it exposes are
    pure."""
    from example_helpers import _Chain, install_modal_stub, load_script

    modal = install_modal_stub()
    if not hasattr(modal.Image, "from_registry"):
        modal.Image.from_registry = staticmethod(lambda *a, **k: _Chain())
    return load_script("zero_rl_format_reward", PAPERS / "zero-rl-format-reward" / "recipe.py")
