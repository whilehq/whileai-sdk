"""Every paper recipe has the README sections, a results.json with the right
keys, and a row in the table in this folder's README, which is generated
from the results files so parallel authors never edit the same lines.

    python recipes/papers/check.py          # verify; exit 1 on the first miss
    python recipes/papers/check.py --write  # regenerate the table, then verify

The band and the seed rule are the package's own (``whileai.simulations``):
this script imports them from the checkout it lives in, so the number a
recipe is held to is the number ``compare()`` prints, never a copy.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PAPERS = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPERS.parents[1]))

from whileai.config import provenance
from whileai.simulations.defaults import MIN_TRAIN_SEEDS
from whileai.simulations.score.delta import UNRESOLVED_LINE
from whileai.simulations.score.stats import _t_quantile as t_quantile
from whileai.simulations.score.stats import noise_band

INDEX = PAPERS / "README.md"
START, END = "<!-- table:start -->", "<!-- table:end -->"
SECTIONS = ["## Recipe", "## Run", "## Result", "## Checks", "## Climb", "## Learned"]
HEADER = ["**Paper:**", "**Book:**", "**Claim:**", "**The change:**"]
KEYS = {
    "recipe",
    "paper",
    "base_model",
    "metric",
    "n_holdout",
    "k",
    "arms",
    "delta",
    "book",
    "checks",
    "gpu",
    "usd",
    "verified",
    "whileai",
}
ARM_KEYS = {"score", "ci", "steps"}
CHECK_KEYS = {
    "run_std",
    "run_std_runs",
    "train_seeds",
    "decontaminated_dropped",
    "over_optimized",
    "length_before",
    "length_after",
    "hack_scan_top",
    "seed",
}
#: The verdict vocabulary of a paper recipe. ``moved`` and ``flat`` need
#: ``MIN_TRAIN_SEEDS`` training seeds on both trained arms; at one seed per
#: arm the only word is ``unresolved`` (#356).
VERDICTS = ("moved", "flat", "unresolved")
# MIN_BASE_RERUNS = 3: CONSTITUTION.md's row for repeatable science reads
# "refuses 'moved' without an interval that excludes zero, three base re-runs,
# a clean holdout, and a proxy-vs-target verdict". Two re-runs do give a
# standard deviation, but at one degree of freedom, where the t quantile is
# 12.71 and the band it makes is wider than any result this directory has
# reported; three is the smallest count the row allows (Lambert 2025, chapter
# Evaluation, appendix C). The structural floor below stays 2, so a recipe that
# is not claiming "moved" still records a run_std that means something.
MIN_BASE_RERUNS = 3
#: The entry point of a recipe: a two-arm training replication runs
#: ``recipe.py``, a step-shaped one (``meta-harness``) runs ``run.py``.
ENTRY_POINTS = ("recipe.py", "run.py")
COLUMNS = (
    "| Recipe | Paper | Base | Metric | Baseline -> Recipe | Verified |\n|---|---|---|---|---|---|"
)


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    sys.exit(1)


def recipe_dirs() -> list[Path]:
    """Every recipe that claims a number: one ``results.json``, one table row,
    and the gates in ``check_recipe`` below.

    The filter is the claim, not the shape. A recipe with no ``results.json``
    yet (``harness-and-weights`` before its full live run) claims nothing, so
    there is no row; the day it writes one it is checked like the rest. Until
    #809 the filter was the shape instead -- any directory with a ``run.py``
    was skipped -- which left ``meta-harness``, the recipe carrying the
    harness-optimization headline, as the one recipe the science gate never
    saw."""
    return [
        d
        for d in sorted(PAPERS.iterdir())
        if d.is_dir() and not d.name.startswith("_") and (d / "results.json").exists()
    ]


def load(d: Path) -> dict:
    return json.loads((d / "results.json").read_text(encoding="utf-8"))


def seeds_per_arm(r: dict) -> int | None:
    """The fewest training seeds behind either trained arm, or ``None`` when
    the recipe has no trained arm at all (``train_seeds: null``: a search over
    harness code trains nothing, so the seed rule has nothing to count)."""
    seeds = r["checks"]["train_seeds"]
    if seeds is None:
        return None
    return min(int(seeds["baseline"]), int(seeds["recipe"]))


def arms_read(r: dict) -> str:
    """How many training seeds stand behind the delta, in words."""
    n = seeds_per_arm(r)
    if n is None:
        return "no trained arm"
    return f"{n} seed{'s' if n != 1 else ''} per arm"


def row(d: Path, r: dict) -> str:
    base, rec = r["arms"]["baseline"], r["arms"]["recipe"]
    delta = r["delta"]
    lo, hi = delta.get("ci", [0.0, 0.0])
    verified = "never run" if str(r["verified"]).startswith("1970") else r["verified"]
    paper_id = r["paper"].rstrip("/").rsplit("/", 1)[-1]
    verdict = f"{delta['verdict']}, {arms_read(r)}"
    return (
        f"| [{d.name}]({d.name}) | [{paper_id}]({r['paper']}) | {r['base_model']} "
        f"| {r['metric']} | {base['score']:.2f} -> {rec['score']:.2f} "
        f"({delta['recipe_vs_baseline']:+.2f} [{lo:+.2f}, {hi:+.2f}], {verdict}) "
        f"| {verified} |"
    )


def table(dirs: list[Path]) -> str:
    rows = [row(d, load(d)) for d in dirs] or ["| _none yet_ | | | | | |"]
    return "\n".join([START, COLUMNS, *rows, END])


def check_recipe(d: Path) -> dict:
    for name in ("README.md", "results.json"):
        if not (d / name).exists():
            fail(f"{d.name}: missing {name}")
    if not any((d / name).exists() for name in ENTRY_POINTS):
        fail(
            f"{d.name}: missing an entry point; a two-arm training replication is recipe.py, "
            f"a step-shaped recipe is run.py ({', '.join(ENTRY_POINTS)})"
        )
    text = (d / "README.md").read_text(encoding="utf-8")
    for s in HEADER + SECTIONS:
        if s not in text:
            fail(f"{d.name}: README missing '{s}'")
    if "<" in text.splitlines()[0]:
        fail(f"{d.name}: README title still has a placeholder")
    r = load(d)
    missing = KEYS - set(r)
    if missing:
        fail(f"{d.name}: results.json missing {sorted(missing)}")
    if r["recipe"] != d.name:
        fail(f"{d.name}: results.json recipe is '{r['recipe']}'")
    for arm in ("base", "baseline", "recipe"):
        if arm not in r["arms"]:
            fail(f"{d.name}: results.json arms missing '{arm}'")
        if ARM_KEYS - set(r["arms"][arm]):
            fail(f"{d.name}: arm '{arm}' missing {sorted(ARM_KEYS - set(r['arms'][arm]))}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(r["verified"])):
        fail(f"{d.name}: verified must be YYYY-MM-DD")
    verdict = r["delta"].get("verdict")
    if verdict not in VERDICTS:
        fail(f"{d.name}: delta.verdict must be one of {', '.join(VERDICTS)}")
    if not re.fullmatch(r"[A-Z][A-Za-z ,-]+", str(r["book"])):
        fail(
            f"{d.name}: book must be a chapter title of Lambert 2025, like 'Reinforcement Learning'"
        )
    if CHECK_KEYS - set(r["checks"]):
        fail(f"{d.name}: checks missing {sorted(CHECK_KEYS - set(r['checks']))}")
    # The science bar: "moved" needs an interval that excludes zero AND a delta
    # larger than the eval's own re-run band (Lambert 2025, chapter Evaluation, appendix C), and no
    # over-optimization verdict (chapter Over-optimization). Otherwise it is "flat". The band is
    # noise_band(run_std, df=run_std_runs - 1), the same number as whileai's
    # eval_variance noise_band and delta_report(run_std=, run_std_runs=) within_noise test.
    runs = r["checks"]["run_std_runs"]
    if not isinstance(runs, int) or runs < 2:  # two runs before a standard deviation exists
        fail(f"{d.name}: checks.run_std_runs is the number of base re-runs behind run_std, 2+")
    # The claim is a delta between two separately trained models, and the
    # band above measures only the eval (#356): "moved" and "flat" need
    # MIN_TRAIN_SEEDS training seeds on both arms; one seed per arm is
    # "unresolved", whatever the interval says.
    seeds = r["checks"]["train_seeds"]
    # ``train_seeds: null`` says the recipe has no trained arm: ``meta-harness``
    # searches over harness code and trains nothing, so the seed rule has
    # nothing to count and the science half below is the whole bar (#809). It
    # is a stated "no trained arm", never a missing measurement: a recipe that
    # trained something and did not count its seeds writes the dict.
    if seeds is not None:
        if not isinstance(seeds, dict) or {"baseline", "recipe"} - set(seeds):
            fail(
                f"{d.name}: checks.train_seeds is {{'baseline': n, 'recipe': n}}, training seeds "
                "per arm, or null when the recipe has no trained arm"
            )
        if any(not isinstance(seeds[arm], int) or seeds[arm] < 1 for arm in ("baseline", "recipe")):
            fail(f"{d.name}: checks.train_seeds counts are whole numbers, 1 or more")
        per_arm = seeds_per_arm(r)
        assert per_arm is not None
        if per_arm < MIN_TRAIN_SEEDS and verdict != "unresolved":
            fail(
                f"{d.name}: verdict {verdict} at {per_arm} training seed per arm; "
                f"{UNRESOLVED_LINE} ({MIN_TRAIN_SEEDS} or more per arm, then moved or flat)"
            )
        if per_arm >= MIN_TRAIN_SEEDS and verdict == "unresolved":
            fail(f"{d.name}: {per_arm} seeds per arm resolve the verdict; say moved or flat")
    if verdict == "moved":
        # The four criteria of CONSTITUTION.md's repeatable-science row, in its
        # order: an interval that excludes zero, three base re-runs, a clean
        # holdout, and a proxy-vs-target verdict. Each one refuses "moved" and
        # names the fix; each one is driven red by tests/recipes/
        # test_papers_science_gate.py, because until #809 no recipe had ever
        # reached this branch and a gate that has never fired is a gate nobody
        # has shown to work.
        lo, hi = r["delta"].get("ci", [0.0, 0.0])
        delta = float(r["delta"]["recipe_vs_baseline"])
        run_std = float(r["checks"]["run_std"])
        if lo <= 0.0 <= hi:
            fail(
                f"{d.name}: verdict moved but the interval [{lo}, {hi}] covers zero; "
                "say flat, or add data until it does not"
            )
        if runs < MIN_BASE_RERUNS:
            fail(
                f"{d.name}: verdict moved on a run_std from {runs} base re-run(s); the bar is "
                f"{MIN_BASE_RERUNS} (CONSTITUTION.md, repeatable science). Re-run the base arm on "
                f"the same holdout until checks.run_std_runs is {MIN_BASE_RERUNS} or more, or "
                "say flat"
            )
        # run_std is an estimate from ``runs`` re-runs, so the band carries
        # its degrees of freedom: the t quantile at runs - 1, not 1.96.
        band = noise_band(run_std, df=runs - 1)
        if abs(delta) < band:
            fail(
                f"{d.name}: verdict moved but |delta| {abs(delta):.3f} < {band:.3f} "
                f"(t(df={runs - 1})={t_quantile(runs - 1):.2f} x run_std x sqrt(1/1 + 1/1), the "
                f"re-run band on a one-run-per-side delta with run_std from {runs} re-runs); "
                "say flat"
            )
        dropped = r["checks"]["decontaminated_dropped"]
        if isinstance(dropped, bool) or not isinstance(dropped, int) or dropped < 0:
            fail(
                f"{d.name}: verdict moved with checks.decontaminated_dropped {dropped!r}, so the "
                "holdout is not known clean. It is the count of train rows that overlapped the "
                "holdout and were dropped, 0 when the two splits cannot overlap: run "
                "wai.decontaminate(train, holdout) and record len(dropped), or say flat"
            )
        over = r["checks"]["over_optimized"]
        if not isinstance(over, bool):
            fail(
                f"{d.name}: verdict moved with checks.over_optimized {over!r}, so there is no "
                "proxy-vs-target verdict. Score the proxy the recipe optimised and the target it "
                "claims on the same holdout and record true or false (Lambert 2025, chapter "
                "Over-Optimization), or say flat"
            )
        if over:
            fail(f"{d.name}: verdict moved but the proxy-vs-target check says over-optimized")
    return r


def skipped_note(r: dict) -> str:
    """Which gates this recipe did not reach, and why.

    A recipe that skipped its checks must not print the same line as one that
    passed them. Every recipe here is unresolved -- the nine trained ones at
    one training seed per arm, ``meta-harness`` for want of a proxy-vs-target
    verdict -- so no recipe has yet reached the interval and band gates in
    ``check_recipe``. What has reached them is
    ``tests/recipes/test_papers_science_gate.py``, which drives each one red
    (#809).
    """
    notes = []
    if r["delta"].get("verdict") != "moved":
        seeds = seeds_per_arm(r)
        stands = "with no trained arm" if seeds is None else f"at {seeds} training seed(s) per arm"
        notes.append(
            f"verdict {r['delta'].get('verdict')} {stands}"
            ": interval, noise band and proxy check not enforced"
        )
    # What the recipe does not measure, named. A criterion recorded as null is
    # a criterion nobody ran, and "moved" is not available to a recipe that has
    # one (#809); saying which one is how the next round knows what to add.
    absent = [
        name
        for name, key in (
            ("a proxy-vs-target verdict", "over_optimized"),
            ("a holdout decontamination count", "decontaminated_dropped"),
        )
        if key in r["checks"] and r["checks"][key] is None
    ]
    if absent:
        notes.append(f"not measured here: {', '.join(absent)}; moved is not available")
    # Reported at every verdict, enforced only on a claimed result. A recipe
    # that publishes its own over-optimization is behaving correctly and must
    # not fail for it; a recipe that hides it behind an unresolved verdict was
    # invisible until now (Lambert 2025, chapter Over-Optimization).
    if r["checks"].get("over_optimized"):
        notes.append("proxy-vs-target says OVER-OPTIMIZED")
    return f"   ({'; '.join(notes)})" if notes else ""


def main(write: bool) -> None:
    # The band comes from whichever whileai this process imported; say which
    # (#443), on stderr so stdout stays the check's own report.
    print(provenance(), file=sys.stderr)
    dirs = recipe_dirs()
    for d in dirs:
        r = check_recipe(d)
        print(f"ok   {d.name}{skipped_note(r)}")
    index = INDEX.read_text(encoding="utf-8")
    if START not in index or END not in index:
        fail(f"{INDEX.name} needs the {START} and {END} markers")
    fresh = table(dirs)
    pre, rest = index.split(START, 1)
    _, post = rest.split(END, 1)
    new_index = pre + fresh + post
    if write and new_index != index:
        INDEX.write_text(new_index, encoding="utf-8")
        print("wrote the table")
    elif new_index != index:
        fail(f"table in {INDEX.name} is stale; run: python recipes/papers/check.py --write")
    print(f"ok   {len(dirs)} paper recipe(s)")


if __name__ == "__main__":
    main(write="--write" in sys.argv[1:])
