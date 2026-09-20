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
COLUMNS = (
    "| Recipe | Paper | Base | Metric | Baseline -> Recipe | Verified |\n|---|---|---|---|---|---|"
)


def fail(msg: str) -> None:
    print(f"FAIL {msg}")
    sys.exit(1)


def recipe_dirs() -> list[Path]:
    return [d for d in sorted(PAPERS.iterdir()) if d.is_dir() and not d.name.startswith("_")]


def load(d: Path) -> dict:
    return json.loads((d / "results.json").read_text(encoding="utf-8"))


def seeds_per_arm(r: dict) -> int:
    """The fewest training seeds behind either trained arm."""
    seeds = r["checks"]["train_seeds"]
    return min(int(seeds["baseline"]), int(seeds["recipe"]))


def row(d: Path, r: dict) -> str:
    base, rec = r["arms"]["baseline"], r["arms"]["recipe"]
    delta = r["delta"]
    lo, hi = delta.get("ci", [0.0, 0.0])
    verified = "never run" if str(r["verified"]).startswith("1970") else r["verified"]
    paper_id = r["paper"].rstrip("/").rsplit("/", 1)[-1]
    n = seeds_per_arm(r)
    verdict = f"{delta['verdict']}, {n} seed{'s' if n != 1 else ''} per arm"
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
    for name in ("README.md", "results.json", "recipe.py"):
        if not (d / name).exists():
            fail(f"{d.name}: missing {name}")
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
    if not isinstance(seeds, dict) or {"baseline", "recipe"} - set(seeds):
        fail(
            f"{d.name}: checks.train_seeds is {{'baseline': n, 'recipe': n}}, training seeds per arm"
        )
    if any(not isinstance(seeds[arm], int) or seeds[arm] < 1 for arm in ("baseline", "recipe")):
        fail(f"{d.name}: checks.train_seeds counts are whole numbers, 1 or more")
    if seeds_per_arm(r) < MIN_TRAIN_SEEDS and verdict != "unresolved":
        fail(
            f"{d.name}: verdict {verdict} at {seeds_per_arm(r)} training seed per arm; "
            f"{UNRESOLVED_LINE} ({MIN_TRAIN_SEEDS} or more per arm, then moved or flat)"
        )
    if seeds_per_arm(r) >= MIN_TRAIN_SEEDS and verdict == "unresolved":
        fail(f"{d.name}: {seeds_per_arm(r)} seeds per arm resolve the verdict; say moved or flat")
    if verdict == "moved":
        lo, hi = r["delta"].get("ci", [0.0, 0.0])
        delta = float(r["delta"]["recipe_vs_baseline"])
        run_std = float(r["checks"]["run_std"])
        if lo <= 0.0 <= hi:
            fail(f"{d.name}: verdict moved but the interval [{lo}, {hi}] covers zero")
        # run_std is an estimate from ``runs`` re-runs, so the band carries
        # its degrees of freedom: the t quantile at runs - 1, not 1.96.
        band = noise_band(run_std, df=runs - 1)
        if abs(delta) < band:
            fail(
                f"{d.name}: verdict moved but |delta| {abs(delta):.3f} < {band:.3f} "
                f"(t(df={runs - 1})={t_quantile(runs - 1):.2f} x run_std x sqrt(1/1 + 1/1), the "
                f"re-run band on a one-run-per-side delta with run_std from {runs} re-runs)"
            )
        if r["checks"]["over_optimized"]:
            fail(f"{d.name}: verdict moved but the proxy-vs-target check says over-optimized")
    return r


def main(write: bool) -> None:
    # The band comes from whichever whileai this process imported; say which
    # (#443), on stderr so stdout stays the check's own report.
    print(provenance(), file=sys.stderr)
    dirs = recipe_dirs()
    for d in dirs:
        check_recipe(d)
        print(f"ok   {d.name}")
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
