"""Offline check for skills/write-a-recipe/SKILL.md: a finished run becomes a recipe a
maintainer will merge.

The setup writes two temporary recipes: ``RECIPE`` follows the rules (a README in the
house shape, smoke.sh, results.json with intervals, a 12-row fixture named in the
README, a frozen test pinned by content hash) and ``BAD`` breaks four of them (no
smoke.sh, an arm with no interval, 3,000 rows in git, a key in a script). ``REAL`` is a
merged recipe from the repo, read only. Every block in SKILL.md is below, verbatim:
pin the test and check it, audit both folders and the real one, then post
``RESULTS`` to a recording fake platform and print the verdict. No key, no network,
no GPU.

    uv run python skills/write-a-recipe/check.py
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
import time
from pathlib import Path

from whileai.platform import Behavior, track

T0 = time.monotonic()
TIME_LIMIT = 60  # seconds; the bar every skill check meets (skills/BRIEF.md)

REPO = Path(__file__).resolve().parents[2]
REAL = (
    REPO / "recipes" / "04-train" / "parsebench"
)  # the newest merged recipe; older ones predate the shape
TMP = Path(tempfile.mkdtemp(prefix="write-a-recipe-"))
RECIPE = TMP / "good"
BAD = TMP / "bad"

# ---------------------------------------------------------------- the two fixtures

ROWS = [{"text": f"row {i}", "label": i % 2, "slice": "fixture"} for i in range(12)]
README_GOOD = """# Refund policy classifier

Train a 22M encoder to flag a refund request, and show which data taught it. On the
frozen test the recipe scores 71 +/- 4 points against the baseline's 52 +/- 5 (n=200,
three seeds, spread 1.8 points).

What you will learn: a frozen test pinned by content, a matched control, a number
with its interval. Needs nothing offline; one L40S for the live path. Takes a minute
offline, ten live.

## Run it

```bash
sh smoke.sh                     # offline, no key
python run.py --dry-run
python fetch_tests.py           # the frozen test from Hugging Face, checked against test.sha256
```

`fixture.jsonl` is the 12-row fixture the offline path runs on.

## Result

| Arm | points | 95% | seeds |
|---|---|---|---|
| baseline | 52 | +/- 5 | 3 |
| recipe | 71 | +/- 4 | 3 |

## Honest limits

The benign side is synthetic.

## References

1. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Evaluation*.
"""
README_BAD = """# A recipe

Scores 0.81.

## Run it

```bash
python run.py
```
"""


def _write_recipe(
    folder: Path, readme: str, *, smoke: bool, ci: bool, rows: int, key: bool
) -> None:
    folder.mkdir(parents=True)
    (folder / "README.md").write_text(readme, encoding="utf-8")
    (folder / "run.py").write_text(
        ("API_KEY = 'sk-ant-" + "a" * 40 + "'\n" if key else "") + "print('ok')\n", encoding="utf-8"
    )
    if smoke:
        (folder / "smoke.sh").write_text(
            "#!/bin/sh\nset -eu\npython run.py --dry-run\n", encoding="utf-8"
        )
    arms = {"baseline": {"score": 52, "ci_half": 5}, "recipe": {"score": 71, "ci_half": 4}}
    if not ci:
        arms = {"baseline": {"score": 52}, "recipe": {"score": 71}}
    (folder / "results.json").write_text(
        json.dumps({"recipe": folder.name, "arms": arms}), encoding="utf-8"
    )
    name = "fixture.jsonl" if rows <= 12 else "train.jsonl"
    with open(folder / name, "w", encoding="utf-8") as fh:
        for i in range(rows):
            fh.write(json.dumps({"text": f"row {i}", "label": i % 2}) + "\n")


_write_recipe(RECIPE, README_GOOD, smoke=True, ci=True, rows=12, key=False)
_write_recipe(BAD, README_BAD, smoke=False, ci=False, rows=3000, key=True)

# ---------------------------------------------------------------- 2. the pin


def pin_rows(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


def check_pin(rows_path: Path, pin_path: Path) -> None:
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line]
    digest = pin_rows(rows)
    pinned = pin_path.read_text(encoding="utf-8").strip()
    if digest != pinned:
        raise SystemExit(
            f"{rows_path.name}: sha256 {digest[:12]} does not match the pin {pinned[:12]}"
        )


PIN = pin_rows(ROWS)
(RECIPE / "test.sha256").write_text(PIN + "\n", encoding="utf-8")
(RECIPE / "out").mkdir()
with open(RECIPE / "out" / "test.jsonl", "w", encoding="utf-8") as fh:
    for r in ROWS:
        fh.write(json.dumps(r) + "\n")
check_pin(RECIPE / "out" / "test.jsonl", RECIPE / "test.sha256")
# a reformat of the file does not move the pin; a changed row does
with open(RECIPE / "out" / "test.jsonl", "w", encoding="utf-8") as fh:
    for r in ROWS:
        fh.write(json.dumps(r, indent=None, separators=(",", ": ")) + "\n")
check_pin(RECIPE / "out" / "test.jsonl", RECIPE / "test.sha256")
changed = [*ROWS[:-1], {**ROWS[-1], "label": 1 - ROWS[-1]["label"]}]
assert pin_rows(changed) != PIN, "a changed row must move the pin"
print(f"pin: t-{PIN[:8]}, survives a reformat, moves on a changed row")

# ---------------------------------------------------------------- 4. the audit

MAX_FIXTURE_ROWS = 500
KEY = re.compile(r"(sk-ant-[A-Za-z0-9\-]{20,}|hf_[A-Za-z0-9]{30,}|zp_[A-Za-z0-9]{20,})")
SECTIONS = ("## Run it", "## References")


def audit(recipe: Path) -> list[str]:
    findings: list[str] = []
    readme = recipe / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    if not readme.exists():
        findings.append("no README.md")
    for section in SECTIONS:
        if section not in text:
            findings.append(f"README has no '{section}' section")
    if "What you will learn" not in text and "what you learn" not in text.lower():
        findings.append("README does not say what you will learn")
    if not (recipe / "smoke.sh").exists():
        findings.append("no smoke.sh (the offline path CI runs on every pull request)")
    results = recipe / "results.json"
    if results.exists():
        arms = json.loads(results.read_text(encoding="utf-8")).get("arms", {})
        for name, arm in arms.items():
            has_ci = any("ci" in k.rsplit(".", 1)[-1] for k in _flatten(arm))
            if not has_ci:
                findings.append(f"results.json arm '{name}' has a number with no interval")
    for path in sorted(recipe.rglob("*.jsonl")):
        if "out" in path.relative_to(recipe).parts:
            continue
        n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if n > MAX_FIXTURE_ROWS:
            findings.append(
                f"{path.name}: {n} rows in git; a fixture is under {MAX_FIXTURE_ROWS}, "
                "the set goes to Hugging Face with a pin"
            )
        elif path.name not in text and path.stem not in text:
            findings.append(f"{path.name}: checked-in rows the README does not name")
    for pin in recipe.glob("*.sha256"):
        if pin.stem not in text:
            findings.append(f"{pin.name}: a pin the README does not explain")
    for py in recipe.rglob("*.py"):
        if KEY.search(py.read_text(encoding="utf-8")):
            findings.append(f"{py.name}: looks like a key")
    return findings


def _flatten(obj, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(f"{prefix}{k}")
            keys += _flatten(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for v in obj:
            keys += _flatten(v, prefix)
    return keys


good = audit(RECIPE)
assert good == [], f"the recipe that follows the rules has findings: {good}"
bad = audit(BAD)
print("audit on BAD:")
for f in bad:
    print("  -", f)
assert any("smoke.sh" in f for f in bad), bad
assert any("no interval" in f for f in bad), bad
assert any("rows in git" in f for f in bad), bad
assert any("looks like a key" in f for f in bad), bad
assert any("References" in f for f in bad), bad
real = audit(REAL)
assert real == [], f"{REAL.name} has findings: {real}"
print(f"audit: good 0 findings, BAD {len(bad)} findings, {REAL.name} 0 findings")

# ---------------------------------------------------------------- 6. post and verdict

RESULTS = {
    "recipe": "refund-classifier",
    "base_model": "nreimers/MiniLM-L6-H384-uncased",
    "metric": "refund_request_flagged",
    "test_version": "t-" + PIN[:8],
    "n_holdout": 200,
    "checks": {"noise_band_points": 4.4, "decontaminated_dropped": 0},
    "arms": {
        "baseline": {"method": "eval", "score": 52.0, "ci_half": 5.0, "note": "Served baseline."},
        "recipe": {
            "method": "classify",
            "score": 71.0,
            "ci_half": 4.0,
            "note": "Changed: matched twins.\nMoved: 52 to 71.\nWhy: the twins take the carrier out of the label.\nLearned: negatives are the whole game.\nReproduce: seeds 1,2,3, test t-"
            + PIN[:8],
        },
    },
}
CALLS: list[tuple[str, str]] = []


def fake(method: str, path: str, body=None):
    """The offline transport: record the call, answer like the API."""
    CALLS.append((method, path))
    if path == "/runs":
        return {"id": f"{body['agent']}-{body['version']}", "version": body["version"]}
    if path.endswith("/evals"):
        return {"evals": body}
    if "/dashboard" in path:
        arms = RESULTS["arms"]
        delta = arms["recipe"]["score"] - arms["baseline"]["score"]
        spread = math.sqrt(arms["recipe"]["ci_half"] ** 2 + arms["baseline"]["ci_half"] ** 2)
        return {
            "agent": {"id": RESULTS["recipe"], "name": RESULTS["recipe"]},
            "behavior": {
                "name": RESULTS["metric"],
                "n": RESULTS["n_holdout"],
                "noiseFloor": RESULTS["checks"]["noise_band_points"],
                "rewardIsJudge": False,
                "contamination": 0,
            },
            "verdict": {
                "candidate": "recipe",
                "serving": "baseline",
                "delta": delta,
                "excludesZero": abs(delta) > spread,
                "regressions": 0,
            },
        }
    if not isinstance(body, dict):
        return {"ok": True}
    return {"id": body.get("id", "")}


def post(results: dict, transport) -> str:
    tracked = track(results["recipe"], model=results["base_model"], transport=transport)
    tracked.behavior(
        Behavior(
            name=results["metric"],
            test_version=results["test_version"],
            n=results["n_holdout"],
            noise_floor=results["checks"]["noise_band_points"],
            contamination=results["checks"]["decontaminated_dropped"],
            graded_by="program",
        )
    )
    for name, arm in results["arms"].items():
        run = tracked.run(name, method=arm["method"], targets=[results["metric"]])
        run.score(results["metric"], arm["score"], ci=arm["ci_half"], n=results["n_holdout"])
        run.note(arm["note"])
        run.finish(say=False)
    return str(tracked.verdict())


verdict = post(RESULTS, fake)
print(verdict)
assert "recipe" in verdict and "baseline" in verdict, verdict
assert any(p == "/runs" for _, p in CALLS) and any(p.endswith("/evals") for _, p in CALLS), CALLS
shutil.rmtree(TMP, ignore_errors=True)
elapsed = time.monotonic() - T0
assert elapsed < TIME_LIMIT, f"check took {elapsed:.0f}s; the bar is {TIME_LIMIT}s"
print(f"write-a-recipe: every step ran offline in {elapsed:.1f}s")
