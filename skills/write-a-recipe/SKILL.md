---
name: write-a-recipe
description: >
  How a coding agent turns a finished run into a recipe under recipes/ that
  a maintainer will merge: what goes in git and what goes to Hugging Face,
  the frozen test pinned by a content hash, the README in the house shape
  with every number carrying its interval and n, smoke.sh that runs offline
  in under a minute, results.json, the platform post, the gates, and the
  pull request. Use before opening a recipe PR, when a recipe has grown past
  a few thousand lines, or when a reviewer asks what all the files are. No
  GPU, no key.
metadata:
  version: "1.0.0"
---

# Write a recipe

A recipe is one run a reader can redo. The merged ones are 5 to 33 files
and 800 to 4,800 lines; rows in git are a fixture the README names, never
the dataset. Every step below is in
`check.py`, which runs in under a minute with no key; its setup defines
`RECIPE` (a temporary recipe that follows the rules), `BAD` (one that
breaks four of them, so the audit is shown to go red), `REAL` (a merged
recipe from the repo, read only), `RESULTS` (a results.json in the house
shape) and `fake` (a platform transport that records calls and answers
like the API).

## 1. Decide what lives in git

Code, the README, `results.json`, `smoke.sh`, the pins, and a fixture of at
most a few hundred rows named in the README. The rows (train, validation,
the full frozen tests), the weights and the scores go to a Hugging Face repo
under the org and to the gitignored `out/`; `fetch_tests.py` pulls them
back. Under 5,000 lines in the pull request, and a JSONL in git only when
it is the fixture the offline path runs on.

## 2. Freeze the test by content, and commit the pin

The test is frozen the moment its hash is written, before any training.
The pin is a sha256 over the rows in order (not the file, so a reformat
does not change it); it is the file that stays in git when the rows leave.
Every scorer checks the rows against the pin before it reads a number
(Lambert 2025, "Evaluation": held-out sets are kept apart and named).

```python
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
```

Name the test on the platform by that pin (`Behavior(test_version="t-"
+ pin[:8])`), and bump the name when the rows change.

## 3. The README, in the house shape

Open with what the recipe does and the number it moved, with its interval
and n, in the first paragraph; then **what you will learn**, **needs**,
**takes**. `## Run it` comes next and its first command is the free one
(`sh smoke.sh`, `python run.py --dry-run`), the paid one and its cost
after it, then a flags table. The result table carries every arm with a
95% interval and the seed count; a flat or negative result is a row, not a
footnote. `## Honest limits` says what the numbers do not show. References
are numbered, every one cited, the textbook by chapter title. Under about
250 lines; a climb of many rounds is a table, not a section per round.

## 4. Audit the folder before the pull request

The audit is what a reviewer looks for first; an empty list is a recipe in
shape. It goes red on `BAD` (no smoke.sh, a mean with no interval, 3,000
rows in git, a key in a script) and green on `REAL`, so it can fail.

```python
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
```

## 5. The offline path and the gates

`smoke.sh` runs `selftest.py` and `run.py --dry-run` from the recipe's own
directory with no key, no network and no GPU, in the SDK's own environment
(no numpy unless the SDK has it), and CI runs it on every pull request. A
selftest pins the failure as well as the pass: a metric that cannot fail
reads as coverage. Then, from the repo root: `uv run pytest -q`, `uv run
ruff check . && uv run ruff format --check .`, `uv run mypy`, the
`scripts/check_*.py` gates, `python scripts/gen_recipe_docs.py` (the docs
page is generated from the README, never written by hand), a row in
`recipes/README.md`, an entry under `## Unreleased` in `CHANGELOG.md`, and
the recipe's entry points listed in `tests/recipes/test_offline_examples.py`.

## 6. Post the numbers, then open the pull request

The run page and the README must agree, so the platform post reads
`results.json`, not a second copy of the numbers. One run per arm, the
baseline as the served version, `targets=` set at creation, `ci=` on every
score, `test_version` the pin, a five-line note on every trained arm
(Changed, Moved, Why, Learned, Reproduce; `manage-experiments/`), and the
verdict printed. Points out of 100, never a fraction.

```python
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
```

The pull request: `gh pr create` fails in this repository; use
`gh api -X POST repos/whilehq/whileai-sdk/pulls -f title=.. -f head=<branch>
-f base=main -F draft=true -F body=@body.md`. The body opens with `Closes
#n` or `No issue: ...`, names what the files are when there are more than
about fifteen, and ends with the attribution line. A pull request that
conflicts with main gets no CI at all, so rebase first and read the checks
after. Never `git add -A`; add by path.
