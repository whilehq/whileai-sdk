"""Offline check for skills/harness-search/SKILL.md: the coding agent as the
proposer in a Meta-Harness loop (Lee et al. 2026, arXiv:2603.28052) on the
agent's own production traffic.

A temporary copy of ``recipes/papers/meta-harness`` starts with the baseline
only; the checked-in candidates stand in for what the proposer would write,
one per round. The fixture writes three days of traffic the way a log
export looks (prompt, steps, final_text, ts), and, once the pick is served,
the day after. Every block in SKILL.md is below, verbatim: run the recipe's
dry run on the traces, read the split, the ledger, the proposal and the
selection, write the next candidate until the gate passes or the rounds run
out, report the five lines, post every candidate as a harness version to a
recording fake platform, then score the next day and post one LiveDay. No
key, no network, no GPU.

    uv run python skills/harness-search/check.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any

import whileai as wai
from whileai.platform import Behavior, Judge, LiveDay, PlatformError, track
from whileai.simulations import evaluate, load_traces

T0 = time.monotonic()
TIME_LIMIT = 60  # seconds; the bar every skill check meets (skills/BRIEF.md)

# ---------------------------------------------------------------- the recipe copy

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "recipes" / "papers" / "meta-harness"
TMP = Path(tempfile.mkdtemp(prefix="harness-search-"))
RECIPE = TMP / "meta-harness"
shutil.copytree(SOURCE, RECIPE, ignore=shutil.ignore_patterns("out", "__pycache__"))
sys.path.insert(0, str(RECIPE))
import common  # noqa: E402  (the recipe's tools, judge and scripted stand-in)

MODEL = "scripted"  # the search model; the recipe's default held-out model is scripted-b
ROUNDS = 3
TOOLS = common.TOOLS
judge = common.judge  # the program judge the holdout uses; the next day is scored by it too

# The checked-in candidates after the baseline are what the proposer would
# write, in order; the copy starts without them so the loop has to.
IDEAS = sorted(p for p in (RECIPE / "candidates").glob("*.py") if not p.name.startswith("00_"))
BODIES = [(p.stem.split("_", 1)[1], p.read_text(encoding="utf-8")) for p in IDEAS]
for p in IDEAS:
    p.unlink()
proposed = 0

# ---------------------------------------------------------------- the traffic

DAYS = [date(2026, 9, 19), date(2026, 9, 20), date(2026, 9, 21)]
DAY_AFTER = date(2026, 9, 22)
TRACES = TMP / "traces.jsonl"
NEXT_DAY = TMP / f"traces-{DAY_AFTER.isoformat()}.jsonl"


def _served(label: str) -> wai.Harness:
    """The harness in production before the search: the baseline's scripted
    stand-in at the baseline's planted rate."""
    return common.build(
        MODEL,
        instructions=common.BASE_INSTRUCTIONS,
        label=label,
        scripted_rate=0.45,
        scripted_behaviors=None,
    )


def write_traffic(harness: wai.Harness, path: Path, days: list[date], *, seed: int) -> int:
    """Rows the way a log export looks: prompt, steps, final_text, ts. The
    offline writer phrases the asks; the harness answers them."""
    data = wai.simulate(
        harness,
        seeds=common.SEEDS,
        situations=24,
        mode="rl",
        repeats=1,
        simulator=False,
        reproducible=True,
        seed=seed,
        concurrency=1,
    )
    seen: set[str] = set()
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in data.rows():
            if row["prompt"] in seen:
                continue
            seen.add(row["prompt"])
            day = days[n % len(days)]
            fh.write(
                json.dumps(
                    {
                        "ts": f"{day.isoformat()}T{8 + n % 10:02d}:00:00Z",
                        "prompt": row["prompt"],
                        "steps": row["steps"],
                        "final_text": row["final_text"],
                    },
                    default=str,
                )
                + "\n"
            )
            n += 1
    return n


N_TRACES = write_traffic(_served("served"), TRACES, DAYS, seed=0)


def run(*flags: str) -> str:
    """The recipe's dry run in the copy: scripted models, no key."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("WHILEAI_")}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "run.py", "--dry-run", *flags],
        cwd=RECIPE,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIME_LIMIT,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc.stdout


def propose_candidate(proposal: str) -> tuple[str, str]:
    """The proposer, scripted: the next checked-in candidate. In use, this is
    the coding agent reading the worst rows in ``proposal`` and writing the
    file; the shape of the file before it is the template."""
    global proposed
    assert "Worst rows:" in proposal and "```python" in proposal
    name, body = BODIES[proposed]
    proposed += 1
    return name, body


def load_harness(candidate: str, model: str) -> wai.Harness:
    """A candidate file as the ``wai.Harness`` it defines, so the platform
    record carries the same fingerprint the ledger does."""
    path = RECIPE / "candidates" / candidate
    spec = importlib.util.spec_from_file_location(f"candidate_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.harness(model)


# ---------------------------------------------------------------- the fake platform


class FakePlatform:
    """Records every call and answers like the API: runs keep their evals,
    notes and record; behaviors by name; the dashboard is built from the
    evals; live rows are kept by day."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self.experiment: dict[str, Any] | None = None
        self.behaviors: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.figures: dict[str, dict[str, Any]] = {}
        self.live: list[dict[str, Any]] = []

    def __call__(self, method: str, path: str, body: Any = None) -> Any:
        self.calls.append((method, path, body))
        if path == "/agents":
            return {**body, "name": body.get("name") or body["id"]}
        if path.endswith("/experiment"):
            if method == "GET":
                if self.experiment is None:
                    raise PlatformError(404, "no experiment")
                return self.experiment
            self.experiment = dict(body)
            return self.experiment
        if "/behaviors" in path:
            if method == "GET":
                return {"behaviors": list(self.behaviors.values())}
            name = path.rsplit("/", 1)[1]
            self.behaviors[name] = {"name": name, **body}
            return self.behaviors[name]
        if "/figures" in path:
            if method == "GET":
                return {"figures": list(self.figures.values())}
            name = path.rsplit("/", 1)[1]
            self.figures[name] = {"name": name, **body}
            return self.figures[name]
        if path == "/runs" and method == "POST":
            rid = f"run_{len(self.runs):03d}"
            self.runs[rid] = {"id": rid, "status": "running", "evals": [], "train": [], **body}
            return self.runs[rid]
        if path.startswith("/runs?"):
            return {"runs": list(self.runs.values())}
        if path.endswith("/train"):
            self.runs[path.split("/")[2]]["train"].extend(body)
            return {"written": len(body)}
        if path.endswith("/evals"):
            run_ = self.runs[path.split("/")[2]]
            run_["evals"].extend({**e, "version": run_["version"]} for e in body)
            return {"evals": body}
        if path.startswith("/runs/") and method == "PATCH":
            self.runs[path.split("/")[2]].update(body)
            return self.runs[path.split("/")[2]]
        if path == "/live":
            self.live.extend(body)
            return {"written": len(body)}
        if "/dashboard" in path:
            agent = path.split("/")[2]
            evals = [e for r in self.runs.values() if not r.get("archived") for e in r["evals"]]
            serving = next(iter(self.runs.values()), {}).get("version", "policy")  # the baseline
            return {
                "agent": {"id": agent, "name": agent, "serving": serving},
                "behavior": next(iter(self.behaviors.values()), None),
                "behaviors": list(self.behaviors),
                "versions": [
                    {"v": e["version"], "score": e["score"], "ci": e.get("ci"), "n": e.get("n")}
                    for e in evals
                ],
            }
        return {"ok": True}


fake = FakePlatform()

# ---------------------------------------------------------------- SKILL.md blocks


def read_ledger(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def state() -> tuple[str, list[dict], dict]:
    out = RECIPE / "out"
    return (
        (out / "proposal.md").read_text(encoding="utf-8"),
        read_ledger(out / "ledger.jsonl"),
        json.loads((out / "selected.json").read_text(encoding="utf-8")),
    )


run("--propose", "--select", "--fresh", "--traces", str(TRACES))
proposal, ledger, selected = state()
split = json.loads((RECIPE / "out" / "split.json").read_text(encoding="utf-8"))
print(split["how"], "|", split["contamination"]["n_dropped"], "train prompt(s) left the window")

assert len(ledger) == 1 and selected["selected"] is None, "the copy starts with the baseline only"
assert "only the baseline has run" in selected["reason"], selected
assert split["how"].startswith("holdout is 2026-09-2"), split["how"]
assert ledger[0]["n_tasks"] + split["contamination"]["n_dropped"] == N_TRACES, "one task a prompt"

NEXT = re.compile(r"write candidates/(\d\d)_<name>\.py")

for _round in range(ROUNDS):
    if selected["selected"]:
        break
    number = NEXT.search(proposal).group(1)
    name, body = propose_candidate(proposal)  # you: read the worst rows, write the file
    (RECIPE / "candidates" / f"{number}_{name}.py").write_text(body, encoding="utf-8")
    run("--propose", "--select", "--traces", str(TRACES))
    proposal, ledger, selected = state()

pick_name = selected["selected"] or selected["best_on_train"]
base, pick = ledger[0], next(e for e in ledger if e["candidate"] == pick_name)
check, cost = selected["checks"][MODEL], selected["checks"]["cost"]
lo, hi = check["ci95"]
worst = read_ledger(RECIPE / "out" / base["worst"])
note = (
    f"Changed: {pick['candidate']}, fingerprint {pick['fingerprint']}, against {base['candidate']}.\n"
    f"Moved: {100 * base['holdout']['pass_at_1']:.0f} to {100 * pick['holdout']['pass_at_1']:.0f} "
    f"points on {pick['holdout']['n_tasks']} held-out tasks ({split['how'].split(':')[0]}), "
    f"{100 * check['delta']:+.0f} [{100 * lo:+.0f}, {100 * hi:+.0f}]; {check['regressed']} task(s) "
    f"the baseline passed and it failed; {cost['ratio']:.2f}x the cost per rollout.\n"
    f"Why: the baseline's worst rows were {'; '.join(sorted({w['why'] for w in worst}))}.\n"
    f"Learned: {selected['reason']} (Meta-Harness, Lee et al. 2026, arXiv:2603.28052).\n"
    f"Reproduce: cd recipes/papers/meta-harness && python run.py --dry-run --traces {TRACES.name} "
    "--propose --select --seed 0"
)

tracked = track("support-bot", model=MODEL, transport=fake)  # drop transport= for real
tracked.behavior(
    Behavior(
        name="plain_answer",
        test_version="t-" + base["fingerprint"][:8],
        n=pick["holdout"]["n_tasks"],
        judge=Judge(name="filler, fault and leak checks as a program"),
        reward_is_judge=False,
    )
)
for entry in ledger:
    version = tracked.run(
        entry["label"],
        method="eval",
        harness=load_harness(entry["candidate"], MODEL),
        targets=["plain_answer"],
    )
    ci = entry["holdout"]["ci95"]
    version.score(
        "plain_answer",
        100 * entry["holdout"]["pass_at_1"],
        ci=50 * (ci[1] - ci[0]),
        n=entry["holdout"]["n_tasks"],
    )
    if entry["candidate"] == pick["candidate"]:
        version.note(note)
    version.finish(say=False)
print(note)
print("verdict:", tracked.verdict())

# The pick is served; the day after arrives in the logs. Here the picked
# harness answers a fresh draw of asks and the rows are written the way
# the first three days were.
write_traffic(load_harness(pick["candidate"], MODEL), NEXT_DAY, [DAY_AFTER], seed=1)

next_day = load_traces(str(NEXT_DAY))  # the day after the pick was served, from your logs
scored = evaluate(next_day, judge, tools=TOOLS)  # the same judge the holdout used
flagged = len(scored.failures())
tracked.live(LiveDay(day=DAY_AFTER, version=pick["label"], replies=len(next_day), flagged=flagged))
live_fail = 100 * flagged / len(next_day)
holdout_fail = 100 * (1 - pick["holdout"]["pass_at_1"])
print(f"next day: {live_fail:.0f} of 100 flagged; the holdout said {holdout_fail:.0f} of 100")

# ---------------------------------------------------------------- assertions

files = sorted(p.name for p in (RECIPE / "candidates").glob("*.py"))
assert [e["candidate"] for e in ledger] == files, "one ledger line per candidate file"
assert all(f"## {f}" in proposal for f in files), "the proposal names every candidate"
assert "The holdout is 2026-09-2" in proposal, "the proposal says which days decide"
assert selected["selected"] is not None, f"the gate never passed in {ROUNDS} rounds: {selected}"
assert proposed >= 1, "the loop wrote at least one candidate"
assert lo > 0, "a selected candidate clears zero on the holdout"
assert cost["clears"] and cost["ratio"] <= 1.0 + 1e-9, "a pick costs no more than the baseline"
assert isinstance(check["regressed"], int)
assert selected["tasks_led"][pick_name] == max(selected["tasks_led"].values()), "led the most"
assert selected["attribution"]["verdict"] in ("harness", "unresolved"), selected["attribution"]
assert all(e["cost"]["calls"] is not None for e in ledger), "cost per rollout on every line"

for word in ("Changed:", "Moved:", "Why:", "Learned:", "Reproduce:", "x the cost"):
    assert word in note, word
posted = [c[2] for c in fake.calls if c[0] == "POST" and c[1] == "/runs"]
assert [p["version"] for p in posted] == [e["label"] for e in ledger], "every candidate is a run"
pins = {p["version"]: p["record"]["provenance"]["pins"]["harness"] for p in posted}
assert pins == {e["label"]: e["fingerprint"] for e in ledger}, "the run pins the ledger's hash"
scores = [e for c in fake.calls if c[1].endswith("/evals") for e in c[2]]
assert all(e["score"] > 1 for e in scores), "points, not fractions"
notes = [c[2]["notes"] for c in fake.calls if c[0] == "PATCH" and "notes" in (c[2] or {})]
assert notes == [note], "the note sits on the picked candidate only"
assert len(fake.live) == 1 and fake.live[0]["version"] == pick["label"], "one LiveDay on the pick"
assert fake.live[0]["replies"] == len(next_day) and fake.live[0]["flagged"] == flagged
assert abs(live_fail - holdout_fail) < 35, (
    f"the next day ({live_fail:.0f}) and the holdout ({holdout_fail:.0f}) disagree by more than "
    "the scripted rates allow; the loop did not close"
)

shutil.rmtree(TMP, ignore_errors=True)
assert time.monotonic() - T0 < TIME_LIMIT
print(f"ok in {time.monotonic() - T0:.1f}s, {len(ledger)} candidates, picked {pick_name}")
