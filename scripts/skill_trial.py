"""Run a cold coding agent on one skill and grade what reached the platform.

The deterministic check (`skills/<name>/check.py`) proves the playbook's
code runs. This proves a fresh agent can follow it. It opens a temporary
repo with a small traces file, hands Claude Code the one-line ask a person
would type plus the skill path, lets it work unattended, then asks the
platform whether a tracked agent with the trial's name now has a run with
scores on more than one behavior.

    WHILEAI_API_KEY=zp_... python scripts/skill_trial.py sft-from-traces
    python scripts/skill_trial.py --all --model sonnet --minutes 20

Needs the `claude` CLI signed in with API credit. Writes one JSON line per
trial to `out/skill_trials.jsonl` so runs can be compared across releases.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKILLS = REPO / "skills"
OUT = REPO / "out" / "skill_trials.jsonl"

ASKS = {
    "sft-from-traces": "Train a specialised model on refund emails from my traces in traces.jsonl.",
    "dpo-pairs": "Make my refund agent's replies better with preference pairs from traces.jsonl.",
    "grpo-verifier": "Train my refund agent with RL; the rule is: look an order up before refunding it.",
    "character": "Make my assistant less sycophantic and warmer, from the constitution in spec.md.",
    "tool-call-efficiency": "My agent solves tasks but calls tools twice as often as it needs to. Fix that.",
    "watch": "Set up the daily check that reports yesterday's traffic for my refund agent.",
}


def make_fixture(root: Path, skill: str) -> None:
    """A tiny repo: traces.jsonl from the skill's own check (offline), a spec if needed."""
    check = SKILLS / skill / "check.py"
    env = {**os.environ, "WHILEAI_SKILL_FIXTURE_DIR": str(root)}
    subprocess.run(
        [sys.executable, str(check)], cwd=root, env=env, check=False, capture_output=True
    )
    if not (root / "traces.jsonl").exists():
        rows = [
            {"prompt": f"Refund order A100{i}", "steps": [], "final_text": "Done.", "reward": i % 2}
            for i in range(24)
        ]
        (root / "traces.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if skill == "character" and not (root / "spec.md").exists():
        (root / "spec.md").write_text(
            "# Constitution\n\n- Be warm.\n- Avoid sycophancy.\n- Be clear.\n"
        )


def run_agent(root: Path, skill: str, name: str, model: str, minutes: int) -> dict:
    ask = (
        f"{ASKS[skill]} Track the result on the While platform under the name '{name}'. "
        f"Follow the playbook at {SKILLS / skill / 'SKILL.md'} (the whileai package is "
        "installed; WHILEAI_API_KEY is set). Work unattended; do not ask me questions."
    )
    claude = shutil.which("claude") or shutil.which("claude.cmd")
    if not claude:
        return {"seconds": 0, "rc": -2, "stdout_tail": "claude CLI not found on PATH"}
    cmd = [
        claude,
        "-p",
        ask,
        "--model",
        model,
        "--output-format",
        "json",
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        "Bash,Read,Write,Edit,Glob,Grep",
    ]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=root, capture_output=True, text=True, timeout=minutes * 60, check=False
        )
        out = proc.stdout
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        out, rc = "", -1
    return {"seconds": round(time.time() - started), "rc": rc, "stdout_tail": out[-2000:]}


def grade(name: str) -> dict:
    from whileai.platform import Tracked, tracked_agents

    agents = {a.id: a for a in tracked_agents()}
    if name not in agents:
        return {"reported": False, "reason": "no tracked agent with that name"}
    t = Tracked(name)
    runs = t.runs()
    d = t.dashboard()
    scored = [x for x in d.deltas if x.candidate is not None]
    return {
        "reported": bool(runs),
        "runs": len(runs),
        "behaviors": d.behaviors,
        "scored_behaviors": [x.name for x in scored],
        "has_interval": any(v.ci is not None for v in d.versions),
        "verdict": str(d.verdict),
        "pass": bool(runs) and len(scored) >= 2 and any(v.ci is not None for v in d.versions),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("skill", nargs="?", choices=sorted(ASKS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--minutes", type=int, default=20)
    ap.add_argument("--keep", action="store_true", help="keep the temp repo for inspection")
    args = ap.parse_args()
    if not os.environ.get("WHILEAI_API_KEY"):
        print("set WHILEAI_API_KEY", file=sys.stderr)
        return 2
    skills = sorted(ASKS) if args.all else [args.skill]
    if not skills or skills == [None]:
        ap.error("name a skill or pass --all")
    OUT.parent.mkdir(exist_ok=True)
    for skill in skills:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        name = f"trial-{skill}-{stamp}"
        root = Path(tempfile.mkdtemp(prefix=f"skill-{skill}-"))
        make_fixture(root, skill)
        result = {"skill": skill, "name": name, "model": args.model, "at": stamp}
        result.update(run_agent(root, skill, name, args.model, args.minutes))
        result.update(grade(name))
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(
            f"{skill}: {'PASS' if result.get('pass') else 'FAIL'} in {result['seconds']} s; {result.get('verdict', result.get('reason'))}"
        )
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
