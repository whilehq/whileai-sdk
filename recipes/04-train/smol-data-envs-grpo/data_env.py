"""SmolDataEnvs, one turn: prompt in, reward out. Shared by both arms.

Ported from the dataset authors' ``04-smoldataenvs/scripts/rollout.py``
(FineEnvs @ 08a5622, Apache-2.0): the same system prompt, user prompt, code
extraction, "last printed line is the answer" rule, command-shaped-answer
rule and grader. One change: their program runs in a Hugging Face Sandbox;
here it runs in a subprocess next to a copy of the task's tables, because the
tables are already on the training container's volume. The reward logic is
theirs so that the only thing the two arms disagree on is which tasks they
train on.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from grader import grade as _grade

SYSTEM = (
    "You are a data analyst. You answer questions about CSV files by writing a short "
    "Python program and reading what it prints."
)
PROMPT = """{question}

The files are in /home/user/input and your program runs in that directory:
{files}

Write one Python program in a ```python block, then stop.
- Look at the data if you need to, then compute the answer.
- The LAST thing the program prints must be the answer on its own: a bare number
  (no commas or units), a short label, yes/no, or a comma-separated list.
- Keep it under 40 lines. pandas, numpy, scipy, sklearn and statsmodels are installed."""

INPUT_DIR = "/home/user/input"
#: The authors' sandbox kills a program after 90 s.
RUN_TIMEOUT_S = 90


def build_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    files = "\n".join(f"- {f}" for f in row["files"])
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT.format(question=row["question"], files=files)},
    ]


_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
# Adapted by the authors from their whitebox-bash grader, where 42% of partial
# credit once went to strings like `echo -n "2.14" > answer.txt`.
_COMMAND_SHAPED = re.compile(
    r"(^|\s)(echo|printf|cat|python3?|bash|sh|tee|awk|sed)\b|[>|]{1,2}\s*\S+|\$\(|`",
)


def extract_code(completion: str | list[dict]) -> str:
    """Last fenced block wins; no fence means the whole reply is the program."""
    if isinstance(completion, list):
        completion = completion[-1]["content"] if completion else ""
    blocks = _CODE_RE.findall(completion or "")
    if blocks:
        return blocks[-1].strip()
    return (completion or "").strip()


def looks_like_a_command(answer: str) -> bool:
    return bool(answer) and bool(_COMMAND_SHAPED.search(answer.strip()))


def last_line(stdout: str) -> str:
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def run_program(code: str, table_dir: str, files: list[str]) -> tuple[str, str]:
    """Run ``code`` with the task's tables in its working directory.

    The prompt tells the model the files are in /home/user/input; that path
    is rewritten to the private working directory so concurrent rollouts of
    different tasks never see each other's files.
    """
    with tempfile.TemporaryDirectory() as tmp:
        for name in dict.fromkeys(files):  # a few tasks list one table twice
            src = Path(table_dir) / name
            if src.exists():
                os.symlink(src, Path(tmp) / name)
        prog = Path(tmp) / "solve.py"
        prog.write_text(code.replace(INPUT_DIR, tmp), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(prog)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=RUN_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return "", f"timed out after {RUN_TIMEOUT_S}s"
    return proc.stdout or "", proc.stderr or ""


def score(row: dict[str, Any], completion: str | list[dict], table_root: str) -> dict[str, Any]:
    """One completion, one graded result. ``reward`` is None when the task's
    tables are not on the volume: an environment failure, not a wrong answer."""
    code = extract_code(completion)
    try:
        compile(code, "<rollout>", "exec")
    except SyntaxError as exc:
        return {"reward": 0.0, "ran": 0.0, "prediction": "", "why": f"syntax: {exc.msg}"}
    table_dir = os.path.join(table_root, row["bucket_prefix"])
    if row["files"] and not all(os.path.exists(os.path.join(table_dir, f)) for f in row["files"]):
        return {"reward": None, "ran": 0.0, "prediction": "", "why": "tables unavailable"}
    try:
        stdout, stderr = run_program(code, table_dir, list(row["files"]))
    except OSError as exc:  # the harness, not the program: ungraded, and counted
        return {"reward": None, "ran": 0.0, "prediction": "", "why": f"harness: {exc}"[:200]}
    prediction = last_line(stdout)
    if looks_like_a_command(prediction):
        return {"reward": 0.0, "ran": 1.0, "prediction": prediction, "why": "command-shaped"}
    reward = 0.0
    if prediction:
        reward = _grade(
            str(row["answer"]),
            prediction,
            reward_mode=str(row["reward_mode"] or ""),
            abs_tol=float(row["atol"] or 0.0),
            rel_tol=float(row["rtol"] or 0.0),
        ).reward
    return {
        "reward": float(reward),
        "ran": 1.0 if prediction else 0.0,
        "prediction": prediction[:120],
        "why": "graded" if prediction else f"no output: {stderr.strip().splitlines()[-1:]}"[:200],
    }
