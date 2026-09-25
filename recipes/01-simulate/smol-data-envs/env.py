"""SmolDataEnvs as a whileai environment: tasks, tables, a program runner, a reward.

A task is a question about real Kaggle tables with one known answer. The
policy writes one Python program; the runner executes it next to the tables;
the last line the program prints is the answer; the dataset's own grader
compares it to the gold. No model grades anything.

Two reward rules come from the dataset's GRPO notes (FineEnvs PR #14):

* a rollout the environment could not grade (a table failed to download) is
  ``None``, never ``0.0``, so an infrastructure failure never reads as a wrong
  answer. The verifier raises ``VerifierError``; ``run_judge`` records
  ``reward=None`` and the RL gates drop the row.
* a shell answer (``echo "2.14"``) earns zero, whatever it prints: the reward
  pays for computing the value from the data, not for typing it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from grader import grade

from whileai.simulations.verify import Verifier, VerifierError

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
RAW = HERE / "raw"

DATASET = "FineEnvs/SmolDataEnvs"
#: The dataset commit this recipe was written against; the vendored grader
#: in grader.py is the same revision.
REVISION = "b2bf35647e2381b1ab12c2ad7862cbbe4e8f857b"
HF = "https://huggingface.co"

#: Wall-clock seconds one program may run. The dataset's tables are up to a
#: few hundred MB of CSV; pandas reads the largest in well under a minute.
RUN_TIMEOUT_S = 60.0
#: Tries per file. The Hugging Face CDN resets a connection now and then; a
#: table that still fails leaves its task ungraded, not wrong.
DOWNLOAD_ATTEMPTS = 3

SYSTEM = (
    "You are a data analyst. You answer a question about data files by writing "
    "one Python program. The program runs once, with the files in ./input, and "
    "pandas and numpy installed. Inspect what you need inside the program. "
    "The last line the program prints is your answer: a bare number (no commas "
    "or units, keep decimal precision), a short label, yes/no, or a "
    "comma-separated list. Reply with the program in a single ```python block."
)

_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.S)
_SHELL_LANGS = {"bash", "sh", "shell", "zsh", "console", "shell-session"}


def prompt_for(task: dict[str, Any]) -> str:
    """The user turn: the question and the files the program will find."""
    files = "\n".join(f"- input/{name}" for name in task["files"])
    return f"Files:\n{files}\n\nQuestion: {task['question']}"


# -- tasks ------------------------------------------------------------------


def fixture_tasks() -> list[dict[str, Any]]:
    """Four tasks in the dataset's row shape over two small tables checked in
    under fixtures/tables. The fourth names a table that is not there, to
    show an ungraded rollout. Offline, no download."""
    tables = FIXTURES / "tables"
    rows = [json.loads(line) for line in (FIXTURES / "tasks.jsonl").read_text().splitlines()]
    for r in rows:
        missing = [name for name in r["files"] if not (tables / name).exists()]
        if missing:
            r["table_error"] = f"missing {', '.join(missing)}"
        else:
            r["table_dir"] = str(tables)
    return rows


def hf_tasks(split: str, limit: int | None = None) -> list[dict[str, Any]]:
    """One split of the dataset (train 5,000 / test 250 / eval 144), pinned to
    ``REVISION``. Tasks with no tables to read are dropped: nothing in the
    sandbox can answer them. Needs pandas and pyarrow to read the parquet."""
    import pandas as pd

    path = _download(
        f"{HF}/datasets/{DATASET}/resolve/{REVISION}/data/{split}-00000-of-00001.parquet",
        RAW / f"{split}.parquet",
    )
    df = pd.read_parquet(path)
    tasks = []
    for rec in df.to_dict("records"):
        rec["files"] = [str(f) for f in rec["files"]]
        if not rec["files"]:
            continue
        tasks.append(rec)
    tasks.sort(key=lambda t: t["task_id"])
    return tasks[:limit] if limit else tasks


def fetch_tables(task: dict[str, Any]) -> dict[str, Any]:
    """Download the task's tables once into raw/tables/<prefix>/ and point the
    task at them. A failed download leaves ``table_dir`` unset, which the
    reward reads as "ungraded", not as a wrong answer."""
    dest = RAW / "tables" / task["bucket_prefix"]
    try:
        for name in task["files"]:
            url = f"{HF}/buckets/{task['hf_bucket']}/resolve/{task['bucket_prefix']}/{name}"
            _download(url, dest / name)
    except Exception as exc:
        task["table_error"] = f"{type(exc).__name__}: {exc}"[:200]
        return task
    task["table_dir"] = str(dest)
    return task


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with part.open("wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
            break
        except requests.ConnectionError:
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    part.replace(dest)
    return dest


# -- running a program --------------------------------------------------------


def split_answer(text: str) -> tuple[str, str]:
    """(language, code) of the last fenced block, else ("", whole text)."""
    blocks = _FENCE.findall(str(text or ""))
    if blocks:
        lang, code = blocks[-1]
        return lang.lower(), code.strip()
    return "", str(text or "").strip()


def run_program(code: str, table_dir: str, files: list[str], *, timeout: float = RUN_TIMEOUT_S):
    """Run ``code`` in a fresh temp directory with the tables copied into
    ./input. Returns (printed answer or None, why).

    A local subprocess with a timeout, the same isolation as the SDK's
    ``CodeExec``: it stops runaway loops, and it is not a security boundary.
    Run a policy you do not trust inside a container or a sandbox service.
    """
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "input"
        inp.mkdir()
        for name in files:
            shutil.copy(Path(table_dir) / name, inp / name)
        prog = Path(tmp) / "solution.py"
        prog.write_text(code, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(prog)],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout:.0f}s"
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        return None, f"program failed: {err[-1] if err else f'exit {proc.returncode}'}"[:200]
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None, "program printed nothing"
    return lines[-1], "ran"


# -- the reward ---------------------------------------------------------------


class DataEnvReward(Verifier):
    """1 when the program's last printed line matches the gold under the
    dataset's grader, 0 when it does not (or crashed, or answered in shell),
    ``None`` when the environment could not grade the rollout.

    The gold is ``privileged.reference``; the task's grading knobs
    (``reward_mode``, ``atol``, ``rtol``) and its tables ride on the row.
    """

    def __init__(self, *, timeout: float = RUN_TIMEOUT_S, name: str = "SmolDataEnvs"):
        super().__init__(name=name)
        self.timeout = timeout

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        if reference is None:
            raise VerifierError("row has no gold answer")
        if not row.get("table_dir"):
            raise VerifierError(f"tables unavailable ({row.get('table_error', 'not fetched')})")
        lang, code = split_answer(candidate)
        if lang in _SHELL_LANGS:
            return 0, "shell answer, no credit: the value must be computed from the data"
        if not code:
            return 0, "no program in the reply"
        printed, why = run_program(code, row["table_dir"], row["files"], timeout=self.timeout)
        if printed is None:
            return 0, why
        result = grade(
            str(reference),
            printed,
            reward_mode=str(row.get("reward_mode") or ""),
            abs_tol=float(row.get("atol") or 0.0),
            rel_tol=float(row.get("rtol") or 0.0),
        )
        return result.reward, f"printed {printed[:60]!r}: {result.method}"


def task_row_fields(task: dict[str, Any]) -> dict[str, Any]:
    """What a rollout row carries from its task: the gold under
    ``privileged`` (the training export never projects it), the grading
    knobs and the tables in the open."""
    return {
        "privileged": {"reference": str(task["answer"])},
        "category": task.get("difficulty_tier") or "unknown",
        "reward_mode": task.get("reward_mode"),
        "atol": task.get("atol"),
        "rtol": task.get("rtol"),
        "files": list(task["files"]),
        "table_dir": task.get("table_dir"),
        "table_error": task.get("table_error"),
    }


def fixture_rollouts() -> list[dict[str, Any]]:
    """Four canned replies per fixture task, written to walk every branch of
    the reward: right, right with a tolerance, wrong, crashed, shell, and
    ungraded. ``note`` says which one each is."""
    lines = (FIXTURES / "rollouts.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]
