"""Code-execution verifier: run the candidate against tests, reward = pass.

RLVR (Lambert et al. 2024, arXiv:2411.15124; Lambert 2025, chapter Tool Use):
the strongest verifiable reward for coding is executing the code against a
test suite. The candidate's code is pulled from the final answer (a ```python
fence, else the whole text). Tests come from, in order: the ``tests=``
argument, ``privileged.tests`` on the row, or the reference.

Isolation: the code runs in a fresh subprocess with ``-I`` (isolated mode),
a private temp working directory, a wall-clock timeout, and, on POSIX, CPU
and address-space limits. This stops runaway loops and accidents. It is NOT
a security boundary against hostile code: for untrusted policies run the
verifier inside a container or the hosted sandbox. ``allow_network`` is
documented as unenforced here for that reason.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

from .base import Verifier, _reference_with_source

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*(.+?)```", re.S)


def extract_code(text: str) -> str:
    """The last python code fence, else the whole text."""
    blocks = _CODE_FENCE.findall(str(text or ""))
    if blocks:
        return blocks[-1].strip()
    return str(text or "").strip()


_POSIX_LIMITS = """
import resource, sys
try:
    resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
    resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
except Exception:
    pass
"""


#: An exception class name, the one part of a failure that is not written
#: by the tests themselves.
_EXC_NAME = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|Exit)\b")


class CodeExec(Verifier):
    """Reward is 1 if the candidate code plus the tests run to completion with
    exit code 0, else 0. ``tests`` is Python that exercises the candidate
    (asserts, or a ``def test_*()`` / unittest / pytest-style file); it is
    appended after the candidate in the same module namespace.

    Parameters:
        tests: test source. If None, read ``privileged.tests`` / reference.
        timeout: wall-clock seconds (default 10).
        setup: code prepended before the candidate (imports, fixtures).
        cpu_seconds / mem_mb: POSIX resource caps (best-effort).
        python_bin: interpreter to run with (default this one).

    When the tests came from ``privileged.tests`` the failure reason names
    the exception and nothing else: the failing line and its assertion
    message are written by the answer key, and ``reason`` is carried into
    every training export. Pass ``tests=`` (your own copy, not the
    teacher's) to see the full failure tail while iterating.
    """

    kind = "rule"

    def __init__(
        self,
        *,
        tests: str | None = None,
        timeout: float = 10.0,
        setup: str = "",
        cpu_seconds: int = 10,
        mem_mb: int = 512,
        python_bin: str | None = None,
        field: str | None = None,
        name: str | None = None,
    ):
        super().__init__(field=field, name=name or "CodeExec")
        self.tests = tests
        self.timeout = timeout
        self.setup = setup
        self.cpu_seconds = cpu_seconds
        self.mem_mb = mem_mb
        self.python_bin = python_bin or sys.executable

    def _tests_for(self, reference: Any, row: dict) -> str | None:
        if self.tests is not None:
            return self.tests
        priv = row.get("privileged") if isinstance(row, dict) else None
        if isinstance(priv, dict) and priv.get("tests"):
            return str(priv["tests"])
        if isinstance(reference, str) and reference.strip():
            return reference
        return None

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        tests = self._tests_for(reference, row)
        if not tests:
            return None
        code = extract_code(candidate)
        if not code:
            return 0, "no code in answer"

        preamble = ""
        if os.name == "posix":
            preamble = _POSIX_LIMITS.format(cpu=self.cpu_seconds, mem=self.mem_mb * 1024 * 1024)
        # If the tests define pytest-style functions, run them; else the plain
        # asserts execute at import. Append a tiny runner that calls any
        # top-level test_* functions so both styles work.
        runner = textwrap.dedent(
            """
            import sys as _sys
            _tests = [(_n, _o) for _n, _o in list(globals().items())
                      if _n.startswith("test_") and callable(_o)]
            _failed = 0
            for _n, _o in _tests:
                try:
                    _o()
                except Exception as _e:
                    _failed += 1
                    print(f"FAIL {_n}: {type(_e).__name__}: {_e}", file=_sys.stderr)
            if _failed:
                _sys.exit(1)
            """
        )
        program = "\n".join([preamble, self.setup, code, "\n# --- tests ---\n", tests, runner])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "candidate_check.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(program)
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
            if os.name == "nt":
                env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
            try:
                proc = subprocess.run(
                    [self.python_bin, "-I", path],
                    cwd=tmp,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return 0, f"timed out after {self.timeout}s"
            except Exception as exc:
                return 0, f"could not run: {type(exc).__name__}: {exc}"[:150]
        if proc.returncode == 0:
            return 1, "all tests passed"
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = err[-1] if err else f"exit {proc.returncode}"
        if self._tests_are_privileged(row):
            # The failing line is echoed out of the test source, and an
            # assertion message is written by whoever wrote the tests -- so
            # for ``privileged.tests`` the tail *is* the answer key, and
            # ``reason`` rides into the training export. Name the exception,
            # which says what went wrong without saying what was expected.
            kind = _EXC_NAME.search("\n".join(err[-5:]))
            detail = kind.group(0) if kind else "no detail (tests are privileged)"
            return 0, f"tests failed: {detail}"
        return 0, f"tests failed: {tail}"[:200]

    def _tests_are_privileged(self, row: dict) -> bool:
        """Whether the tests this run used came out of the teacher's block."""
        if self.tests is not None or not isinstance(row, dict):
            return False
        priv = row.get("privileged")
        if isinstance(priv, dict) and priv.get("tests"):
            return True
        # the same tests, read as ``privileged.reference`` instead
        return _reference_with_source(row, self.field)[1]
