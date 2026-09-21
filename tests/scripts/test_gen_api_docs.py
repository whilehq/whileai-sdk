"""scripts/gen_api_docs.py: a docstring renders to MDX with its code left alone.

Prose gets the MDX escapes (braces, angle brackets) and the RST-style
``name`` collapsed to `name`. A fenced block and a doctest are code: the
fence line stays a fence and the lines inside come out verbatim.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "gen_api_docs.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_api_docs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fenced_example():
    """Build the ``rows`` and print them.

    ```python
    report = build({"a": 1}, mode="``")
    print(report["n"])
    ```

    See ``other`` for the {inverse}, or ``a
    wrapped name`` on the next line.
    """


def _doctest_example():
    """Count them.

    >>> x = count({"a": 1})
    >>> x
    1

    Done.
    """


def test_fenced_example_stays_fenced(gen):
    lines = gen._docstring(_fenced_example).split("\n")
    assert lines[0] == "Build the `rows` and print them."
    assert "```python" in lines
    start = lines.index("```python")
    assert lines[start + 1] == 'report = build({"a": 1}, mode="``")'
    assert lines[start + 2] == 'print(report["n"])'
    assert lines[start + 3] == "```"
    assert lines[-2] == "See `other` for the \\{inverse\\}, or `a"
    assert lines[-1] == "wrapped name` on the next line."
    assert not any(line.startswith("``python") for line in lines)


def test_doctest_is_wrapped_in_a_python_fence(gen):
    lines = gen._docstring(_doctest_example).split("\n")
    start = lines.index("```python")
    assert lines[start + 1] == '>>> x = count({"a": 1})'
    assert lines[start + 2] == ">>> x"
    assert lines[start + 3] == "1"
    assert lines[start + 4] == "```"
    assert lines[-1] == "Done."
