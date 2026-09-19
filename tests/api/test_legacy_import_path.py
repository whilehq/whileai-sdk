"""``compat/zeroproof``: the old import names are aliases of ``whileai``, not copies.

The shim is a separate distribution and is not installed in the dev
environment; the test puts its source on ``sys.path`` instead.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import pytest

COMPAT = Path(__file__).resolve().parents[2] / "compat" / "zeroproof"


@pytest.fixture
def compat_path(monkeypatch):
    monkeypatch.syspath_prepend(str(COMPAT))
    for name in list(sys.modules):
        if name == "zeroproof" or name.startswith(("zeroproof.", "zeroproof_simulations")):
            monkeypatch.delitem(sys.modules, name)
    yield
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_AliasFinder"]


def test_zeroproof_is_whileai(compat_path):
    import whileai
    import whileai.simulations as new

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = importlib.import_module("zeroproof")
    assert old is whileai
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert importlib.import_module("zeroproof.simulations") is new
    assert new.__spec__.name == "whileai.simulations"  # alias must not clobber it
    assert importlib.import_module("zeroproof.simulations.run.engine") is importlib.import_module(
        "whileai.simulations.run.engine"
    )


def test_zeroproof_simulations_is_whileai_simulations(compat_path):
    import whileai.simulations as new

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = importlib.import_module("zeroproof_simulations")
        judging = importlib.import_module("zeroproof_simulations.score.judging")
    assert legacy is new
    assert judging is importlib.import_module("whileai.simulations.score.judging")
