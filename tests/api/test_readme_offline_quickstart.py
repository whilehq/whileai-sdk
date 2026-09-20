"""The first program a reader runs writes a file (#471, #458).

The README's "Sixty seconds, offline" block and the docs landing page carry
the same four calls: ``simulate``, ``grade``, ``select``, ``export``. On 0.82
the fourth raised ``privileged_leak``: ``seeded_agent`` quotes its privileged
block on purpose on its ``leak`` rows, ``select`` kept them and
``export(validate=True)`` refused them. ``select`` now drops them as a gate
(#475). These tests run the pages' fences as written, from the files, so a
rewrite of either block goes through the same check.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import whileai as wai
from whileai.simulations import SEEDED_BEHAVIORS, leak_report

ROOT = Path(__file__).resolve().parents[2]
FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _fence(path: Path, after: str = "") -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index(after) if after else 0
    m = FENCE.search(text, start)
    assert m, f"no python fence in {path} after {after!r}"
    return m.group(1)


PAGES = {
    "readme": (ROOT / "README.md", "## Quick start"),
    "landing": (ROOT / "docs" / "index.mdx", ""),
}


@pytest.mark.parametrize("page", sorted(PAGES))
def test_the_offline_quickstart_exports_a_file(page, tmp_path, monkeypatch, capsys):
    path, after = PAGES[page]
    code = _fence(path, after)
    assert "seeded_agent" in code and "select(mode=" in code, "not the quickstart block"
    if "export(" not in code:
        # the README shows the block through `print(rows)` and then says
        # "`rows.export(path)` writes them trainer-ready"; the reader types it
        code += '\nrows.export("train.jsonl")\n'

    monkeypatch.chdir(tmp_path)
    ns: dict = {"__name__": "__quickstart__"}
    exec(compile(code, str(path), "exec"), ns)  # the page, verbatim
    capsys.readouterr()

    out = tmp_path / "train.jsonl"
    assert out.exists(), "the last step must leave a file behind"
    rows = ns["rows"]
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == len(rows) > 0

    # The stand-in leaked, so this proves the gate and not an empty check.
    leaked = leak_report(ns["data"])
    assert leaked["checked"] and leaked["n_leaked"] > 0
    assert rows.report["privileged_leaks_dropped"] > 0
    assert "privileged leaks dropped" in str(rows)


def test_seeded_agent_labels_the_leak_it_plants():
    """Not a fixture bug (#458, option 1): the quoting is the ``leak``
    behavior, on purpose and on the row, and the other flaws do not quote."""

    @wai.tool
    def get_order(order_id: str) -> dict:
        """Look up an order by id."""
        ...

    kw = dict(tools=[get_order], system_prompt="Help customers with orders.", simulator=False)
    quiet = [b for b in SEEDED_BEHAVIORS if b != "leak"]
    honest = wai.simulate(wai.seeded_agent([get_order], rate=1.0, behaviors=quiet), budget=24, **kw)
    assert leak_report(honest)["n_leaked"] == 0
    leaky = wai.simulate(
        wai.seeded_agent([get_order], rate=1.0, behaviors=["leak"]), budget=24, **kw
    )
    report = leak_report(leaky)
    assert report["n_leaked"] > 0
    flagged = {(h["scenario_id"], h["rollout_index"]) for h in report["leaked"]}
    labeled = {
        (t["scenario_id"], t["rollout_index"]) for t in leaky.trajectories if t.get("seeded")
    }
    assert flagged <= labeled
