"""The style ratchet: the public surface may shrink, never grow.

``docs/reference/style.md`` sets the coding standard (PyTorch/DSPy
ergonomics: few names, objects carry configuration, calls carry data,
reports print themselves). This test pins today's counts for the shapes
the standard retires and fails when any of them grows. Lower a pin when
you retire a name; never raise one without saying why in the PR body.

Run ``uv run pytest tests/api/test_style_ratchet.py -q -rA`` to print
the current counts next to the pins.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

import whileai
import whileai.simulations as wai

# Rule 3: a public call takes at most this many parameters.
MAX_PARAMS = 8

# Today's counts. Each is a ceiling; the test names the rule it guards.
PINS = {
    "exports": 208,  # rule 1: names in whileai.simulations.__all__
    # rule 1: names in whileai.__all__. The rule says "under thirty"; `tool`
    # (#447) made it exactly thirty, and `rows` (#613, the front door for a
    # public benchmark) thirty-one by the maintainer's call. The next name
    # added takes one off (`Fireworks` took `Settings` off, `Harness` took
    # `Selection` off, #712). See #456.
    # 33, was 31: `OPD` and `prime_rl_config` were reachable only as
    # `wai.OPD`/`wai.prime_rl_config` (lazy, `whileai._LAZY`) and documented
    # in `methods.py`'s module docstring, but absent from the front door
    # itself, so a caller (or a coding agent) reading `whileai.__all__` never
    # learned they existed. Method-routing task, 2026-09-24: raised on
    # purpose, not silently; no name was retired to make room this time.
    "front_door": 33,
    "wide_calls": 27,  # rule 3: public calls with more than MAX_PARAMS parameters
    # rule 3, the other direction. `wide_calls` counts HOW MANY calls are over
    # the cap and says nothing about how far over, so a call that is already
    # failing can keep taking arguments without moving a pin. Between #459
    # being filed (2026-09-18) and 0.123, `simulate` went 43 -> 45,
    # `delta_report` 17 -> 18 and `judge_trust` 14 -> 15, and the count stayed
    # at 27 the whole time. These two pin the size of the problem, not just
    # its shape: the widest signature, and the total number of parameters over
    # the cap across all of them.
    "widest_call": 45,  # parameters on the widest public call (`simulate`)
    # 195, was 194: #842 measured the pin at its branch point and #843 landed
    # `compare(lower_is_better=)` after it, so the two were never counted
    # together. The argument is a correctness fix (a metric a run set out to
    # reduce read as a regression), not new surface for its own sake. Rule 3's
    # own answer is a typed options object; that refactor is #859.
    "wide_call_overage": 195,  # sum of (params - MAX_PARAMS) over every wide call
    "format_twins": 12,  # rule 5: format_* functions instead of __str__ on a report
    "bare_returns": 3,  # rule 5: front-door calls returning a bare dict or tuple
    "in_place_mutators": 6,  # rule 4: attach_* / stamp_* free functions over rows
    "implementation_names": 17,  # rule 6: build_/load_/run_ prefixes, _of/_rows suffixes
}

IMPLEMENTATION_PREFIXES = ("build_", "load_", "run_")
IMPLEMENTATION_SUFFIXES = ("_of", "_rows")


def _public_callables() -> dict[str, object]:
    """Public functions and classes a user calls. Dataclasses are records
    (a row schema, a result, a typed options object); their field count is
    not a call's parameter count, so they are not held to rule 3."""
    out: dict[str, object] = {}
    for n in wai.__all__:
        obj = getattr(wai, n)
        if callable(obj) and not dataclasses.is_dataclass(obj):
            out[n] = obj
    return out


def _param_count(obj: object) -> int:
    try:
        sig = inspect.signature(obj)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return sum(
        1
        for p in sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and p.name != "self"
    )


def _bare_return_calls() -> list[str]:
    """Front-door calls whose return annotation is a bare ``dict`` or
    ``tuple``: rule 5 says a measurement is an object that prints itself,
    not a mapping the user has to know the keys of."""
    out: list[str] = []
    for n in whileai.__all__:
        obj = getattr(whileai, n)
        if not callable(obj) or isinstance(obj, type):
            continue
        try:
            ann = inspect.signature(obj).return_annotation
        except (TypeError, ValueError):
            continue
        if isinstance(ann, str) and ann.startswith(("dict", "tuple")):
            out.append(n)
    return sorted(out)


def _counts() -> dict[str, int]:
    names = list(wai.__all__)
    calls = _public_callables()
    return {
        "exports": len(names),
        "front_door": len(whileai.__all__),
        "bare_returns": len(_bare_return_calls()),
        "wide_calls": sum(1 for obj in calls.values() if _param_count(obj) > MAX_PARAMS),
        "widest_call": max((_param_count(obj) for obj in calls.values()), default=0),
        "wide_call_overage": sum(
            _param_count(obj) - MAX_PARAMS
            for obj in calls.values()
            if _param_count(obj) > MAX_PARAMS
        ),
        "format_twins": sum(1 for n in names if n.startswith("format_")),
        "in_place_mutators": sum(1 for n in names if n.startswith(("attach_", "stamp_"))),
        "implementation_names": sum(
            1
            for n in names
            if n.startswith(IMPLEMENTATION_PREFIXES) or n.endswith(IMPLEMENTATION_SUFFIXES)
        ),
    }


@pytest.mark.parametrize("key", sorted(PINS))
def test_surface_does_not_grow(key: str) -> None:
    now = _counts()[key]
    pin = PINS[key]
    if key in ("widest_call", "wide_call_overage"):
        assert now <= pin, (
            f"{key}: {now}, pin is {pin}. docs/reference/style.md rule 3 caps a public "
            f"call at {MAX_PARAMS} parameters. A call already over the cap does not get "
            f"to keep growing: put the new argument on a typed options object (rule 2) "
            f"or take one off. test_wide_calls_are_named prints every call and its count."
        )
    else:
        assert now <= pin, (
            f"{key}: {now} public names, pin is {pin}. docs/reference/style.md retires this "
            f"shape; put the new name one dot down, or make it a method on the rows object."
        )
    if now < pin:
        pytest.fail(
            f"{key}: {now} < pin {pin}. Good: lower PINS[{key!r}] to {now} in this PR "
            f"so the ratchet holds the gain.",
            pytrace=False,
        )


def test_reports_print_themselves() -> None:
    """Rule 5: the measurements that have a ``format_*`` twin return an
    object whose ``str`` is that block, so ``print(report)`` is the report
    and not a dict literal. They are still dicts, so every key a caller
    reads today keeps working."""
    from whileai.report import Report
    from whileai.simulations.score.delta import DeltaReport, format_delta_report
    from whileai.simulations.score.hack_scan import HackScanReport, format_hack_scan
    from whileai.simulations.score.judge_trust import JudgeTrustReport, format_judge_trust
    from whileai.simulations.score.privileged import (
        LeakReport,
        format_leak_report,
        leak_report,
    )

    rows = [
        {"scenario_id": "a", "rollout_index": i, "reward": i % 2, "final_text": "hi"}
        for i in range(8)
    ]
    for report, formatter in (
        (whileai.judge_trust(rows), format_judge_trust),
        (whileai.hack_scan(rows), format_hack_scan),
        (whileai.compare(rows, rows), format_delta_report),
        (leak_report(rows), format_leak_report),
    ):
        assert isinstance(report, Report), type(report)
        assert isinstance(report, dict)  # keys still read the same
        assert str(report) == formatter(report)
        assert str(report) != repr(dict(report))
        assert report._repr_html_().startswith("<pre>")

    assert issubclass(JudgeTrustReport, Report)
    assert issubclass(HackScanReport, Report)
    assert issubclass(DeltaReport, Report)
    assert issubclass(LeakReport, Report)


def test_bare_returns_are_named() -> None:
    """The front-door calls still handing back a bare dict or tuple."""
    assert _bare_return_calls() == ["decontaminate", "export", "preflight"]


def test_wide_calls_are_named() -> None:
    """The calls over the cap, so a reviewer sees which ones a PR should split."""
    wide = sorted(
        (n, _param_count(obj))
        for n, obj in _public_callables().items()
        if _param_count(obj) > MAX_PARAMS
    )
    assert len(wide) <= PINS["wide_calls"], wide
    # The widest one is the migration's target: style.md's migration section
    # puts `simulate`'s kwargs behind `world=`, `sampling=` and `budget=`.
    widest = max(wide, key=lambda pair: pair[1])
    assert widest[1] <= PINS["widest_call"], widest


def test_the_standard_quotes_the_numbers_the_ratchet_holds() -> None:
    """``docs/reference/style.md`` has a "What the ratchet checks" table and
    a rule 3 sentence naming ``simulate``'s parameter count. Both are written
    by hand and both had drifted by 0.123: the table said 212 exports and 13
    ``format_*`` twins against 208 and 12, and rule 3 said forty-three
    parameters against forty-five. #459 and #460 were filed off those numbers
    and arrived stale. The standard quotes the test now, and this fails when
    it stops."""
    import re
    from pathlib import Path

    from tests.api.test_alias_surface import LEGACY_IMPORTS

    page = (Path(__file__).resolve().parents[2] / "docs" / "reference" / "style.md").read_text(
        encoding="utf-8"
    )

    rows = {
        "names in `whileai.simulations.__all__`": PINS["exports"],
        "names in `whileai.__all__` (the front door)": PINS["front_door"],
        "front-door calls returning a bare `dict` or tuple": PINS["bare_returns"],
        "public calls or constructors with more than 8 parameters": PINS["wide_calls"],
        "parameters on the widest public call": PINS["widest_call"],
        "parameters over the cap, summed across": PINS["wide_call_overage"],
        "public names starting `format_`": PINS["format_twins"],
        "public names starting `attach_` or `stamp_`": PINS["in_place_mutators"],
        "public names starting `build_`, `load_`, `run_`": PINS["implementation_names"],
        "imports that are not `import whileai as wai`": LEGACY_IMPORTS,
    }
    wrong = []
    for label, pin in rows.items():
        line = next((ln for ln in page.split("\n") if ln.startswith("|") and label in ln), None)
        if line is None:
            wrong.append(f"{label!r}: no row in the table")
            continue
        printed = int(line.rstrip("| ").rsplit("|", 1)[-1].strip())
        if printed != pin:
            wrong.append(f"{label!r}: the page says {printed}, the pin is {pin}")
    assert not wrong, "docs/reference/style.md drifted from the pins:\n" + "\n".join(wrong)

    words = {8: "eight", 43: "forty-three", 44: "forty-four", 45: "forty-five", 46: "forty-six"}
    said = re.search(r"`simulate`\s*\ntakes ([a-z-]+) today", page)
    assert said, "rule 3 no longer names simulate's parameter count"
    assert said.group(1) == words.get(PINS["widest_call"], "?"), (
        f"rule 3 says simulate takes {said.group(1)}; the pin is {PINS['widest_call']}"
    )
