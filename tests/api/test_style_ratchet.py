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
    # (#447) made it exactly thirty, so this pin is the line itself and the
    # next name added has to take one off. See #456.
    "front_door": 30,
    "wide_calls": 27,  # rule 3: public calls with more than MAX_PARAMS parameters
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
