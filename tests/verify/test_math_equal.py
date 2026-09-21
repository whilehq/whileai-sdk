"""MathEqual is Math-Verify. The rows are from 3,840 held-out MATH-500
completions where the string-and-number rule it replaced disagreed with it
(Lambert 2025, chapter Evaluation)."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from whileai.simulations.verify import MathEqual, extract_answer

pytest.importorskip("math_verify")


def _row(text: str, gold: str) -> dict:
    return {"prompt": "", "final_text": text, "privileged": {"reference": gold}}


def _reward(text: str, gold: str):
    return MathEqual()(_row(text, gold))["reward"]


@pytest.mark.parametrize(
    "cand, gold",
    [
        ("\\boxed{\\frac{5}{9}}", "\\frac 59"),
        ("\\boxed{C}", "\\text{(C)}"),
        ("\\boxed{120}", "120^\\circ"),
        ("\\boxed{1, -2}", "-2,1"),
        ("\\boxed{2516_{8}}", "2516_8"),
        ("so \\[ \\boxed{\\frac{\\sqrt{3}}{3}} \\] is the value", "\\frac{\\sqrt{3}}{3}"),
    ],
)
def test_the_correct_answers_the_old_rule_failed(cand, gold):
    assert _reward(cand, gold) == 1


@pytest.mark.parametrize(
    "cand, gold",
    [
        ("\\boxed{6\\sqrt{3}}", "1+2\\sqrt{3}"),  # last digit matched
        ("\\boxed{-9 - 12i}", "1 - 12i"),
        ("\\boxed{\\text{C}}", "\\text{(E)}"),
        ("\\boxed{2156_8}", "2516_8"),
    ],
)
def test_the_wrong_answers_the_old_rule_passed(cand, gold):
    assert _reward(cand, gold) == 0


def test_boxed_with_nested_braces_is_the_answer_not_the_line():
    assert (
        extract_answer("so \\[ \\boxed{\\frac{\\sqrt{3}}{3}} \\] there") == "\\frac{\\sqrt{3}}{3}"
    )


def test_grades_from_worker_threads():
    """Math-Verify's timeout is signal.alarm, main thread only; the engine
    grades in a pool."""
    with ThreadPoolExecutor(4) as ex:
        assert (
            list(ex.map(lambda _: _reward("\\boxed{\\frac{5}{9}}", "\\frac 59"), range(8)))
            == [1] * 8
        )


def test_words_compare_as_symbols_and_an_empty_reference_is_unjudged():
    assert _reward("\\boxed{Tuesday}", "Tuesday") == 1
    assert _reward("\\boxed{Monday}", "Tuesday") == 0
    out = MathEqual()(_row("\\boxed{3}", ""))
    assert out["reward"] is None and "not a math expression" in out["reason"]


def test_missing_math_verify_names_the_install(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_math_verify(name, *a, **k):
        if name == "math_verify":
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_math_verify)
    with pytest.raises(ImportError, match="whileai\\[math\\]"):
        MathEqual()
