"""A stopped Modal app is named, not retried.

Modal answers a 404 with `modal-http: invalid function call` for an app that
is not deployed. That is permanent, so the run should say which app is gone
rather than spend its budget re-rolling. On 2026-09-21 a run against a
stopped app sat at "0/1000 rollouts" for 7m14s with no error (#816).
"""

from __future__ import annotations

import pytest

from whileai.simulations.generate import agents

DEAD = "modal-http: invalid function call"


@pytest.mark.parametrize(
    "url,app",
    [
        ("https://zeroproofai--stressd-vllm-serve.modal.run/v1", "stressd-vllm-serve"),
        ("zeroproofai--zeroproof-judge-serve.modal.run", "zeroproof-judge-serve"),
        ("https://zeroproofai--whileai-serve-phi-4.modal.run/v1", "whileai-serve-phi-4"),
    ],
)
def test_the_message_names_the_app_that_is_gone(url, app):
    msg = agents.public_llm_error(f"{url} returned 404: {DEAD}")
    assert app in msg
    assert "is not deployed" in msg
    # it must say retrying cannot help, or the reader waits
    assert "permanent" in msg
    # and it must name the way out that does not depend on us shipping
    assert "vllm:<model>@<your-url>" in msg


def test_a_dead_app_is_not_reported_as_a_dropped_request():
    # HOSTED_DROPPED tells the reader to lower concurrency and wait, which is
    # the wrong advice entirely when the app does not exist
    msg = agents.public_llm_error(f"zeroproofai--stressd-vllm-serve.modal.run returned 404: {DEAD}")
    assert msg != agents.HOSTED_DROPPED
    assert "Lower concurrency" not in msg


def test_a_real_transient_is_still_a_transient():
    # a 500 from a live app is the dropped-request case and keeps its message
    dropped = agents.public_llm_error(
        "zeroproofai--whileai-serve-qwen3-4b.modal.run returned 500: modal-http: internal error"
    )
    assert dropped == agents.HOSTED_DROPPED


def test_an_unrelated_error_passes_through_unchanged():
    assert agents.public_llm_error("connection reset by peer") == "connection reset by peer"


def test_the_detector_needs_both_the_body_and_a_modal_host():
    # the phrase alone, on someone else's endpoint, is not ours to reinterpret
    assert not agents._is_dead_modal_app(f"https://my-vllm.example.com/v1 returned 404: {DEAD}")
    assert agents._is_dead_modal_app(f"zeroproofai--x.modal.run returned 404: {DEAD}")
