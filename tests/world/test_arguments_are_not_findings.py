"""A caller argument names which record to read, never what the record says."""

import hashlib
import json

from whileai.simulations.world.sandbox import _invented_record


def call(tool: str, arguments: dict) -> dict:
    digest = hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()
    return _invented_record(tool, arguments, 7, digest)


def test_an_argument_does_not_overwrite_a_generated_field():
    """The agent cannot manufacture an outcome by naming it in an argument.

    run_tests(status="passed") returning status "passed" is a green test run the
    agent invented, and a rubric checking the reply against tool output scores it
    as grounded because the claim really is in a tool result.
    """
    honest = call("run_tests", {"path": "tests/"})
    asserted = call("run_tests", {"path": "tests/", "status": "passed"})
    assert asserted["status"] == honest["status"]
    assert asserted["status"] != "passed" or honest["status"] == "passed"


def test_the_record_still_echoes_the_identifier_it_was_asked_for():
    """Reading back the locator is how the agent knows it got the right record."""
    record = call("get_order", {"order_id": "A-1001"})
    assert record["order_id"] == "A-1001"


def test_a_key_the_record_lacks_is_still_filled():
    """Unchanged behaviour: arguments fill keys the generated record does not have."""
    record = call("get_order", {"order_id": "A-1001", "carrier": "dhl"})
    assert record["carrier"] == "dhl"


def test_the_single_record_branch_matches_the_search_branch():
    """Both branches of the same function follow one rule."""
    listing = call("list_orders", {"status": "closed"})
    item = listing["items"][0]
    single = call("get_order", {"order_id": "A-1", "status": "closed"})
    # neither branch lets the caller dictate a field the record already carries
    assert item["status"] != "closed" or "status" not in item
    assert (
        single["status"] != "closed"
        or single["status"] == call("get_order", {"order_id": "A-1"})["status"]
    )
