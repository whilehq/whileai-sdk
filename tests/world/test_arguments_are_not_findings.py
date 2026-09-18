"""A caller argument names which record to read, never what the record says.

Two kinds of argument reach ``_invented_record``. A locator (``id``, ``name``,
``date``) picks the record, and the record echoes it: that is how the agent
knows it got the record it asked for. A finding (``status``, ``owner``,
``quantity``, ``amount``) describes the record's state, and only the generated
record decides that. An argument the agent wrote must not come back as a fact
the tool reported, or a rubric that checks the reply against tool output
scores the agent's own guess as grounded.

Every call goes through ``_invented_payload`` so ``_result_kind`` routes it the
way the sandbox does. ``run_tests(status="passed")`` is not a case here: any
``run_*`` tool is a shell result and never reaches the record branch.
"""

import hashlib
import json

import pytest

from whileai.simulations.world.sandbox import (
    _invented_payload,
    _is_locator,
    _result_kind,
)

SEEDS = (0, 7, 11, 42)


def call(tool: str, arguments: dict, n: int = 7) -> dict:
    digest = hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()
    return _invented_payload(tool, arguments, n, digest)


def test_the_examples_reach_the_record_branch():
    """The cases below are real: they route to a record, not a shell result."""
    assert _result_kind("get_ticket", None, {"owner": "alice"}) == "record"
    assert _result_kind("check_inventory", None, {"sku": "SKU-1", "quantity": 500}) == "record"
    assert _result_kind("get_order", None, {"order_id": "A-1", "status": "closed"}) == "record"
    assert _result_kind("run_tests", None, {"path": "tests/", "status": "passed"}) == "shell"


@pytest.mark.parametrize("n", SEEDS)
def test_an_owner_argument_does_not_become_the_owner(n):
    """get_ticket(owner="alice") is a guess; the record names its own owner."""
    honest = call("get_ticket", {"ticket_id": "T-9"}, n)
    asserted = call("get_ticket", {"ticket_id": "T-9", "owner": "alice"}, n)
    assert asserted["owner"] == honest["owner"]
    assert asserted["owner"] != "alice"


@pytest.mark.parametrize("n", SEEDS)
def test_a_quantity_argument_does_not_become_the_stock_level(n):
    """check_inventory(quantity=500) must not report 500 in stock."""
    honest = call("check_inventory", {"sku": "SKU-1"}, n)
    asserted = call("check_inventory", {"sku": "SKU-1", "quantity": 500}, n)
    assert asserted["quantity"] == honest["quantity"]
    assert asserted["quantity"] != 500


@pytest.mark.parametrize("n", SEEDS)
def test_a_status_argument_does_not_become_the_status(n):
    honest = call("get_order", {"order_id": "A-1"}, n)
    asserted = call("get_order", {"order_id": "A-1", "status": "closed"}, n)
    assert asserted["status"] == honest["status"]
    assert asserted["status"] != "closed"


@pytest.mark.parametrize("n", SEEDS)
def test_id_name_and_date_locators_still_echo(n):
    """Reading the locator back is how the agent knows it got the right record.

    A get_user(id="u_42") that answers id 1007 is a worse grounding failure
    than an echoed finding: it is the wrong record.
    """
    assert call("get_user", {"id": "u_42"}, n)["id"] == "u_42"
    assert call("get_project", {"name": "billing"}, n)["name"] == "billing"
    assert call("get_availability", {"date": "2026-09-20"}, n)["date"] == "2026-09-20"
    assert call("get_order", {"order_id": "A-1001"}, n)["order_id"] == "A-1001"


def test_a_locator_and_a_finding_in_one_call_split_the_same_way():
    record = call("get_ticket", {"id": "T-9", "owner": "alice", "status": "closed"})
    assert record["id"] == "T-9"
    assert record["owner"] != "alice"
    assert record["status"] != "closed"


def test_a_key_the_record_lacks_is_still_filled():
    """Unchanged behaviour: arguments fill keys the generated record does not have."""
    record = call("get_order", {"order_id": "A-1001", "carrier": "dhl"})
    assert record["carrier"] == "dhl"


@pytest.mark.parametrize("n", SEEDS)
def test_the_search_branch_follows_the_same_finding_rule(n):
    """A listing filtered by a finding does not stamp that finding on every hit."""
    listing = call("list_orders", {"status": "closed"}, n)
    assert listing["items"]
    assert all(item["status"] != "closed" for item in listing["items"])
    tickets = call("list_tickets", {"owner": "alice"}, n)
    assert all(item["owner"] != "alice" for item in tickets["items"])


def test_which_keys_are_locators():
    assert all(_is_locator(k) for k in ("id", "order_id", "sku", "name", "title", "date"))
    assert not any(_is_locator(k) for k in ("status", "owner", "quantity", "amount"))
