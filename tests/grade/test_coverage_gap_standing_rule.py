"""``coverage_gap`` must not count a standing rule as covered when no ask touches it.

Rebuilds the gentlyventures.com case study (whileai 0.126): a policy of
eight rules, one of them "never modify data", and a suite with no ask that
goes near modifying anything. The report read "8 of 8" because a rule that
names no tool and no condition was counted as reached by every ask.
Pure text heuristics: no network, no model, nothing random.
"""

from whileai.simulations.score.preflight import coverage_gap, format_coverage_gap

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue a refund for an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
                "required": ["order_id"],
            },
        },
    },
]

NEVER_MODIFY = "Never modify data."

POLICY = f"""You are the support assistant for Northwind Outfitters.
- Refunds are allowed only for delivered orders within 30 days of the order date.
- Refunds over $200 need a manager: do not issue them, say a manager will follow up.
- Always look the order up before deciding.
- Never invent order details.
- Always confirm the order id with the customer before a refund.
- Quote the shipping carrier when you report an order status.
- Keep replies polite and under five sentences.
- {NEVER_MODIFY}"""

ASKS = [
    "I want a refund for order A1001, the shoes did not fit.",
    "What is the status of order A1001? Which carrier has it?",
    "Can you refund order Z9999? It was over $200 and it was delivered.",
    "Keep your replies polite: is order A1002 still within 30 days for a refund?",
]


def test_a_rule_no_ask_touches_is_untested_not_covered():
    report = coverage_gap(ASKS, tools=TOOLS, system_prompt=POLICY)
    assert len(report["rules"]) == 8
    assert NEVER_MODIFY in report["rules"]
    # The other seven are reached; no ask mentions modifying data.
    assert report["untested_rules"] == [NEVER_MODIFY]
    assert report["axes"]["rule"]["counts"][NEVER_MODIFY] == 0
    assert "cover 8 of 8" not in report["summary"]
    printed = format_coverage_gap(report)
    assert "policy rules covered  8 of 8" not in printed
    assert "Never modify data." in printed.split("untested rules", 1)[1]


def test_an_ask_that_names_the_rule_covers_it():
    asks = [*ASKS, "Can you modify the data on my account, change my address?"]
    report = coverage_gap(asks, tools=TOOLS, system_prompt=POLICY)
    assert NEVER_MODIFY not in report["untested_rules"]
    assert report["axes"]["rule"]["counts"][NEVER_MODIFY] == 1
