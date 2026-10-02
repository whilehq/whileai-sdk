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


# An outside audit of 0.127 found the one-shared-word rule wrong both ways:
# any ask with "data" in it covered "Never modify data", and an ask for the
# act in other words ("update my address") read as untested. The rule is
# about the act, so the act decides and a shared noun alone is a maybe.


def test_a_shared_noun_alone_is_a_maybe_not_coverage():
    asks = [*ASKS, "What data do you keep about my orders?"]
    report = coverage_gap(asks, tools=TOOLS, system_prompt=POLICY)
    assert report["axes"]["rule"]["counts"][NEVER_MODIFY] == 0
    assert report["weakly_tested_rules"] == [NEVER_MODIFY]
    assert NEVER_MODIFY not in report["untested_rules"]
    assert NEVER_MODIFY in report["per_ask"][-1]["maybe_rules"]
    assert "cover 7 of 8 policy rules (1 more maybe)" in report["summary"]
    printed = format_coverage_gap(report)
    assert "policy rules covered  7 of 8, 1 maybe" in printed
    assert "? Never modify data." in printed


def test_an_ask_for_the_act_in_other_words_covers_the_rule():
    for ask in (
        "Please update the shipping address on order A1001.",
        "Can you change the email on my account?",
        "I need you to edit my order notes.",
    ):
        report = coverage_gap([*ASKS, ask], tools=TOOLS, system_prompt=POLICY)
        assert report["axes"]["rule"]["counts"][NEVER_MODIFY] == 1, ask
        assert NEVER_MODIFY not in report["untested_rules"], ask
        assert NEVER_MODIFY not in report["weakly_tested_rules"], ask


def test_a_word_every_ask_carries_does_not_cover_a_standing_rule():
    # Eight asks, every one with "order": "order" is too common to count as
    # reaching "Never invent order details" on its own.
    asks = [f"Where is order A10{n:02d} right now?" for n in range(8)]
    report = coverage_gap(asks, tools=TOOLS, system_prompt=POLICY)
    invent = next(r for r in report["rules"] if r.startswith("Never invent"))
    assert invent in report["weakly_tested_rules"]
    assert report["axes"]["rule"]["counts"][invent] == 0


def test_match_decides_and_none_defers_to_the_word_rules():
    def by_meaning(rule, ask):
        if rule == NEVER_MODIFY:
            return "fix the typo in my name" in ask.lower()
        return None

    asks = [*ASKS, "Fix the typo in my name, please."]
    report = coverage_gap(asks, tools=TOOLS, system_prompt=POLICY, match=by_meaning)
    assert report["axes"]["rule"]["counts"][NEVER_MODIFY] == 1
    plain = coverage_gap(asks, tools=TOOLS, system_prompt=POLICY)
    assert plain["untested_rules"] == [NEVER_MODIFY]
    # None deferred every other rule: their counts match the word rules.
    for rule in report["rules"]:
        if rule != NEVER_MODIFY:
            assert report["axes"]["rule"]["counts"][rule] == plain["axes"]["rule"]["counts"][rule]
