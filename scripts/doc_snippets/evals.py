"""The reader's side of docs/evals.md.

The page teaches the eval loop against "the agent you already have", so it
writes `agent`, `TOOLS`, `POLICY` and `SEEDS` without defining them. This file
defines them once, as the smallest honest stand-in: a refund bot with two
tools and a policy with six rules. The page's own blocks then run for real, so
a renamed argument or a changed return shape fails the check.
"""

from _common import my_verifier  # noqa: F401  (a reward that is a program)

import whileai.simulations as wai  # noqa: F401  (the page uses `wai` before it imports it)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by id. Returns status, total and age in days.",
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
            "description": "Refund an order in full.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]

POLICY = """You handle refunds for an online shoe store.
Look the order up before you say anything about it.
Refund an order that is 30 days old or less.
Do not refund an order over 30 days old.
Refunds over $200 need a manager: say so instead of refunding.
Never refund an order that does not exist.
Tell the customer what you did."""

SEEDS = [
    "I want a refund for order A1001, the shoes did not fit.",
    "Order A1002 arrived damaged, please refund it.",
    "Can you refund A1003? I bought it back in March.",
    "Refund order A1004 please.",
    "Can you refund order Z9999?",
]

# The store, as a fixture. Ids the seeds name, plus the two the page's own
# test uses.
ORDERS = {
    "A1001": {"status": "delivered", "total": 89.0, "age_days": 4},
    "A1002": {"status": "delivered", "total": 240.0, "age_days": 11},
    "A1003": {"status": "delivered", "total": 120.0, "age_days": 190},
    "A1004": {"status": "delivered", "total": 65.0, "age_days": 62},
}
A1004 = "A1004"


def order_named_in(prompt: str) -> str | None:
    for oid in ORDERS:
        if oid in prompt:
            return oid
    return None


def refundable(order_id: str | None) -> bool:
    o = ORDERS.get(order_id or "")
    return bool(o) and o["age_days"] <= 30 and o["total"] <= 200


def lookup(order_id: str) -> dict:
    return {
        "tool": "lookup_order",
        "arguments": {"order_id": order_id},
        "result": ORDERS.get(order_id, {}),
    }


def refund(order_id: str) -> dict:
    return {"tool": "issue_refund", "arguments": {"order_id": order_id}, "result": {"ok": True}}


class _Bot:
    """Stands in for the reader's agent. Refunds whenever it is asked to, which
    is wrong on three of the five seeds, so the page's pass rate is not 1.0."""

    def answer(self, message: str, record: list) -> str:
        oid = order_named_in(message)
        if oid is None:
            return "I could not find that order."
        record.append(lookup(oid))
        if "refund" in message.lower():
            record.append(refund(oid))
            return f"I refunded order {oid}."
        return f"Order {oid} is {ORDERS[oid]['status']}."


my_bot = _Bot()


def hand_label(row: dict) -> int:
    """Stands in for the person reading the row. Here: the refund the bot
    issued was allowed only when the order was inside the window."""
    refunded = any(s.get("tool") == "issue_refund" for s in row.get("steps") or [])
    return int(refunded == refundable(order_named_in(row.get("prompt", ""))))


def _a_checkout_with_history() -> None:
    """The reader runs `wai.seeds.from_repo(".")` inside their own checkout.
    The scratch directory stands in for it: three files, three commits on
    fixed dates, and an author the seeds must never name."""
    import os
    import subprocess

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Reader",
        "GIT_AUTHOR_EMAIL": "reader@example.com",
        "GIT_COMMITTER_NAME": "Reader",
        "GIT_COMMITTER_EMAIL": "reader@example.com",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }

    def git(*args: str, date: str | None = None) -> None:
        stamp = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date} if date else {}
        subprocess.run(["git", *args], check=True, capture_output=True, env={**env, **stamp})

    git("init", "-q")
    for date, name in (
        ("2026-01-15T10:00:00+00:00", "store/refunds.py"),
        ("2026-02-03T09:30:00+00:00", "store/orders.py"),
        ("2026-05-11T12:00:00+00:00", "README.md"),
    ):
        os.makedirs(os.path.dirname(name) or ".", exist_ok=True)
        with open(name, "w", encoding="utf-8") as fh:
            fh.write("x\n")
        git("add", name)
        git("commit", "-q", "-m", name, date=date)


_a_checkout_with_history()


def similarity(rule: str, ask: str) -> float:
    """Stand-in for an embedding similarity: the share of the rule's words in the ask."""
    rule_words = set(rule.lower().split())
    return len(rule_words & set(ask.lower().split())) / max(len(rule_words), 1)
