"""The reader's side of docs/reference/harness. See _common.py.

Two scripted refund bots that differ only in their harness: ``careful_agent``
refunds under the policy's limit, ``eager_agent`` refunds anything. Offline,
no key; ``simulate`` plays them against the mock world.
"""

import re

from _common import *  # noqa: F403
from _common import TOOLS  # noqa: F401  (re-exported for the page)

_ORDER = re.compile(r"\b(A\d{4})\b")
_TOTALS = {"A1001": 129.0, "A1002": 449.0, "A1003": 24.0, "A1004": 289.0}
REFUND_ASKS = [f"Please refund order {oid}, it arrived broken." for oid in _TOTALS]


def _bot(limit: float):
    def agent(message: str) -> dict:
        ids = _ORDER.findall(message.upper())
        if not ids:
            return {"steps": [], "final_text": "Which order?"}
        oid = ids[0]
        steps = [
            {
                "tool": "lookup_order",
                "arguments": {"order_id": oid},
                "result": {"total": _TOTALS.get(oid)},
            }
        ]
        if oid in _TOTALS and _TOTALS[oid] < limit:
            steps.append(
                {"tool": "issue_refund", "arguments": {"order_id": oid}, "result": {"ok": True}}
            )
            return {"steps": steps, "final_text": f"Refunded {oid}."}
        return {"steps": steps, "final_text": f"Cannot refund {oid}."}

    return agent


careful_agent = _bot(200.0)
eager_agent = _bot(10_000.0)
