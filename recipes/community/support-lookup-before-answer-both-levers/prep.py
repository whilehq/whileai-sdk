"""Turn published support traces into decision-point tasks."""

import json
from collections import Counter

rows = json.load(open("traces.json"))


def tc(m):
    return m.get("tool_calls") or []


tasks = []
skipped = Counter()
for r in rows:
    ms = r["messages"]
    system = ms[0]["content"] if ms[0]["role"] == "system" else ""
    # find first assistant message carrying a tool call
    idx = next((i for i, m in enumerate(ms) if m["role"] == "assistant" and tc(m)), None)
    if idx is not None:
        call = tc(ms[idx])[0]["function"]
        try:
            args = (
                json.loads(call["arguments"])
                if isinstance(call["arguments"], str)
                else (call["arguments"] or {})
            )
        except Exception:
            args = {}
        target, tool_name, tool_args = "CALL", call["name"], args
        prefix = ms[1:idx]
        target_text = None
    else:
        # reference answered with no tool call: decision point is the last user turn
        last_user = max((i for i, m in enumerate(ms) if m["role"] == "user"), default=None)
        if last_user is None:
            skipped["no_user_turn"] += 1
            continue
        target, tool_name, tool_args = "NO_CALL", None, None
        prefix = ms[1 : last_user + 1]
        nxt = ms[last_user + 1] if last_user + 1 < len(ms) else None
        target_text = (nxt.get("content") or "") if nxt and nxt["role"] == "assistant" else ""
    if not any(m["role"] == "user" for m in prefix):
        skipped["no_user_before_decision"] += 1
        continue
    tasks.append(
        {
            "task_id": f"{r['_cfg']}:{r['scenario_id']}",
            "domain": r["_cfg"],
            "system": system,
            "tools": r["tools"],
            "prefix": prefix,
            "target": target,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "ask_family": r["ask_family"],
            "tier": r["tier"],
            "grader_reason": r["grader_reason"],
            "target_text": target_text,
        }
    )

print("tasks", len(tasks), "skipped", dict(skipped))
print("target", Counter(t["target"] for t in tasks))
print("tier x target", Counter((t["tier"], t["target"]) for t in tasks).most_common())
print("tools per task", Counter(len(t["tools"]) for t in tasks).most_common(5))
print("distinct tool names", len(Counter(t["tool_name"] for t in tasks if t["tool_name"])))
print("top targets", Counter(t["tool_name"] for t in tasks if t["tool_name"]).most_common(10))
# duplicate task_ids?
ids = Counter(t["task_id"] for t in tasks)
print("dup task_ids", sum(1 for v in ids.values() if v > 1))
json.dump(tasks, open("tasks_all.json", "w"))
