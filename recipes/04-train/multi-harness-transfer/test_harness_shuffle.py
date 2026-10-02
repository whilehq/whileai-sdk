"""Offline checks for the per-session spelling: python -m pytest test_harness_shuffle.py -q"""

import copy
import json

from harness_shuffle import draw, restore_response, rewrite_request

TOOLS = [
    {"type": "function", "function": {"name": n, "parameters": {"type": "object", "properties": p, "required": list(p)}}}
    for n, p in [
        ("Bash", {"command": {"type": "string"}}),
        ("Read", {"file_path": {"type": "string"}}),
        ("Write", {"file_path": {"type": "string"}, "content": {"type": "string"}}),
        ("Edit", {"file_path": {"type": "string"}, "old_string": {"type": "string"}}),
    ]
]


def test_same_session_same_spelling_and_names_are_distinct():
    a, b = draw("sess_1", TOOLS), draw("sess_1", TOOLS)
    assert a == b
    assert len(set(a.tools.values())) == len(TOOLS)
    assert len(set(a.args.values())) == len(a.args)


def test_sessions_differ():
    assert len({tuple(sorted(draw(f"s{i}", TOOLS).tools.items())) for i in range(20)}) > 5


def test_round_trip_and_prefix_consistency():
    sp = draw("sess_7", TOOLS)
    req = {"messages": [{"role": "system", "content": "Use the `Bash` tool."}, {"role": "user", "content": "go"}], "tools": copy.deepcopy(TOOLS)}
    rewrite_request(req, sp)
    shown = {t["function"]["name"] for t in req["tools"]}
    assert shown == set(sp.tools.values())
    assert f"`{sp.tools['Bash']}`" in req["messages"][0]["content"]

    # The model answers in the shown spelling; the harness must receive the real one.
    args_shown = json.dumps({sp.args["file_path"]: "/workdir/a.txt", sp.args["content"]: "1"})
    model_msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": sp.tools["Write"], "arguments": args_shown}}]}
    resp = {"choices": [{"message": copy.deepcopy(model_msg)}]}
    restore_response(resp, sp)
    call = resp["choices"][0]["message"]["tool_calls"][0]["function"]
    assert call["name"] == "Write"
    assert json.loads(call["arguments"]) == {"file_path": "/workdir/a.txt", "content": "1"}

    # Next turn: the harness echoes its (real-named) history; the engine must see exactly what the
    # model produced, so the token prefix extends.
    turn2 = {"messages": [*req["messages"], resp["choices"][0]["message"], {"role": "tool", "tool_call_id": "c1", "content": "ok"}], "tools": copy.deepcopy(TOOLS)}
    turn2["messages"][0] = {"role": "system", "content": "Use the `Bash` tool."}
    rewrite_request(turn2, sp)
    assert turn2["messages"][2]["tool_calls"] == model_msg["tool_calls"]
    assert [t["function"]["name"] for t in turn2["tools"]] == [t["function"]["name"] for t in req["tools"]]
