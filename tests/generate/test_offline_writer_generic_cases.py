"""The offline writer on a non-support agent (gentlyventures case study,
whileai 0.126): 94 of 115 code-history asks carried an invented support
reference (REF-nnnn) and nothing said the cases were a generic template.
"""

import re

import whileai as wai

POLICY = (
    "You answer questions about a git repository's commit history: commit counts "
    "per month, which files change most often, and who last touched a file."
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Run git log on the repository.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "since": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_commits",
            "description": "Count commits in a month.",
            "parameters": {"type": "object", "properties": {"month": {"type": "string"}}},
        },
    },
]
TICKET_ID = re.compile(r"\b[A-Z]{2,4}-\d{4}\b")
GENERIC = "generic template"


def _agent(message):
    return {"steps": [], "final_text": "ok"}


def _run(**kw):
    return wai.simulate(
        _agent, tools=TOOLS, system_prompt=POLICY, simulator=False, budget=115, seed=0, **kw
    )


def test_offline_writer_without_seeds_warns_and_invents_no_ticket_ids():
    data = _run()
    prompts = [str(r.get("prompt") or "") for r in data.rows()]
    with_ids = sum(bool(TICKET_ID.search(p)) for p in prompts)
    print(f"ticket ids: {with_ids} of {len(prompts)}")
    assert len(prompts) >= 100
    assert with_ids == 0
    assert any(GENERIC in w for w in data.warnings)


def test_offline_writer_with_seeds_does_not_warn():
    data = _run(seeds=["How many commits landed in March 2026?"])
    assert not any(GENERIC in w for w in data.warnings)


def test_offline_writer_keeps_ids_the_tools_name():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "description": "Look up an order. Orders on file: A1001, A1002.",
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
            },
        }
    ]
    data = wai.simulate(_agent, tools=tools, simulator=False, budget=40, seed=0)
    prompts = [str(r.get("prompt") or "") for r in data.rows()]
    assert any("A1001" in p or "A1002" in p for p in prompts)
