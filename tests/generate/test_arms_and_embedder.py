"""Five search arms and hash embedder fallback."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import time

import whileai.simulations as wai
from whileai.simulations.generate.embeddings import HashEmbedder, resolve_embedder

CALENDAR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_events",
            "description": "List calendar events in a date range",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["start_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Create a calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start_date": {"type": "string"},
                },
                "required": ["title", "start_date"],
            },
        },
    },
]
CALENDAR_POLICY = (
    "Always list existing events before creating a duplicate. Report tool failures honestly."
)
_FAKE_WAVES = itertools.count()


def _calendar_agent(message: str) -> dict:
    low = message.lower()
    if "mongolia" in low or "capital" in low:
        return {"steps": [], "final_text": "I only handle calendar tasks."}
    if re.search(r"\d{3,}", message):
        return {
            "steps": [
                {
                    "tool": "create_event",
                    "arguments": {"title": "Meet", "start_date": "2099-01-01"},
                    "result": {"status": "created", "id": "evt_9"},
                }
            ],
            "final_text": "Your event is booked.",
        }
    return {
        "steps": [
            {
                "tool": "list_events",
                "arguments": {"start_date": "2026-03-01"},
                "result": {"status": "not_found"},
            }
        ],
        "final_text": "All set, events listed.",
    }


def _fake_complete(_url, _model, messages, **_kwargs):
    prompt = messages[-1]["content"]
    region_ids = re.findall(r'"region_id":\s*"(sc-[^"]+)"', prompt)
    rnd = int(hashlib.sha256(prompt.encode()).hexdigest()[:6], 16) % 1000
    wave = next(_FAKE_WAVES)
    payload = []
    for i, region_id in enumerate(region_ids[:6]):
        token = wave * 6 + i
        letters = "abcdefghijklmnopqrstuvwxyz"
        topic = (
            f"{letters[(token // 676) % 26]}{letters[(token // 26) % 26]}{letters[token % 26]}topic"
        )
        payload.append(
            {
                "region_id": region_id,
                "message": f"Please check {topic}.",
            }
        )
    payload.append({"region_id": None, "message": f"What is the capital of Mongolia? ({rnd})"})
    payload.append(
        {
            "region_id": None,
            "turns": [f"Need room Mar 12 pass {rnd}", "Actually make it Mar 13 instead"],
        }
    )
    return {"content": json.dumps(payload)}


def test_modal_dead_url_falls_back_to_hash():
    started = time.monotonic()
    embedder = resolve_embedder("modal:http://127.0.0.1:9/dead")
    elapsed = time.monotonic() - started
    assert isinstance(embedder, HashEmbedder)
    assert elapsed < 5.0
    vecs = embedder.embed(["calendar hold", "coding task"])
    assert len(vecs) == 2


def test_hash_embedder_does_not_treat_changed_ids_as_novel():
    embedder = HashEmbedder()
    first, second = embedder.embed(
        [
            "check order 123456 for me",
            "check order 987654 for me",
        ]
    )
    assert first == second


def test_offline_fallback_arms():
    """Without a model: structured, open_ended, behavior_targeted, failure_mutation."""
    data = wai.simulate(
        _calendar_agent,
        tools=CALENDAR_TOOLS,
        policy=CALENDAR_POLICY,
        budget=36,
        seed=4,
        grade="conduct",
        embedder="hash",
        simulator=False,
        concurrency=8,
        advanced={"per_round": 10, "mutate_failures": True},
    )
    arms = {t.get("arm") for t in data.trajectories}
    assert {"structured", "open_ended"} <= arms
    assert "behavior_targeted" in arms
    assert "failure_mutation" in arms


def test_llm_guided_with_mocked_model(monkeypatch, tmp_path):
    monkeypatch.setattr("whileai.simulations.generate.generator.complete", _fake_complete)
    data = wai.simulate(
        _calendar_agent,
        tools=CALENDAR_TOOLS,
        policy=CALENDAR_POLICY,
        budget=48,
        seed=3,
        grade="conduct",
        embedder="hash",
        simulator="vllm:fake@http://127.0.0.1:9",
        concurrency=12,
        until="budget_only",
        mode="adaptive",
        time_budget=None,
        advanced={"per_round": 12, "mutate_failures": True},
    )
    arms = {t.get("arm") for t in data.trajectories}
    assert "llm_guided" in arms
    assert {"structured", "open_ended"} & arms
    for t in data.trajectories:
        prompt = t["prompt"]
        assert not prompt.startswith("Please handle all of")
        assert not prompt.startswith("The exact request I")
        assert not prompt.startswith("User request payload")
    assert len(data.trajectories) == 48
    assert all(t["behavior_signature"] for t in data.trajectories)
    path = str(tmp_path / "calendar_rollout.jsonl")
    data.save(path)
    row = json.loads(open(path).readline())
    assert {"prompt", "steps", "final_text"} <= set(row)
    assert isinstance(row.get("reward"), (int, float))
    # #149: the saved row keeps the search bookkeeping that explains it
    assert row["selection_reason"]
    assert row["arm"]
    assert row["behavior_signature"]
