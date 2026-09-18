"""Row-level helpers for the run engine: usability, mutation worth,
coverage keys, stratified prompt picks, and the conversation stub."""

from __future__ import annotations

import concurrent.futures
import json
import re
from typing import Any

from ..data import SimulationData
from ..generate.coverage import cell_key as _cell_key_from_row
from ..generate.coverage import coverage_point
from ..generate.diversity import (
    behavior_tier,
    conversation_features,
    mix_items_by_tier,
    sample_cell_tags,
)
from ..generate.scenarios import SEARCH_ARMS
from ..score.grading import as_dict

_SEARCH_ARMS = dict(SEARCH_ARMS)


_RAW_TOOL_MARKUP = re.compile(r"</?tool_call>", re.I)


_TOOL_SCHEMA_DUMP = re.compile(
    r'"name"\s*:\s*"[^"]+".{0,500}"description"\s*:'
    r'.{0,500}"parameters"\s*:',
    re.I | re.S,
)


#: Why a finished rollout is not a row. ``agent_error``: the callable raised
#: (a cold endpoint, an auth failure, a timeout). ``empty_reply``: the agent
#: came back with no final text. ``tool_markup``: raw ``<tool_call>`` tags or
#: a tool schema dump leaked into the visible text. Only the first is an agent
#: error; the other two arrive without an exception, which is how a run ends
#: at ``degraded=[]`` with rows missing (#303).
LOST_REASONS = ("agent_error", "empty_reply", "tool_markup")


def _unusable_reason(row: dict) -> str | None:
    """One of ``LOST_REASONS`` when the rollout cannot be a row, else None."""
    final = str((row or {}).get("final_text") or "").strip()
    if final.lower().startswith("<agent error"):
        return "agent_error"
    if not final:
        return "empty_reply"
    assistant_text = [final]
    for step in (row or {}).get("steps") or []:
        if isinstance(step, dict) and step.get("text") is not None:
            assistant_text.append(str(step["text"]))
    visible = "\n".join(assistant_text)
    if _RAW_TOOL_MARKUP.search(visible) or _TOOL_SCHEMA_DUMP.search(visible):
        return "tool_markup"
    return None


def _usable_rollout(row: dict) -> bool:
    """Infrastructure and parser failures are not training trajectories."""
    return _unusable_reason(row) is None


def _collect_finished(pending: dict, wait_s: float, *, retry: bool = False):
    """Take finished rollouts. Drop hung slots. Do not write a stub row."""
    results: list[dict] = []
    jobs_for: list = []
    if not pending or wait_s < 0:
        return results, jobs_for
    done, not_done = concurrent.futures.wait(pending, timeout=max(0.0, wait_s))
    for fut in done:
        results.append(fut.result())
        jobs_for.append(pending[fut])
    if not_done and retry:
        done, not_done = concurrent.futures.wait(not_done, timeout=max(0.0, wait_s))
        for fut in done:
            results.append(fut.result())
            jobs_for.append(pending[fut])
    return results, jobs_for


def mutation_worthy(row: dict) -> bool:
    """Re-roll and mutate on tool/sandbox faults. Ignores any score column.

    A step's ``result`` is ``Any`` by the canonical schema, and most real tools
    return text. Six of this package's own adapters do: `from_langchain`,
    `from_openai_agents`, `claude_code` and friends all store
    ``str(...)`` there. Calling ``.get`` on it straight crashed
    ``simulate(agent=...)`` with `'str' object has no attribute 'get'` for every
    one of them. `grading.as_dict` is the shared way to ask a result for a
    field; a plain string simply has no status, which is the right answer.
    """
    if row.get("faults"):
        return True
    for step in row.get("steps") or []:
        if not isinstance(step, dict):
            continue
        status = str(as_dict(step.get("result")).get("status", "")).lower()
        if status in {"error", "timeout", "not_found", "denied", "malformed"}:
            return True
    return False


_mutation_worthy = mutation_worthy  # old private name, kept for imports that still use it


def row_cell_key(row: dict) -> str:
    return _cell_key_from_row(row)


_cell_key = row_cell_key  # old private name, kept for imports that still use it


def record_coverage(
    data: SimulationData,
    trajectories: list[dict],
    *,
    cells: set[str],
    shape_keys: set[str],
    arm_weights: dict | None,
    batch_fresh_rate: float | None = None,
    mean_batch_novelty: float | None = None,
    stopped_because: str | None = None,
) -> None:
    point = coverage_point(
        trajectories,
        cells=cells,
        shape_keys=shape_keys,
        arm_weights=arm_weights,
        batch_fresh_rate=batch_fresh_rate,
        mean_batch_novelty=mean_batch_novelty,
        stopped_because=stopped_because,
    )
    if data.coverage_curve and stopped_because is None:
        prev = data.coverage_curve[-1]
        if (
            prev.get("n_rows") == point["n_rows"]
            and prev.get("batch_fresh_rate") == batch_fresh_rate
        ):
            return
    data.coverage_curve.append(point)


_record_coverage = record_coverage  # old private name, kept for imports that still use it


def _prompt_arm(prompt: str, generator: Any) -> str:
    meta = (
        generator.meta.get(prompt)
        or getattr(generator, "last_candidate_provenance", {}).get(prompt)
        or {}
    )
    arm = str(meta.get("arm") or generator.provenance.get(prompt, "open_ended"))
    return arm if arm in _SEARCH_ARMS else "open_ended"


def _stratified_prompts(
    candidates: list[str], take: int, generator: Any, *, used_situations: set[str]
) -> list[str]:
    """Breadth-first pick: arm quotas, prefer unseen situation keys."""
    if not candidates or take <= 0:
        return []
    by_arm: dict[str, list[str]] = {arm: [] for arm in _SEARCH_ARMS}
    for prompt in candidates:
        by_arm.setdefault(_prompt_arm(prompt, generator), []).append(prompt)

    def sort_key(prompt: str) -> tuple[int, str]:
        meta = generator.meta.get(prompt) or {}
        sk = _situation_key_from_meta(meta, prompt)
        return (0 if sk and sk not in used_situations else 1, prompt)

    for arm in by_arm:
        by_arm[arm].sort(key=sort_key)

    picked: list[str] = []
    seen: set[str] = set()
    for arm in _SEARCH_ARMS:
        for prompt in by_arm.get(arm) or []:
            if prompt not in seen:
                picked.append(prompt)
                seen.add(prompt)
                break
        if len(picked) >= take:
            return picked[:take]
    leftover = [prompt for prompt in candidates if prompt not in seen]
    leftover.sort(key=sort_key)

    def _tier(prompt: str) -> str:
        meta = (
            generator.meta.get(prompt)
            or getattr(generator, "last_candidate_provenance", {}).get(prompt)
            or {}
        )
        assignment = meta.get("assignment") or meta.get("scenario_dimensions") or {}
        return behavior_tier(assignment if isinstance(assignment, dict) else {})

    picked.extend(mix_items_by_tier(leftover, take - len(picked), _tier))
    return picked[:take]


def _row_conversation(meta: dict, prompt: str, default_seed: int) -> dict:
    """Conversation labels already on meta, or the same draw the writer used."""
    ready = meta.get("conversation")
    if isinstance(ready, dict) and ready.get("tier"):
        return {k: v for k, v in ready.items() if v is not None}
    assignment = meta.get("assignment") or meta.get("scenario_dimensions") or {}
    if not isinstance(assignment, dict):
        assignment = {}
    rid = str(meta.get("region_id") or prompt)
    rnd = int(meta.get("round") or 0)
    row_seed = int(meta.get("seed", default_seed))
    tags = sample_cell_tags(row_seed, rnd, rid, assignment)
    from ..generate.generator import _ask_family

    return conversation_features(
        assignment,
        tags,
        ask_family=_ask_family(row_seed, rnd, rid),
        tool=str(assignment.get("tool") or ""),
    )


def _situation_key_from_meta(meta: dict, prompt: str = "") -> str:
    """Coverage cell key for unique-situation dedup."""
    assignment = meta.get("assignment") or meta.get("scenario_dimensions")
    if isinstance(assignment, dict) and assignment:
        return json.dumps(assignment, sort_keys=True, default=str)
    rid = meta.get("region_id")
    if rid:
        return str(rid)
    if prompt:
        return f"prompt:{prompt}"
    return ""
