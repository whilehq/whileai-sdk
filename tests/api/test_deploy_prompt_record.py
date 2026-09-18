"""#296: a stored row says which deploy prompt it was generated under.

``policy_version`` is ``<model>@<hash>`` and reads as a model id, so a
base rate read off stored rows could not be checked against the prompt
regime that produced it. Every row now carries the hash and the opening
chars in ``lineage``, the run keeps the full text once under that hash,
and ``delta_report`` says when the two arms were made under different
prompts (a base rate under a full policy vs a bare prompt)."""

from __future__ import annotations

import hashlib
import json

from tests.helpers import POLICY, simulate_offline
from whileai.simulations.score.delta import delta_report
from whileai.simulations.score.passat import run_config


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _run(**kw):
    kw.setdefault("budget", 6)
    kw.setdefault("advanced", {"per_round": 6, "mutate_failures": False})
    return simulate_offline(concurrency=1, seed=0, **kw)


def test_every_row_carries_the_deploy_prompt_hash_and_head(tmp_path):
    out = tmp_path / "run.jsonl"
    data = _run(output=str(out))
    assert data.trajectories
    sha = _sha(POLICY)
    for row in data.trajectories:
        assert row["lineage"]["system_prompt_sha"] == sha
        assert row["lineage"]["system_prompt_head"] == POLICY[:120]
        assert row["lineage"]["system_prompt_chars"] == len(POLICY)
        # the same hash policy_version already carried after "@"
        assert row["policy_version"].endswith("@" + sha)
    # the full text once per run, keyed by the hash; the row and this
    # text are the hash's only homes (policy_version's suffix is the same
    # value, and run_config reads lineage)
    assert data.system_prompts == {sha: POLICY}
    assert "system_prompt_sha" not in data.metadata
    # the stamp survives the trip to disk, and the sidecar keeps the text
    on_disk = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert all(r["lineage"]["system_prompt_sha"] == sha for r in on_disk)
    sidecar = json.loads((tmp_path / "run.meta.json").read_text())
    assert sidecar["system_prompts"] == {sha: POLICY}
    assert run_config(on_disk)["prompt_hash"] == sha


def test_scaffold_changes_the_hash_and_the_recorded_text():
    plain = _run()
    scaffolded = _run(scaffold="Be terse.")
    a = plain.trajectories[0]["lineage"]["system_prompt_sha"]
    b = scaffolded.trajectories[0]["lineage"]["system_prompt_sha"]
    assert a != b
    assert scaffolded.trajectories[0]["lineage"]["system_prompt_chars"] == len(
        f"{POLICY}\n\nBe terse."
    )
    # the text on record is what the agent was actually run under
    assert scaffolded.system_prompts[b] == f"{POLICY}\n\nBe terse."
    assert plain.system_prompts[a] == POLICY


def _rows(sha: str, reward_of) -> list[dict]:
    return [
        {
            "scenario_id": f"scn_{t}",
            "prompt": f"situation {t}",
            "reward": reward_of(t, r),
            "steps": [],
            "final_text": "x",
            "policy_version": f"qwen@{sha}",
            "lineage": {"system_prompt_sha": sha, "system_prompt_head": sha[:4]},
        }
        for t in range(8)
        for r in range(2)
    ]


def test_delta_report_warns_when_the_arms_were_made_under_different_prompts():
    full_policy = _rows("aaaa000011112222", lambda t, r: (t + r) % 2)
    bare_prompt = _rows("bbbb000011112222", lambda t, r: 1)
    report = delta_report(full_policy, bare_prompt, target="pass_at_1")
    assert report["config"]["before"]["prompt_hash"] == "aaaa000011112222"
    assert report["config"]["after"]["prompt_hash"] == "bbbb000011112222"
    [note] = [w for w in report["warnings"] if "system prompt" in w]
    assert note.startswith("NOT COMPARABLE: ")
    assert "aaaa000011112222" in note and "bbbb000011112222" in note
    assert "system_prompt=" in note  # the fix, not only the problem
    assert report["not_comparable"] == ["prompt_hash"]
    # the same prompt on both sides says nothing about prompts
    same = _rows("aaaa000011112222", lambda t, r: 1)
    report = delta_report(full_policy, same, target="pass_at_1")
    assert not [w for w in report["warnings"] if "system prompt" in w]
    assert report["not_comparable"] == []
    # a side with no stamp at all is not accused of differing
    quiet = [{k: v for k, v in r.items() if k not in ("lineage", "policy_version")} for r in same]
    report = delta_report(full_policy, quiet, target="pass_at_1")
    assert not [w for w in report["warnings"] if "system prompt" in w]


def test_prompt_hash_reads_the_lineage_stamp_before_policy_version():
    row = {"prompt": "p", "reward": 1, "lineage": {"system_prompt_sha": "cccc000011112222"}}
    assert run_config([row])["prompt_hash"] == "cccc000011112222"
    old = {"prompt": "p", "reward": 1, "policy_version": "m@dddd000011112222"}
    assert run_config([old])["prompt_hash"] == "dddd000011112222"
