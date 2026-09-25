"""The SmolDataEnvs recipe: every branch of the reward scores what the
README says, an environment failure is None and never 0, and the gold never
reaches a training row."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import whileai as wai

RECIPE = Path(__file__).resolve().parents[2] / "recipes" / "01-simulate" / "smol-data-envs"


@pytest.fixture(scope="module")
def scored() -> dict[tuple[str, int], dict]:
    # A subprocess, not an import: other recipes also ship run.py and env.py.
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "run.py"],
        cwd=RECIPE,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = (RECIPE / "out" / "offline.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    return {(r["scenario_id"], r["rollout_index"]): r for r in rows}


def test_each_reward_branch(scored: dict[tuple[str, int], dict]) -> None:
    reward = {key: row["reward"] for key, row in scored.items()}
    assert reward[("fx_orders_top_region", 0)] == 1  # right
    assert reward[("fx_orders_top_region", 2)] == 0  # ran, wrong value
    assert reward[("fx_orders_top_region", 3)] == 0  # shell echo of the right answer
    assert "shell" in scored[("fx_orders_top_region", 3)]["reason"]
    assert reward[("fx_orders_kettle_revenue", 1)] == 1  # 5768 within the task's tolerance
    assert reward[("fx_orders_kettle_revenue", 3)] == 0  # crashed
    assert "KeyError" in scored[("fx_orders_kettle_revenue", 3)]["reason"]


def test_missing_table_is_ungraded_not_wrong(scored: dict[tuple[str, int], dict]) -> None:
    rows = [r for (sid, _), r in scored.items() if sid == "fx_returns_count"]
    assert len(rows) == 4
    assert all(r["reward"] is None for r in rows)
    assert wai.pass_at(list(scored.values()), min_k=2).n_groups == 3


def test_gold_stays_out_of_training_rows(
    scored: dict[tuple[str, int], dict], tmp_path: Path
) -> None:
    rows = list(scored.values())
    assert all("5768.04" not in str(r.get("reason")) for r in rows)
    picked = wai.select(rows, mode="rl")
    assert picked.report["groups_selected"] == 2  # the all-pass and ungraded groups drop out
    out = tmp_path / "rl.jsonl"
    picked.export(str(out))
    written = out.read_text(encoding="utf-8")
    assert written
    assert "privileged" not in written
    assert "5768.04" not in written
