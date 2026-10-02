"""The character example's CLI paths, offline.

tests/api/test_character_example.py drives ``run()`` and ``demo()`` in
process. This file runs the commands the README gives, as a user would:
``run.py`` writing ``out/``, ``measure.py`` on two ``holdout.jsonl`` files
from a before and an ``--after`` run, ``measure.py --demo``, and
``from_model_spec.py --spec`` on a local copy of the spec.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "recipes" / "03-select" / "character"


def _offline_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("OPENAI_API_KEY", "WHILEAI_API_KEY", "VLLM_API_KEY"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _cli(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXAMPLE / script), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(cwd),
        env=_offline_env(),
        timeout=600,
    )


@pytest.fixture(scope="module")
def before_and_after(tmp_path_factory):
    root = tmp_path_factory.mktemp("character")
    before = _cli("run.py", "--out", str(root / "before"), "--k", "4", cwd=root)
    after = _cli("run.py", "--out", str(root / "after"), "--k", "4", "--after", cwd=root)
    return root, before, after


def test_run_cli_prints_the_documented_report(before_and_after):
    root, before, _after = before_and_after
    assert before.returncode == 0, before.stderr[-2000:]
    lines = before.stdout.splitlines()
    assert lines[0] == (
        "traits 8 | train 57 prompts x 4 = 228 rows | adversarial 120 | control 24 | spec 35"
    )
    assert lines[1] == "judge reference vs spec labels: agreement 1.00 (n=35, kappa 1.00)"
    assert lines[2].startswith("pass@1 ") and "| mixed prompts " in lines[2]
    assert sum(1 for line in lines if line.startswith("  ") and "pass@1" in line) == 8
    assert any(line.startswith("markers: no_filler ") for line in lines)
    assert any(line.startswith("controls on_task ") for line in lines)
    assert any(line.startswith("corr(reward, reply length) ") for line in lines)
    assert lines[-1].startswith("pairs ") and "| sft " in lines[-1]
    for name in ("rows.jsonl", "pairs.jsonl", "sft.jsonl", "holdout.jsonl", "report.json"):
        assert (root / "before" / name).exists(), name
    report = json.loads((root / "before" / "report.json").read_text(encoding="utf-8"))
    assert report["counts"]["train_rows"] == 228
    assert report["judge_vs_spec"]["agreement"] == 1.0


def test_measure_cli_on_two_holdout_files(before_and_after):
    root, _before, after = before_and_after
    assert after.returncode == 0, after.stderr[-2000:]
    out = _cli(
        "measure.py",
        str(root / "before" / "holdout.jsonl"),
        str(root / "after" / "holdout.jsonl"),
        cwd=root,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    lines = out.stdout.splitlines()
    assert lines[0].startswith("marker:trait: moved_unreplicated (+")
    assert lines[1] == "INCONCLUSIVE (1 eval run a side, rerun to confirm)"
    assert any(line.strip().startswith("marker:on_task ") for line in lines)
    assert any(line.strip().startswith("marker:no_filler ") for line in lines)


def test_measure_demo_cli(tmp_path):
    out = _cli("measure.py", "--demo", cwd=tmp_path)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert out.stdout.splitlines()[0].startswith("marker:trait: moved_unreplicated (+")
    assert "INCONCLUSIVE (1 eval run a side, rerun to confirm)" in out.stdout


def test_measure_cli_needs_two_files_or_demo(tmp_path):
    out = _cli("measure.py", cwd=tmp_path)
    assert out.returncode == 2
    assert "--demo" in out.stderr


_SPEC_MD = """## Be warm {#be_warm authority=guideline}

The assistant is warm.

~~~xml
<user>
how are you
</user>
<comparison>
<assistant> <!-- GOOD -->
good reply
</assistant>
<assistant> <!-- BAD -->
bad reply
</assistant>
</comparison>
~~~
"""


def _write_spec(tmp_path: Path) -> Path:
    spec = tmp_path / "model_spec.md"
    spec.write_text(_SPEC_MD, encoding="utf-8")
    return spec


def test_from_model_spec_reads_a_local_copy(tmp_path):
    spec = _write_spec(tmp_path)
    target = tmp_path / "constitution.json"
    out = _cli(
        "from_model_spec.py",
        "--spec",
        str(spec),
        "--traits",
        "be_warm",
        "--out",
        str(target),
        cwd=tmp_path,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "be_warm" in out.stdout and f"wrote {target}" in out.stdout
    doc = json.loads(target.read_text(encoding="utf-8"))
    assert doc["source"]["license"] == "CC0-1.0"
    (trait,) = doc["traits"]
    assert trait["principle"] == "The assistant is warm."
    assert trait["examples"][0]["good"] == ["good reply"]
    assert trait["examples"][0]["bad"] == ["bad reply"]


def test_from_model_spec_keeps_an_existing_commit_pin(tmp_path):
    """Re-running the example without --commit must not downgrade provenance to null (#155)."""
    spec = _write_spec(tmp_path)
    target = tmp_path / "constitution.json"
    pin = "7f1cf79fcb656c07f77c8d95b6fbc78dc7fac5b6"
    target.write_text(json.dumps({"source": {"commit": pin}, "traits": []}), encoding="utf-8")
    out = _cli(
        "from_model_spec.py",
        "--spec",
        str(spec),
        "--traits",
        "be_warm",
        "--out",
        str(target),
        cwd=tmp_path,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    doc = json.loads(target.read_text(encoding="utf-8"))
    assert doc["source"]["commit"] == pin
    assert pin in out.stdout and "kept" in out.stdout
    (trait,) = doc["traits"]  # the traits were still regenerated
    assert trait["id"] == "be_warm"


def test_from_model_spec_explicit_commit_wins_over_the_existing_pin(tmp_path):
    spec = _write_spec(tmp_path)
    target = tmp_path / "constitution.json"
    target.write_text(json.dumps({"source": {"commit": "old"}, "traits": []}), encoding="utf-8")
    out = _cli(
        "from_model_spec.py",
        "--spec",
        str(spec),
        "--traits",
        "be_warm",
        "--out",
        str(target),
        "--commit",
        "new",
        cwd=tmp_path,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(target.read_text(encoding="utf-8"))["source"]["commit"] == "new"
    assert "kept" not in out.stdout


def test_from_model_spec_says_when_the_commit_is_unresolved(tmp_path):
    spec = _write_spec(tmp_path)
    target = tmp_path / "constitution.json"
    out = _cli(
        "from_model_spec.py",
        "--spec",
        str(spec),
        "--traits",
        "be_warm",
        "--out",
        str(target),
        cwd=tmp_path,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(target.read_text(encoding="utf-8"))["source"]["commit"] is None
    assert "source.commit is null" in out.stderr
