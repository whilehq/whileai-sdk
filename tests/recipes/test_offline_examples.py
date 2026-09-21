"""Every checked-in example still imports and runs with no key on the machine.

An example is the first thing a coding agent copies, so a broken one is
worse than a missing one. These checks are cheap: a ``--help`` that exits 0
proves every import in the module resolved, which is where examples rot (a
helper file that was never committed, a renamed parameter, a moved import).
"""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "recipes"

# Scripts that parse arguments. ``--help`` runs the whole module body up to
# the parser, so it catches import-time breakage without a real run.
CLI_EXAMPLES = [
    "01-simulate/bring-your-own-agent/run.py",
    "04-train/hosted-loop/run.py",
    "04-train/prime-rl/run.py",
    "04-train/report-run/run.py",
    "04-train/sft/wiring.py",
    "04-train/fireworks/export_fireworks.py",
    "04-train/fireworks/prove.py",
    "03-select/character/from_model_spec.py",
    "03-select/character/measure.py",
    "03-select/character/run.py",
    "05-export/hugging-face/roundtrip.py",
    "05-export/bedrock-import/compare.py",
    "04-train/identity/generate.py",
    "02-measure/pass-at-k/measure.py",
    "02-measure/public-benchmark/run.py",
    "02-measure/is-your-eval-any-good/check_eval.py",
    "02-measure/character-to-the-wall/run.py",
    "02-measure/compare-judges/run.py",
    "02-measure/eval-your-agent/run.py",
    "03-select/prime-intellect-rl/diagnose.py",
    "03-select/prime-intellect-rl/export_prompts.py",
    "03-select/prime-intellect-rl/generate.py",
    "02-measure/reward-hacking/run.py",
    "02-measure/safety-evals/run.py",
    "02-measure/safety-evals-marketplace/live.py",
    "02-measure/safety-evals-marketplace/run.py",
    "03-select/schema/migrate.py",
    "03-select/schema/project.py",
    "04-train/text-to-sql/build.py",
    "04-train/text-to-sql/delta.py",
    "04-train/text-to-sql/distill.py",
    "04-train/text-to-sql/rollout.py",
    "04-train/text-to-sql/train.py",
    "04-train/text-to-sql/sql_verifier.py",
    "04-train/grpo/reward.py",
    "04-train/dpo/pairs.py",
    "04-train/resist-planted-instruction/run.py",
    "04-train/resist-planted-instruction/analyse.py",
    "01-simulate/verifiers/run.py",
    "community/same-entrypoint-before-after/run.py",
    "community/force-the-branch/run.py",
    "community/can-the-judge-be-trusted/run.py",
    "community/hosted-grpo-vs-sft/run.py",
    "community/who-protects-the-holdout/run.py",
    "community/how-much-contamination-survives/run.py",
    "community/the-step-the-course-skips/run.py",
    "community/what-trl-does-with-the-loss-mask/run.py",
    "community/identity-spec-no-unasked-maker-aas/run.py",
]

# Need the ``modal`` client, which is not a dev dependency. Compiled, not run.
NEEDS_MODAL = {
    "04-train/dpo/train_modal.py",
    "04-train/grpo/train_modal.py",
    "05-export/bedrock-import/merge_upload.py",
    # boto3 is the ``whileai[bedrock]`` extra, not a dev dependency; compiled, not run
    "05-export/bedrock-import/presign.py",
    "04-train/sft/train_modal.py",
    "04-train/identity/eval_modal.py",
    "04-train/identity/train_modal.py",
    # text-to-sql: the trainer needs modal, the task writer needs anthropic
    "04-train/text-to-sql/author.py",
    "04-train/text-to-sql/train_grpo_modal.py",
    "04-train/resist-planted-instruction/modal_train_eval.py",
    "04-train/prime-rl/modal_prime_rl.py",
    "community/who-protects-the-holdout/inflation_modal.py",
    "community/the-step-the-course-skips/train_modal.py",
    "community/what-trl-does-with-the-loss-mask/train_modal.py",
    "community/identity-spec-no-unasked-maker-aas/train_modal.py",
    "community/identity-spec-no-unasked-maker-aas/eval_modal.py",
    "community/identity-spec-no-unasked-maker-aas/serve_modal.py",
}


def _offline_env() -> dict[str, str]:
    """The environment of an agent that has not configured anything yet."""
    env = dict(os.environ)
    for key in (
        "OPENAI_API_KEY",
        "WHILEAI_API_KEY",
        "VLLM_API_KEY",
        "WHILEAI_MODEL_URL",
        "WHILEAI_API_URL",
        "HF_TOKEN",
    ):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO)
    # A saved `whileai login` credential would count as a key too.
    env["WHILEAI_HOME"] = str(REPO / "tests" / "fixtures" / "no-such-home")
    return env


def _run(script: Path, *args: str, cwd: Path, timeout: int = 120):
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=_offline_env(),
        timeout=timeout,
    )


def test_every_example_directory_is_tracked_by_git():
    """Every recipe directory on disk is tracked by git (no untracked recipe)."""
    tracked = subprocess.run(
        ["git", "ls-files", "recipes"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=True,
    ).stdout.split()
    # recipes/<step>/<name>: a recipe is the second level
    tracked_dirs = {"/".join(Path(p).parts[1:3]) for p in tracked if len(Path(p).parts) > 3}
    on_disk = {
        f"{step.name}/{d.name}"
        for step in EXAMPLES.iterdir()
        if step.is_dir() and not step.name.startswith((".", "__"))
        for d in step.iterdir()
        if d.is_dir() and not d.name.startswith((".", "__"))
    }
    assert on_disk - tracked_dirs == set(), (
        "recipe directory on disk but not in git; add it (and unignore any data file it needs)"
    )


def test_every_example_module_compiles():
    # recipes/papers/ has its own contract and test (tests/recipes/test_papers.py)
    scripts = sorted(p for p in EXAMPLES.glob("*/*/*.py") if p.parts[-3] != "papers")
    assert scripts, "no recipe scripts found"
    for script in scripts:
        py_compile.compile(str(script), doraise=True)
    listed = set(CLI_EXAMPLES) | NEEDS_MODAL
    on_disk = {p.relative_to(EXAMPLES).as_posix() for p in scripts}
    unlisted = sorted(on_disk - listed)
    # Library modules imported by a CLI script are covered through it.
    recipe = lambda rel: "/".join(rel.split("/")[:2])  # noqa: E731
    for rel in unlisted:
        assert recipe(rel) in {recipe(p) for p in listed}, (
            f"{rel}: new recipe directory with no CLI entry point in this test"
        )


@pytest.mark.parametrize("rel", CLI_EXAMPLES)
def test_cli_example_imports_and_answers_help(rel, tmp_path):
    out = _run(EXAMPLES / rel, "--help", cwd=tmp_path)
    assert out.returncode == 0, f"{rel} --help failed:\n{out.stderr[-2000:]}"
    assert "usage:" in out.stdout


def test_modal_examples_are_listed_not_forgotten():
    for rel in NEEDS_MODAL:
        assert (EXAMPLES / rel).exists(), f"{rel} is listed here but gone from disk"


def test_hosted_loop_without_a_key_names_the_env_var(tmp_path):
    out = _run(EXAMPLES / "04-train/hosted-loop/run.py", cwd=tmp_path)
    assert out.returncode != 0
    message = out.stdout + out.stderr
    assert "WHILEAI_API_KEY" in message, message[-2000:]
    assert "http" in message, message[-2000:]


# Entry points that cannot do anything without a credential. Each must say so
# in one line; --help passing proves only that the imports resolved.
NEEDS_CREDENTIAL = [
    "04-train/hosted-loop/run.py",
    "04-train/fireworks/prove.py",
    "04-train/prime-rl/run.py",
    "05-export/hugging-face/roundtrip.py",
    "03-select/prime-intellect-rl/generate.py",
]


@pytest.mark.parametrize("rel", NEEDS_CREDENTIAL)
def test_missing_credential_is_a_message_not_a_traceback(rel, tmp_path):
    """A traceback is not an error message.

    ``roundtrip.py`` let ``PlatformError`` escape, so the one sentence that
    says how to authenticate arrived under ten frames of stack. The SDK's
    message was already right; the example just had to catch it.
    """
    out = _run(EXAMPLES / rel, cwd=tmp_path, timeout=180)
    message = out.stdout + out.stderr
    assert out.returncode != 0, f"{rel} succeeded with no credential:\n{message[-2000:]}"
    assert "Traceback (most recent call last)" not in message, (
        f"{rel} raised instead of exiting with a message:\n{message[-2000:]}"
    )
    assert any(
        var in message for var in ("WHILEAI_API_KEY", "VLLM_API_KEY", "FIREWORKS_API_KEY")
    ), f"{rel} does not name the env var to set:\n{message[-2000:]}"
