"""Run the smithtune CLI on Modal, in the same pinned image smithtune.Dockerfile builds.

For machines where Docker is not available. Dataset directories live on the
Modal volume `deepagents-review-st`; pull them back with `modal volume get`.
The data-rights acknowledgment is the one you made locally, copied in.

    modal run smithtune_modal.py --cmd "smithtune doctor"
    modal run smithtune_modal.py --cmd "smithtune dataset triage council-v2 --confirm"
    modal volume get deepagents-review-st council-v2 .cache/st/
"""

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess

import modal

HERE = pathlib.Path(__file__).resolve().parent
ACK = pathlib.Path.home() / ".config" / "smithtune" / "data-rights.json"

app = modal.App("deepagents-review-smithtune")
image = (
    modal.Image.from_dockerfile(HERE / "smithtune.Dockerfile")
    .env({"XDG_CONFIG_HOME": "/config", "PYTHONUTF8": "1"})
    .add_local_file(ACK, "/config/smithtune/data-rights.json")
    .add_local_file(HERE / "export_sft.py", "/recipe/export_sft.py")
    .add_local_file(HERE / "rubric_codesigned.md", "/recipe/rubric_codesigned.md")
)
st = modal.Volume.from_name("deepagents-review-st", create_if_missing=True)
KEYS = ("LANGSMITH_API_KEY", "BASETEN_API_KEY")


@app.function(image=image, volumes={"/work": st}, timeout=6 * 3600)
def run(cmd: str, env: dict[str, str]) -> int:
    os.environ.update(env)
    code = subprocess.call(shlex.split(cmd), cwd="/work")
    st.commit()
    return code


@app.local_entrypoint()
def main(cmd: str) -> None:
    env = {k: os.environ[k] for k in KEYS if os.environ.get(k)}
    raise SystemExit(run.remote(cmd, env))
