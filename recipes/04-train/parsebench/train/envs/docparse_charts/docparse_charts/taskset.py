"""docparse-charts: chart image -> HTML data table(s), scored by ParseBench's chart rules.

One task = one synthetic chart (ChartNet core_permissive, built by train/build_chart_data.py)
plus the production chart prompt (`waiparse.prompts.CHART`). Single turn, no tools. The
reward is `waiparse.rewards.chart_reward`: the share of the task's ChartDataPointRule rules
(value + row/column labels, relative tolerance) that the reply's HTML passes.

How the image reaches the model: the prompt is one `vf.UserMessage` whose content is
`[ImageUrlContentPart(data:image/png;base64,...), TextContentPart(CHART)]`, the same order
the production client sends (waiparse/client.py). In training the orchestrator renders it
with the qwen3.8 renderer, which runs the Qwen3.5-family image processor, expands the
<|image_pad|> tokens, and ships `multi_modal_data` (pixel_values + image_grid_thw) to vLLM
and to the trainer (prime-rl `[model.vlm]`). Same pattern as prime-envs' charxiv/mmk12.
A data URL (not file://) is required: verifiers' network mediation drops non-data URLs.

Data layout (volume docparse-data): <data_dir>/{train,val}.jsonl, rows
{"id","image":"images/<id>.png","rules":[...],"chart_type","library"}. <data_dir> comes
from `data_dir` in the taskset config, else $DOCPARSE_CHARTS_DIR, else /data/train/charts.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import importlib.util
import json
import os
import random
import sys
from functools import cache
from pathlib import Path
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import Field

DEFAULT_DATA_DIR = "/data/train/charts"


@cache
def _waiparse(name: str):
    """Load `waiparse.<name>` from $WAIPARSE_DIR by file path. Importing the `waiparse`
    package runs its __init__ -> pipelines -> agent chain (aiohttp, provider registration),
    none of which the reward needs; rewards.py only needs parse_bench."""
    root = os.environ.get("WAIPARSE_DIR")
    path = Path(root, f"{name}.py") if root else None
    if path is None or not path.exists():
        return importlib.import_module(f"waiparse.{name}")
    spec = importlib.util.spec_from_file_location(f"_waiparse_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def chart_prompt(kind: str = "crop") -> str:
    """ "crop" = prompts.CHART (one chart image); "page" = prompts.CHART_PAGE (a whole page), the
    prompt the agent's full-page chart pass uses."""
    return _waiparse("prompts").CHART_PAGE if kind == "page" else _waiparse("prompts").CHART


def strip_think(text: str) -> str:
    """The renderer already moves reasoning into `reasoning_content`; this guards the
    edge cases (a truncated think block, or a template that leaves it inline) so numbers
    written while thinking can never earn reward."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "<think>" in text:
        return ""
    return text.strip()


def score(reply: str, rules: list[dict[str, Any]]) -> float:
    return float(_waiparse("rewards").chart_reward(strip_think(reply), rules))


MAX_PIXELS = (
    1_600_000  # ~1.6k image tokens at 32 px per merged token; a few ChartNet renders hit 16k
)


def image_data_url(path: Path, max_pixels: int | None = None) -> str:
    import io

    from PIL import Image

    img = Image.open(path)
    cap = max_pixels or MAX_PIXELS
    if img.width * img.height > cap:
        s = (cap / (img.width * img.height)) ** 0.5
        img = img.convert("RGB").resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
    else:
        data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


def prompt_messages(
    image_path: Path, text: str, max_pixels: int | None = None
) -> list[vf.UserMessage]:
    image = vf.ImageUrlContentPart(
        image_url=vf.ImageUrlSource(url=image_data_url(image_path, max_pixels))
    )
    return [vf.UserMessage(content=[image, vf.TextContentPart(text=text)])]


class DocparseChartsData(vf.TaskData):
    # NOTE: not `image`: TaskData.image is the sandbox container image.
    chart_id: str
    rules: list[dict[str, Any]]
    chart_type: str = ""
    library: str = ""


class DocparseChartsTask(vf.Task[DocparseChartsData]):
    @vf.reward(weight=1.0)
    async def rules_passed(self, trace: vf.Trace) -> float:
        # ParseBench rule parsing is CPU work; keep it off the env worker's event loop.
        return await asyncio.to_thread(score, trace.last_reply, self.data.rules)

    @vf.metric
    async def has_table(self, trace: vf.Trace) -> float:
        return float("<table" in strip_think(trace.last_reply).lower())

    @vf.metric
    async def said_none(self, trace: vf.Trace) -> float:
        return float(strip_think(trace.last_reply).upper().startswith("NONE"))

    @vf.metric
    async def num_rules(self, trace: vf.Trace) -> float:
        return float(len(self.data.rules))


def _default_data_dir() -> str:
    return os.environ.get("DOCPARSE_CHARTS_DIR", DEFAULT_DATA_DIR)


class DocparseChartsConfig(vf.TasksetConfig):
    split: Literal["train", "val"] = "train"
    data_dir: str = Field(default_factory=_default_data_dir)
    """Directory holding {split}.jsonl and images/."""
    max_tasks: int | None = None
    """Cap on tasks loaded (after a seeded shuffle). The orchestrator materializes every
    task with its image inlined as base64, so this bounds orchestrator memory."""
    seed: int = 0
    prompt: Literal["crop", "page"] = "crop"
    """"page" for whole-page tasks (data/train/chart_pages), matching the agent's page chart pass."""
    max_pixels: int | None = None
    """Image pixel cap; None = MAX_PIXELS. Pages use ~3.2M, what the agent sends (2048 px long side)."""


class DocparseChartsTaskset(vf.Taskset[DocparseChartsTask, DocparseChartsConfig]):
    def load(self) -> list[DocparseChartsTask]:
        c = self.config
        root = Path(c.data_dir)
        jsonl = root / f"{c.split}.jsonl"
        if not jsonl.exists():
            raise FileNotFoundError(
                f"{jsonl} not found (set DOCPARSE_CHARTS_DIR or taskset.data_dir)"
            )
        rows = [
            json.loads(line)
            for line in jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if c.max_tasks is not None and c.max_tasks < len(rows):
            random.Random(c.seed).shuffle(rows)
            rows = rows[: c.max_tasks]
        text = chart_prompt(c.prompt)
        return [
            DocparseChartsTask(
                DocparseChartsData(
                    idx=i,
                    name=str(r["id"]),
                    prompt=prompt_messages(root / r["image"], text, c.max_pixels),
                    chart_id=str(r["id"]),
                    rules=r["rules"],
                    chart_type=str(r.get("chart_type") or ""),
                    library=str(r.get("library") or ""),
                ),
                c.task,
            )
            for i, r in enumerate(rows)
        ]
