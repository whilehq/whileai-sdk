# Detector-layout iteration without re-running the VLM.
#
#   modal run serve/relayout.py::attach --src wai_agent --dst wai_agent_det_off
#       copy <src> raw results (layout group), add PP-DocLayoutV3 boxes per page (page["det"])
#   modal run serve/relayout.py::score --pipeline wai_agent_det_t40 --base wai_agent_det_off
#       re-normalize <base> raws with <pipeline>'s config, evaluate the layout group, print metrics
#
# Markdown is emitted from the same layout-pass items either way, so only layout_pages change.

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

# Your Modal workspace and any whole-URL overrides (waiparse/endpoints.py), baked into the
# image so the remote side resolves the same servers your shell would.
ENDPOINT_ENV = {
    k: os.environ[k]
    for k in (
        "DOCPARSE_WORKSPACE",
        "DOCPARSE_SERVER",
        "DOCPARSE_TUNED_SERVER",
        "DOCPARSE_LAYOUT_URL",
    )
    if os.environ.get(k)
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0", "poppler-utils")
    .uv_pip_install(
        "parse-bench[runners,fast] @ git+https://github.com/run-llama/ParseBench.git",
        "pypdfium2",
        "aiohttp",
    )
    .env({"PYTHONPATH": "/root/pkg", **ENDPOINT_ENV})
    .add_local_dir(ROOT / "waiparse", "/root/pkg/waiparse")
)
data = modal.Volume.from_name("docparse-data")
runs = modal.Volume.from_name("docparse-runs")
app = modal.App("docparse-relayout")


def _rename(raw: dict, name: str, cfg: dict) -> dict:
    raw["pipeline_name"] = name
    raw["pipeline"]["pipeline_name"] = name
    raw["pipeline"]["config"] = cfg
    raw["raw_output"].setdefault("_config", {}).update(cfg)
    return raw


@app.function(
    image=image,
    volumes={"/data": data, "/runs": runs},
    timeout=3 * 3600,
    cpu=8,
    memory=16384,
    secrets=[modal.Secret.from_name("docparse-vllm-key")],
)
def attach_remote(src: str, dst: str, split: str, group: str) -> str:
    import asyncio
    import sys

    import aiohttp

    sys.path.insert(0, "/root/pkg")
    os.environ["DOCPARSE_KEY"] = os.environ["VLLM_API_KEY"]
    from waiparse.layout_det import detect
    from waiparse.render import load_pages

    runs.reload()
    sdir = Path(f"/runs/bench/{src}/{split}/{src}/{group}")
    ddir = Path(f"/runs/bench/{dst}/{split}/{dst}/{group}")
    ddir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in sdir.glob("*.raw.json") if not p.name.endswith(".error.raw.json"))
    cfg = {"layout_boxes": "detector"}

    async def one(s, sem, p):
        async with sem:
            raw = json.loads(p.read_text())
            src_pdf = raw["request"]["source_file_path"]
            local = Path("/data") / src_pdf[src_pdf.index(f"split_{split}") :]
            pages = await asyncio.to_thread(load_pages, local, 200)
            dets = await asyncio.gather(*(detect(s, pg) for pg in pages))
            for page, det in zip(raw["raw_output"].get("pages", []), dets):
                page["det"] = det
            (ddir / p.name).write_text(json.dumps(_rename(raw, dst, cfg)))
            return sum(len(d["boxes"]) for d in dets), sum("error" in d for d in dets)

    async def main():
        sem = asyncio.Semaphore(16)
        async with aiohttp.ClientSession() as s:
            return await asyncio.gather(*(one(s, sem, p) for p in files))

    res = asyncio.run(main())
    runs.commit()
    return f"{len(files)} docs, {sum(r[0] for r in res)} boxes, {sum(r[1] for r in res)} page errors -> {ddir}"


@app.local_entrypoint()
def attach(
    src: str = "wai_agent",
    dst: str = "wai_agent_det_off",
    split: str = "dev",
    group: str = "layout",
):
    print(attach_remote.remote(src, dst, split, group))


@app.function(
    image=image,
    volumes={"/data": data, "/runs": runs},
    timeout=3 * 3600,
    cpu=8,
    memory=16384,
    secrets=[modal.Secret.from_name("docparse-vllm-key")],
)
def ocr_remote(src: str, dst: str, split: str, group: str) -> str:
    """Add page["det"]["ocr"] (region re-reads, waiparse.layout_det.ocr_regions) to <src> raws."""
    import asyncio
    import sys

    import aiohttp

    sys.path.insert(0, "/root/pkg")
    os.environ["DOCPARSE_KEY"] = os.environ["VLLM_API_KEY"]
    from waiparse.client import VLM  # server: waiparse.endpoints.vlm_server()
    from waiparse.layout_det import ocr_regions
    from waiparse.render import load_pages

    runs.reload()
    sdir = Path(f"/runs/bench/{src}/{split}/{src}/{group}")
    ddir = Path(f"/runs/bench/{dst}/{split}/{dst}/{group}")
    ddir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in sdir.glob("*.raw.json") if not p.name.endswith(".error.raw.json"))
    vlm = VLM()

    async def one(s, sem, p):
        async with sem:
            raw = json.loads(p.read_text())
            src_pdf = raw["request"]["source_file_path"]
            local = Path("/data") / src_pdf[src_pdf.index(f"split_{split}") :]
            pages = await asyncio.to_thread(load_pages, local, 200)
            n = 0
            for img, page in zip(pages, raw["raw_output"].get("pages", [])):
                if page.get("det", {}).get("boxes"):
                    page["det"]["ocr"] = await ocr_regions(vlm, s, img, page["items"], page["det"])
                    n += len(page["det"]["ocr"])
            (ddir / p.name).write_text(
                json.dumps(_rename(raw, dst, {"layout_boxes": "detector", "det_ocr": True}))
            )
            return n

    async def main():
        sem = asyncio.Semaphore(24)
        async with aiohttp.ClientSession() as s:
            return await asyncio.gather(*(one(s, sem, p) for p in files))

    res = asyncio.run(main())
    runs.commit()
    return f"{len(files)} docs, {sum(res)} regions re-read -> {ddir}"


@app.local_entrypoint()
def ocr(
    src: str = "wai_agent_det_off",
    dst: str = "wai_agent_det_ocr_off",
    split: str = "dev",
    group: str = "layout",
):
    print(ocr_remote.remote(src, dst, split, group))


@app.function(
    image=image, volumes={"/data": data, "/runs": runs}, timeout=3600, cpu=8, memory=16384
)
def score_remote(pipeline: str, base: str, split: str, group: str) -> dict:
    import sys

    sys.path.insert(0, "/root/pkg")
    runs.reload()
    import waiparse.pipelines  # noqa: F401
    from parse_bench.inference.pipelines import get_pipeline

    cfg = get_pipeline(pipeline).config
    bdir = Path(f"/runs/bench/{base}/{split}/{base}/{group}")
    out_root = Path(f"/runs/bench/{pipeline}/{split}")
    pdir = out_root / pipeline / group
    if pdir.exists() and pipeline != base:
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True, exist_ok=True)
    if pipeline != base:
        for p in bdir.glob("*.raw.json"):
            (pdir / p.name).write_text(
                json.dumps(_rename(json.loads(p.read_text()), pipeline, dict(cfg)))
            )
    for p in [*pdir.glob("_evaluation_*"), *pdir.parent.glob("_evaluation_*")]:
        p.unlink()
    env = {**os.environ, "PYTHONPATH": "/root/pkg", "DOCPARSE_SERVER": "http://unused"}
    subprocess.check_call(
        [
            "python",
            "-m",
            "waiparse.cli",
            "inference",
            "renormalize",
            str(pdir),
            f"--pipeline_name={pipeline}",
            "--force",
        ],
        cwd="/root/pkg",
        env=env,
        stdout=subprocess.DEVNULL,
    )
    ev = subprocess.run(
        [
            "python",
            "-m",
            "waiparse.cli",
            "run",
            pipeline,
            "--input_dir",
            f"/data/split_{split}",
            "--output_dir",
            str(out_root),
            "--group",
            group,
            "--skip_inference",
            "--open_report=False",
        ],
        cwd="/root/pkg",
        env=env,
        capture_output=True,
        text=True,
    )
    runs.commit()
    rp = pdir.parent / "_evaluation_report.json"  # --group writes the report one level up
    if not rp.exists():
        raise RuntimeError(ev.stdout[-3000:] + ev.stderr[-3000:])
    rep = json.loads(rp.read_text())
    return {k: v for k, v in rep["aggregate_metrics"].items() if k.startswith("avg_")}


@app.local_entrypoint()
def score(
    pipeline: str, base: str = "wai_agent_det_off", split: str = "dev", group: str = "layout"
):
    for k, v in score_remote.remote(pipeline, base, split, group).items():
        print(f"{k:60} {v:.4f}")
