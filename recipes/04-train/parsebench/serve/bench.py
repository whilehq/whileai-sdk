# ParseBench runs on Modal CPU containers, next to the data (a laptop's link to the HF hub is
# slow and resets on a 2 GB snapshot).
#
#   modal run serve/bench.py::download
#   modal run serve/bench.py::run --pipeline <name> [--group table] [--split dev]
#
# Needs DOCPARSE_WORKSPACE (waiparse/endpoints.py) and the Modal secrets docparse-vllm-key
# and huggingface-secret in your workspace.
#
# Data lives in volume docparse-data at /data (the HF dataset snapshot); results in
# docparse-runs at /runs/bench/<pipeline>/<split>. Our harness package (../waiparse) is
# mounted and imported so its providers register before the CLI resolves pipelines.

import os
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
        "huggingface_hub[hf_transfer]",
        "pypdfium2",
        "aiohttp",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/root/pkg", **ENDPOINT_ENV})
    .add_local_dir(ROOT / "waiparse", "/root/pkg/waiparse")
)

data = modal.Volume.from_name("docparse-data", create_if_missing=True)
runs = modal.Volume.from_name("docparse-runs", create_if_missing=True)
app = modal.App("docparse-bench")


@app.function(
    image=image,
    volumes={"/data": data},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download():
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="llamaindex/ParseBench", repo_type="dataset", local_dir="/data/full")
    snapshot_download(
        repo_id="llamaindex/ParseBench",
        repo_type="dataset",
        revision="test-data",
        local_dir="/data/test",
    )
    data.commit()
    for d in ("/data/full", "/data/test"):
        n = sum(1 for _ in Path(d).rglob("*.pdf"))
        print(d, n, "pdfs")


@app.function(
    image=image,
    volumes={"/data": data, "/runs": runs},
    timeout=24 * 3600,
    cpu=8,
    memory=16384,
    secrets=[modal.Secret.from_name("docparse-vllm-key")],
)
def run_bench(pipeline: str, group: str = "", split: str = "dev", extra: str = ""):
    import sys

    sys.path.insert(0, "/root/pkg")
    from waiparse.endpoints import vlm_server

    os.environ["DOCPARSE_URL"] = vlm_server().rstrip("/") + "/v1"
    os.environ["DOCPARSE_KEY"] = os.environ["VLLM_API_KEY"]
    os.environ["DOCPARSE_SERVER"] = os.environ["DOCPARSE_URL"].removesuffix("/v1")
    input_dir = "/data/test" if split == "smoke" else f"/data/split_{split}"
    if split not in ("smoke", "full") and not Path(input_dir).exists():
        from waiparse.split import materialize

        materialize("/data/full", input_dir, split)
        data.commit()
    if split == "full":
        input_dir = "/data/full"
    out = f"/runs/bench/{pipeline}/{split}"
    cmd = [
        "python",
        "-m",
        "waiparse.cli",
        "run",
        pipeline,
        "--input_dir",
        input_dir,
        "--output_dir",
        out,
        "--open_report=False",
        "--max_concurrent=48",
    ]
    if group:
        cmd += ["--group", group]
    cmd += extra.split()
    print(*cmd, flush=True)
    rc = subprocess.call(cmd, cwd="/root/pkg")
    runs.commit()
    return rc


@app.local_entrypoint()
def run(pipeline: str, group: str = "", split: str = "dev", extra: str = ""):
    print(run_bench.remote(pipeline, group, split, extra))


HEADLINE = {
    "table": "avg_grits_trm_composite",
    "chart": "avg_rule_pass_rate",
    "text_content": "avg_content_faithfulness",
    "text_formatting": "avg_semantic_formatting",
    "layout": "avg_layout_element_rule_pass_rate",
}


@app.function(image=image, volumes={"/runs": runs}, timeout=600)
def headline(specs: list[str]) -> list[dict]:
    import json

    runs.reload()
    rows = []
    for spec in specs:
        pipeline, split = spec.split(":")
        row = {"run": spec}
        for cat, key in HEADLINE.items():
            p = Path(f"/runs/bench/{pipeline}/{split}/{pipeline}/{cat}/_evaluation_report.json")
            if p.exists():
                rep = json.loads(p.read_text())
                row[cat] = 100 * rep["aggregate_metrics"].get(key, float("nan"))
                row[cat + "_failed"] = rep.get("failed", 0)
        vals = [row.get(c) for c in HEADLINE]
        row["overall"] = sum(vals) / 5 if all(v is not None for v in vals) else None
        rows.append(row)
    return rows


@app.local_entrypoint()
def scores(runs_: str):
    cols = ["overall", *HEADLINE]
    print(f"{'run':34} " + " ".join(f"{c[:7]:>7}" for c in cols))
    for r in headline.remote(runs_.split(",")):
        cells = " ".join(f"{r[c]:7.2f}" if r.get(c) is not None else f"{'-':>7}" for c in cols)
        fails = sum(r.get(c + "_failed", 0) for c in HEADLINE)
        print(f"{r['run']:34} {cells}  failed={fails}")


@app.function(image=image, volumes={"/runs": runs}, timeout=600)
def details_remote(spec: str, cat: str) -> dict:
    import json

    runs.reload()
    pipeline, split = spec.split(":")
    rep = json.loads(
        Path(f"/runs/bench/{pipeline}/{split}/{pipeline}/{cat}/_evaluation_report.json").read_text()
    )
    return {k: v for k, v in rep["aggregate_metrics"].items() if k.startswith("avg_")}


@app.local_entrypoint()
def details(spec: str, cat: str):
    for k, v in details_remote.remote(spec, cat).items():
        print(f"{k:60} {v:.3f}")


@app.function(image=image, volumes={"/runs": runs, "/data": data}, timeout=1200)
def fmt_failures_remote(
    spec: str, types: str = "is_bold,is_title,is_sup,is_strikeout", limit: int = 40
) -> str:
    """Failing formatting rules with the output markdown around the target text."""
    import json

    from parse_bench.evaluation.metrics.parse.rules_formatting import FormattingRule, TitleLevelRule

    runs.reload()
    pipeline, split = spec.split(":")
    base = Path(
        f"/runs/bench/{pipeline}/{split}/{pipeline}/text"
    )  # inference results live under the doc folder name
    want = set(types.split(","))
    lines, n_fail, n_all = [], 0, 0
    for row in (
        json.loads(line)
        for line in open(f"/data/split_{split}/text_formatting.jsonl", encoding="utf-8")
    ):
        if row["type"] not in want:
            continue
        res = base / (Path(row["pdf"]).stem + ".result.json")
        if not res.exists():
            continue
        md = json.loads(res.read_text())["output"]["markdown"]
        rule = {"type": row["type"], "id": row["id"], **json.loads(row["rule"])}
        cls = TitleLevelRule if row["type"] == "is_title" else FormattingRule
        try:
            ok = cls(rule).run(md)[0]
        except Exception as e:
            ok = f"ERR {e}"
        n_all += 1
        if ok is True:
            continue
        n_fail += 1
        if n_fail <= limit:
            t = rule["text"]
            i = md.lower().find(t.lower()[:25])
            ctx = (
                md[max(0, i - 60) : i + len(t) + 60].replace("\n", "\n")
                if i >= 0
                else "<NOT FOUND>"
            )
            lines.append(f"[{row['type']}] {Path(row['pdf']).stem} :: {t!r}\n    {ctx}")
    return f"failed {n_fail}/{n_all}\n" + "\n".join(lines)


@app.local_entrypoint()
def fmt_failures(spec: str, types: str = "is_bold,is_title,is_sup,is_strikeout", limit: int = 40):
    print(fmt_failures_remote.remote(spec, types, limit))


DOC_COL = {
    "table": "grits_trm_composite",
    "chart": "rule_pass_rate",
    "text_content": "content_faithfulness",
    "text_formatting": "semantic_formatting",
    "layout": "layout_element_rule_pass_rate",
}


@app.function(image=image, volumes={"/runs": runs}, timeout=1200)
def bands_remote(specs: list[str], pairs: list[str], n_boot: int = 2000, seed: int = 0) -> dict:
    """95% percentile-bootstrap bands over documents: per dimension, and for Overall (mean of the
    five dimension means, each dimension resampled independently). `pairs` = "a:split|b:split" for
    paired differences on the same documents."""
    import csv
    import random

    runs.reload()

    def load(spec):
        pipeline, split = spec.split(":")
        out = {}
        for cat, col in DOC_COL.items():
            p = Path(f"/runs/bench/{pipeline}/{split}/{pipeline}/{cat}/_evaluation_results.csv")
            vals = {}
            for r in csv.DictReader(open(p, encoding="utf-8")):
                v = r.get(col) or r.get("avg_" + col) or ""
                if v not in ("", "None"):  # the report's avg_* skips documents with no value
                    vals[r["test_id"]] = 100 * float(v)
            out[cat] = vals
        return out

    rng = random.Random(seed)

    def boot(per_cat):  # per_cat: cat -> list of per-doc values
        means = {c: sum(v) / len(v) for c, v in per_cat.items()}
        overall = sum(means.values()) / len(means)
        draws = []
        for _ in range(n_boot):
            m = []
            for v in per_cat.values():
                s = [v[rng.randrange(len(v))] for _ in range(len(v))]
                m.append(sum(s) / len(s))
            draws.append(sum(m) / len(m))
        draws.sort()
        return {
            "overall": overall,
            "lo": draws[int(0.025 * n_boot)],
            "hi": draws[int(0.975 * n_boot) - 1],
            "dims": means,
            "n_docs": {c: len(v) for c, v in per_cat.items()},
        }

    res = {}
    for spec in specs:
        d = load(spec)
        res[spec] = boot({c: list(v.values()) for c, v in d.items()})
    for pair in pairs:
        a, b = pair.split("|")
        da, db = load(a), load(b)
        diffs = {c: [db[c][k] - da[c][k] for k in da[c] if k in db[c]] for c in DOC_COL}
        res[pair] = boot(diffs)
    return res


@app.local_entrypoint()
def bands(specs: str, pairs: str = ""):
    import json

    out = bands_remote.remote(
        [s for s in specs.split(",") if s], [p for p in pairs.split(",") if p]
    )
    print(json.dumps(out, indent=1))
