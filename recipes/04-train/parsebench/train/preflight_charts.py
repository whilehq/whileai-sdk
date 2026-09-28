# RL pre-flight for the chart task: base-model reward mean,
# per-task spread and the share of zero-variance groups, on N val charts x K samples,
# through the same server + prompt + reward the RL run uses.
#
#   modal run train/preflight_charts.py --n 100 --k 4

import os

import modal

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
    .apt_install("git")
    .uv_pip_install(
        "parse-bench @ git+https://github.com/run-llama/ParseBench.git", "aiohttp", "pillow"
    )
    .env({"PYTHONPATH": "/root/pkg", **ENDPOINT_ENV})
    .add_local_dir(
        __import__("pathlib").Path(__file__).resolve().parent.parent / "waiparse",
        "/root/pkg/waiparse",
    )
)
data = modal.Volume.from_name("docparse-data")
app = modal.App("docparse-preflight")


@app.function(
    image=image,
    volumes={"/data": data},
    timeout=3 * 3600,
    secrets=[modal.Secret.from_name("docparse-vllm-key")],
)
def preflight(
    n: int = 100,
    k: int = 4,
    split: str = "val",
    effort: str = "xhigh",
    model: str = "qwen3.8-27b",
    data_dir: str = "/data/train/charts",
    page: bool = False,
):
    import asyncio
    import json
    import os
    import statistics
    from collections import defaultdict

    import aiohttp
    from PIL import Image

    os.environ["DOCPARSE_KEY"] = os.environ["VLLM_API_KEY"]
    from waiparse import prompts
    from waiparse.agent import _extract_tables
    from waiparse.client import VLM
    from waiparse.rewards import chart_reward

    rows = [json.loads(line) for line in open(f"{data_dir}/{split}.jsonl")][:n]
    vlm = VLM(model=model)
    sem = asyncio.Semaphore(32)

    async def one(s, row):
        img = Image.open(f"{data_dir}/{row['image']}").convert("RGB")
        if page:
            from waiparse.render import fit

            img = fit(img)
        async with sem:
            try:
                outs = await vlm.ask(
                    s,
                    img,
                    prompts.CHART_PAGE if page else prompts.CHART,
                    thinking=True,
                    effort=effort,
                    max_tokens=8192,
                    n=k,
                )
            except Exception as e:
                return row, [0.0] * k, str(e)[:200]
        return row, [chart_reward(_extract_tables(o), row["rules"]) for o in outs], None

    async def main():
        async with aiohttp.ClientSession() as s:
            return await asyncio.gather(*(one(s, r) for r in rows))

    res = asyncio.run(main())
    means = [statistics.mean(r) for _, r, _ in res]
    zero_var = sum(1 for _, r, _ in res if max(r) == min(r))
    by_type = defaultdict(list)
    for row, r, _ in res:
        by_type[row["chart_type"]].append(statistics.mean(r))
    out = {
        "n": len(res),
        "k": k,
        "mean_reward": statistics.mean(means),
        "zero_variance_share": zero_var / len(res),
        "solved_share": sum(m >= 0.999 for m in means) / len(res),
        "zero_share": sum(m == 0 for m in means) / len(res),
        "band_0.2_0.8": sum(0.2 <= m <= 0.8 for m in means) / len(res),
        "errors": sum(1 for *_, e in res if e),
        "by_type": {
            t: round(statistics.mean(v), 3)
            for t, v in sorted(by_type.items(), key=lambda x: -len(x[1]))
        },
    }
    print(json.dumps(out, indent=1))
    return out


@app.local_entrypoint()
def main(
    n: int = 100,
    k: int = 4,
    effort: str = "xhigh",
    data_dir: str = "/data/train/charts",
    page: bool = False,
):
    preflight.remote(n, k, "val", effort, "qwen3.8-27b", data_dir, page)
