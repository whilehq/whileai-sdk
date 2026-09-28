# Chart RL data: ChartNet core_permissive (CDLA-Permissive-2.0, synthetic, image + source csv)
# -> {image, rules} tasks whose reward is ParseBench's ChartDataPointMatch on our rules.
#
#   modal run train/build_chart_data.py --shards 3
#
# Decontamination: a ChartNet chart is dropped when its (label, value) pairs cover >= 30%
# of any ParseBench chart page's rules (value match within 1%, label match after
# alphanumeric folding). Output: volume docparse-data at /data/train/charts/{train,val}.jsonl
# + images/, and decontam.json with every drop and the page it collided with.

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "parse-bench @ git+https://github.com/run-llama/ParseBench.git",
        "huggingface_hub[hf_transfer]",
        "pyarrow",
        "pillow",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/root/pkg"})
    .add_local_dir(
        __import__("pathlib").Path(__file__).resolve().parent.parent / "waiparse",
        "/root/pkg/waiparse",
    )
)
data = modal.Volume.from_name("docparse-data")
app = modal.App("docparse-build-charts")


def _fold(s: str) -> str:
    import re

    return re.sub(r"[^0-9a-z]", "", s.lower())


@app.function(
    image=image,
    volumes={"/data": data},
    timeout=4 * 3600,
    cpu=8,
    memory=32768,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def build(shards: int = 3, val_every: int = 50):
    import io
    import json
    import random
    import re
    from collections import defaultdict
    from pathlib import Path

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image
    from waiparse.rewards import chart_rules

    # ParseBench chart pages -> set of (folded label, value) per page (full set: dev AND test).
    bench = defaultdict(list)
    for line in open("/data/full/chart.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rule = json.loads(r["rule"])
        try:
            v = float(re.sub(r"[,\s$%]", "", str(rule["value"])))
        except ValueError:
            continue
        bench[r["pdf"]].append(({_fold(x) for x in rule["labels"]}, v))

    by_value = defaultdict(set)
    for pdf, bp in bench.items():
        for _, v in bp:
            by_value[round(v, 2)].add(pdf)

    files = sorted(
        f
        for f in HfApi().list_repo_files("ibm-granite/ChartNet", repo_type="dataset")
        if "core_permissive" in f and f.endswith(".parquet")
    )[:shards]
    out = Path("/data/train/charts")
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    kept, dropped, drops = defaultdict(list), 0, []
    for f in files:
        local = hf_hub_download("ibm-granite/ChartNet", f, repo_type="dataset")
        table = pq.read_table(local, columns=["id", "image", "csv", "chart_type", "library"])
        for row in table.to_pylist():
            rules = chart_rules(row["csv"], rng=rng)
            if len(rules) < 3:
                continue
            pairs = []
            for r in rules:
                try:  # noqa: SIM105  # a non-numeric value is skipped
                    pairs.append(
                        ({_fold(x) for x in r["labels"]}, float(re.sub(r"[,\s$%]", "", r["value"])))
                    )
                except ValueError:
                    pass
            hit = None
            # Only pages sharing at least two exact-ish values are candidates.
            cand = defaultdict(int)
            for _, pv in pairs:
                for pdf in by_value.get(round(pv, 2), ()):
                    cand[pdf] += 1
            for pdf in (p for p, c in cand.items() if c >= 2):
                bp = bench[pdf]
                cover = sum(
                    any(bl & pl and abs(bv - pv) <= 0.01 * max(abs(bv), 1e-9) for pl, pv in pairs)
                    for bl, bv in bp
                )
                if bp and cover / len(bp) >= 0.30:
                    hit = pdf
                    break
            if hit:
                dropped += 1
                drops.append({"id": row["id"], "parsebench_pdf": hit})
                continue
            img = row["image"]
            raw = img["bytes"] if isinstance(img, dict) else img
            name = f"{row['id']}.png"
            Image.open(io.BytesIO(raw)).convert("RGB").save(out / "images" / name)
            split = (
                "val" if len(kept["train"]) % val_every == 0 and len(kept["val"]) < 500 else "train"
            )
            kept[split].append(
                {
                    "id": row["id"],
                    "image": f"images/{name}",
                    "rules": rules,
                    "chart_type": row["chart_type"],
                    "library": row["library"],
                }
            )
        print(f, {k: len(v) for k, v in kept.items()}, "dropped", dropped, flush=True)
    for split, rows in kept.items():
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    json.dump({"dropped": dropped, "drops": drops}, open(out / "decontam.json", "w"))
    data.commit()
    return {k: len(v) for k, v in kept.items()} | {"dropped": dropped}


@app.local_entrypoint()
def main(shards: int = 3):
    print(build.remote(shards))


# Chart types whose source csv is raw samples, not plotted values (pre-flight 2026-09-25: mean
# reward 0.0-0.27 on these, i.e. the table is not recoverable from the picture). ParseBench's
# charts are report charts (bar/line/pie/area/stacked/combo), so train on those.
UNRECOVERABLE = {
    "Box Plot",
    "Violin Plot",
    "Kernel Density Estimate Plot",
    "Histogram",
    "Swarm Plot",
    "Scatter Plot",
    "Bubble Chart",
}


@app.function(image=image, volumes={"/data": data}, timeout=1800)
def filter_types(src: str = "/data/train/charts", dst: str = "/data/train/charts_pb"):
    import json
    from collections import Counter
    from pathlib import Path

    Path(dst).mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ("train", "val"):
        rows = [json.loads(line) for line in open(f"{src}/{split}.jsonl", encoding="utf-8")]
        kept = [
            dict(r, image="../charts/" + r["image"])
            for r in rows
            if r["chart_type"] not in UNRECOVERABLE
        ]
        with open(f"{dst}/{split}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in kept)
        stats[split] = {
            "kept": len(kept),
            "of": len(rows),
            "types": Counter(r["chart_type"] for r in kept).most_common(),
        }
    data.commit()
    print(json.dumps(stats, indent=1))


@app.local_entrypoint()
def filter_main():
    filter_types.remote()


# Page-level chart tasks (2026-09-26): RL on isolated ChartNet crops with the crop prompt did not move
# ParseBench dev charts (step 100: 82.1 vs base 83.5-85.2), where the agent reads charts from a WHOLE
# page with prompts.CHART_PAGE. These tasks compose 1-2 ChartNet charts onto a report-like page
# (running header, heading, prose from the chart summaries, "Figure N." captions, page number) so the
# policy trains on the same input shape and prompt it is used with. Rules = the charts' own rules.
page_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("fonts-dejavu-core", "fonts-liberation")
    .uv_pip_install("pillow")
)


@app.function(image=page_image, volumes={"/data": data}, timeout=4 * 3600, cpu=8, memory=16384)
def build_pages(n_train: int = 6000, n_val: int = 300, seed: int = 0):
    import json
    import random
    import textwrap
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageFont

    src = Path("/data/train/charts")
    out = Path("/data/train/chart_pages")
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    fonts = [f for f in Path("/usr/share/fonts").rglob("*.ttf") if "Mono" not in f.name]
    pools = {}
    for split in ("train", "val"):
        rows = [json.loads(line) for line in open(src / f"{split}.jsonl", encoding="utf-8")]
        pools[split] = [r for r in rows if r["chart_type"] not in UNRECOVERABLE]
    words = [
        "the",
        "of",
        "and",
        "to",
        "in",
        "that",
        "for",
        "is",
        "on",
        "with",
        "as",
        "by",
        "this",
        "were",
        "from",
        "at",
        "are",
        "be",
        "which",
        "an",
        "has",
        "growth",
        "share",
        "total",
        "market",
        "report",
        "index",
        "rate",
        "change",
        "year",
        "region",
        "sector",
        "survey",
        "data",
        "increase",
        "decline",
        "trend",
        "compared",
        "average",
        "respondents",
        "percent",
        "level",
        "policy",
    ]

    def para(k):
        return " ".join(rng.choice(words) for _ in range(k)).capitalize() + "."

    def page(rows, name):
        W, H = 1700, 2200
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)

        def font(sz, bold=False):
            return ImageFont.truetype(
                str(rng.choice([f for f in fonts if ("Bold" in f.name) == bold] or fonts)), sz
            )

        m, y = 130, 90
        d.text((m, y), f"{para(4)[:-1]} | {rng.randint(2019, 2026)}", fill="#666", font=font(26))
        y += 80
        d.text((m, y), para(rng.randint(4, 8))[:-1], fill="black", font=font(44, True))
        y += 90
        body = font(30)
        for _ in range(rng.randint(1, 2)):
            for line in textwrap.wrap(para(rng.randint(30, 60)), 95):
                d.text((m, y), line, fill="black", font=body)
                y += 40
            y += 25
        rules = []
        for i, r in enumerate(rows):
            title = f"Figure {rng.randint(1, 9)}.{i + 1}. {para(rng.randint(4, 9))[:-1]}"
            d.text((m, y), title, fill="black", font=font(32, True))
            y += 55
            ch = Image.open(src / r["image"]).convert("RGB")
            wmax = (W - 2 * m) if len(rows) == 1 or rng.random() < 0.7 else (W - 2 * m) // 2
            s = min(wmax / ch.width, (H - y - 260) / max(1, len(rows) - i) / ch.height)
            ch = ch.resize((max(1, int(ch.width * s)), max(1, int(ch.height * s))))
            img.paste(ch, (m, y))
            y += ch.height + 20
            d.text((m, y), "Source: " + para(5), fill="#444", font=font(24))
            y += 60
            rules += [dict(x, id=f"c{i}_{x['id']}") for x in r["rules"]]
        if y < H - 300:
            for line in textwrap.wrap(para(rng.randint(20, 50)), 95):
                if y > H - 150:
                    break
                d.text((m, y), line, fill="black", font=body)
                y += 40
        d.text((W // 2, H - 90), str(rng.randint(3, 180)), fill="#666", font=font(26))
        img.save(out / "images" / name)
        return rules

    stats = {}
    for split, n in (("train", n_train), ("val", n_val)):
        pool, rows_out = pools[split], []
        for k in range(n):
            picks = rng.sample(pool, 2 if rng.random() < 0.35 else 1)
            name = f"{split}_{k:05d}.png"
            rules = page(picks, name)
            rows_out.append(
                {
                    "id": f"{split}_{k:05d}",
                    "image": f"images/{name}",
                    "rules": rules[:32],
                    "chart_type": "+".join(p["chart_type"] for p in picks),
                    "library": "page",
                }
            )
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows_out)
        stats[split] = len(rows_out)
    data.commit()
    print(stats)


@app.local_entrypoint()
def pages_main(n_train: int = 6000, n_val: int = 300):
    build_pages.remote(n_train, n_val)


@app.function(image=page_image, volumes={"/data": data}, timeout=3600, cpu=16)
def shrink_pages(max_side: int = 2048):
    """Resize page PNGs in place to the agent's page size (render.fit: 2048 px long side)."""
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    files = sorted(Path("/data/train/chart_pages/images").glob("*.png"))
    with ProcessPoolExecutor(16) as ex:
        n = sum(ex.map(_shrink_one, [(str(f), max_side) for f in files], chunksize=64))
    data.commit()
    print("resized", n, "of", len(files))


def _shrink_one(arg):
    from PIL import Image

    path, max_side = arg
    img = Image.open(path)
    s = max_side / max(img.size)
    if s >= 1:
        return 0
    img.convert("RGB").resize((round(img.width * s), round(img.height * s)), Image.LANCZOS).save(
        path
    )
    return 1


@app.local_entrypoint()
def shrink_main(max_side: int = 2048):
    shrink_pages.remote(max_side)
