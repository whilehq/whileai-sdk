"""CPU self-check run by train/modal_prime_rl.py `validate` after `rl --dry-run`.

    python -m docparse_charts.selfcheck [--model Qwen/Qwen3.8-27B] [--n 3]

1. CISPO loss: d loss / d logprob equals -min(rho, eps_max) * A per token; masked tokens get 0.
2. Reward: a gold table rebuilt from a task's own rules scores high, "NONE" and garbage score 0.
3. Data + render (if the split exists): loads val, renders the first tasks with the qwen3.8
   renderer (thinking on), and prints prompt tokens, image tokens and image size, which is
   what seq_len has to cover besides the completion.
Prints `SELFCHECK OK` on success; any failed assert raises.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_loss() -> None:
    import torch
    from prime_rl.trainer.rl.loss import LossInputs

    from docparse_charts.losses import cispo_loss

    t = torch.tensor([-0.5, -1.0, -2.0, -0.1], requires_grad=True)
    inf = torch.tensor([-0.6, -1.0, -3.0, -0.1])  # rho = e^0.1, 1, e^1 (> eps 2), 1
    adv = torch.tensor([1.0, -1.0, 0.5, 2.0])
    mask = torch.tensor([True, True, True, False])
    out = cispo_loss(LossInputs(t, inf, None, adv, mask), eps_max=2.0)
    out.loss.backward()
    rho = torch.exp(t.detach() - inf)
    want = -torch.clamp(rho, max=2.0) * adv * mask
    assert torch.allclose(t.grad, want, atol=1e-6), (t.grad, want)
    assert abs(out.metrics["cispo/truncated"].item() - 1 / 3) < 1e-6
    print("[selfcheck] cispo loss grad ok:", [round(x, 4) for x in t.grad.tolist()])


def gold_html(rules: list[dict]) -> str:
    """One table from rules: row label -> column label -> value."""
    cols: list[str] = []
    rows: dict[str, dict[str, str]] = {}
    for r in rules:
        labels = r["labels"]
        row, col = labels[0], (labels[1] if len(labels) > 1 else "value")
        if col not in cols:
            cols.append(col)
        rows.setdefault(row, {})[col] = str(r["value"])
    head = "".join(f"<th>{c}</th>" for c in ["category", *cols])
    body = "".join(
        "<tr><td>" + row + "</td>" + "".join(f"<td>{vals.get(c, '')}</td>" for c in cols) + "</tr>"
        for row, vals in rows.items()
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def check_reward(rows: list[dict]) -> None:
    from docparse_charts.taskset import score

    if not rows:
        rows = [
            {
                "rules": [
                    {
                        "type": "chart_data_point",
                        "labels": ["2021", "Revenue"],
                        "value": "12.5",
                        "max_diffs": 0,
                        "normalize_numbers": True,
                        "relative_tolerance": 0.01,
                        "id": "r0",
                    },
                    {
                        "type": "chart_data_point",
                        "labels": ["2022", "Revenue"],
                        "value": "14",
                        "max_diffs": 0,
                        "normalize_numbers": True,
                        "relative_tolerance": 0.05,
                        "id": "r1",
                    },
                    {
                        "type": "chart_data_point",
                        "labels": ["2021", "Cost"],
                        "value": "7",
                        "max_diffs": 0,
                        "normalize_numbers": True,
                        "relative_tolerance": 0.05,
                        "id": "r2",
                    },
                ]
            }
        ]
    golds = []
    for r in rows:
        g = score(gold_html(r["rules"]), r["rules"])
        golds.append(g)
        assert score("NONE", r["rules"]) == 0.0
        assert score("<think>" + gold_html(r["rules"]), r["rules"]) == 0.0  # truncated think block
    print(
        f"[selfcheck] reward on gold tables: {[round(g, 3) for g in golds]} (mean {sum(golds) / len(golds):.3f})"
    )
    assert sum(golds) / len(golds) > 0.5, "gold tables should pass most of their own rules"


def check_render(tasks, model: str) -> None:
    from renderers import create_renderer
    from renderers.base import load_tokenizer
    from renderers.configs import Qwen38RendererConfig

    tok = load_tokenizer(model)
    r = create_renderer(tok, Qwen38RendererConfig(enable_thinking=True, reasoning_effort="xhigh"))
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    for task in tasks:
        msgs = [m.model_dump(exclude_none=True) for m in task.data.prompt]
        ids = r.render_ids(msgs, add_generation_prompt=True)
        n_img = sum(1 for i in ids if i == pad)
        print(
            f"[selfcheck] render {task.data.name}: prompt_tokens={len(ids)} image_tokens={n_img} "
            f"tail={tok.decode(ids[-12:])!r}"
        )
        assert n_img > 0, "image tokens missing from the rendered prompt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()

    check_loss()

    from docparse_charts.taskset import DocparseChartsConfig, DocparseChartsTaskset

    cfg = DocparseChartsConfig(id="docparse-charts", split="val")
    jsonl = Path(cfg.data_dir) / "val.jsonl"
    rows = []
    if jsonl.exists():
        lines = jsonl.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(x) for x in lines[: a.n]]
        n_train = (
            sum(1 for _ in open(Path(cfg.data_dir) / "train.jsonl", encoding="utf-8"))
            if (Path(cfg.data_dir) / "train.jsonl").exists()
            else 0
        )
        print(f"[selfcheck] data: val={len(lines)} train={n_train} at {cfg.data_dir}")
    else:
        print(f"[selfcheck] DATA MISSING: {jsonl} (reward checked on a synthetic task only)")
    check_reward(rows)

    if rows:
        from PIL import Image

        tasks = DocparseChartsTaskset(cfg.model_copy(update={"max_tasks": a.n})).load()
        for r in rows:
            w, h = Image.open(Path(cfg.data_dir) / r["image"]).size
            print(
                f"[selfcheck] {r['id']}: {w}x{h}px, {len(r['rules'])} rules, {r.get('chart_type')}"
            )
        if not a.no_render:
            check_render(tasks, a.model)
    print("SELFCHECK OK" if rows else "SELFCHECK OK (no data yet)")


if __name__ == "__main__":
    main()
