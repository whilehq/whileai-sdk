"""Draw the recipe's chart from results.json: accuracy at each budget for every router.

Run: python charts.py   (needs `matplotlib`; writes a PNG into docs/figures/)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parents[2] / "docs" / "figures"

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SERIES = (  # name in results.json, label, color, line style
    ("true-kind", "True dataset label, same table (control)", "#888780", ":"),
    ("jev-task", "Jev names the kind (untrained)", "#8a5cd6", "-"),
    ("avengers-pro", "Avengers-Pro (part 1)", "#2a78d6", "-"),
    ("pointer-kind", "Our pointer model + the kind as a prior", "#c0392b", "-"),
    ("pointer", "Our pointer model, Qwen3-0.6B", "#eb6834", "--"),
    ("encoder", "Our encoder, ModernBERT-base", "#1baf7a", "--"),
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default=str(HERE / "results.json"))
    p.add_argument("--out", default=str(FIGURES))
    args = p.parse_args()
    res = json.loads(Path(args.results).read_text(encoding="utf-8"))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=160)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    budgets = [int(b) for b in res["budgets"]]
    for name, label, color, dash in SERIES:
        ys = [res["budgets"][str(b)].get(name, {}).get("accuracy") for b in budgets]
        if any(y is None for y in ys):
            continue
        ax.plot(budgets, ys, dash, marker="o", ms=5, color=color, lw=2.2, label=label)
    ax.set_xscale("log")
    ax.set_xticks(budgets, [f"${b}" for b in budgets])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(
        "Budget per 1,000 questions, knob picked on val (log scale)", color=MUTED, fontsize=11
    )
    ax.set_ylabel("Accuracy on 1,061 held-out questions", color=MUTED, fontsize=11)
    ax.set_title(
        "Our models, Jev and part 1's router, at equal budgets",
        color=INK,
        fontsize=14,
        loc="left",
        pad=12,
    )
    ax.legend(frameon=False, fontsize=10, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "decision-router-budgets.png", facecolor="white")
    print(f"wrote {out / 'decision-router-budgets.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
