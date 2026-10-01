"""Draw the recipe's two charts from results.json: cost against accuracy, and the OpenRouter comparison.

Run: python charts.py   (needs `matplotlib`; writes PNGs into docs/figures/)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parents[2] / "docs" / "figures"

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
GRAY, BLUE, ORANGE, GREEN = "#888780", "#2a78d6", "#eb6834", "#1baf7a"
SHORT = {
    "qwen3-235b-a22b-2507": "Qwen3-235B",
    "qwen3-235b-a22b-thinking-2507": "Qwen3-235B thinking",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-5": "GPT-5",
    "claude-sonnet-4": "Claude Sonnet 4",
    "deepseek-r1-0528": "DeepSeek R1",
}


# Label offsets in points, placed by eye so no label sits on a line or another label.
LABEL_AT = {
    "gemini-2.5-pro": (-6, -16),
    "gpt-5": (-6, -14),
    "qwen3-235b-a22b-2507": (6, -14),
    "qwen3-235b-a22b-thinking-2507": (-6, -14),
    "deepseek-r1-0528": (6, -14),
}


def frontier(curve):
    """The best accuracy reached at or under each cost along a router's knob sweep."""
    pts, best = [], -1.0
    for c in sorted(curve, key=lambda c: c["usd_per_1k"]):
        if c["accuracy"] > best:
            pts.append((c["usd_per_1k"], c["accuracy"]))
            best = c["accuracy"]
    return pts


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_frontier(res, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    style(ax)
    for m, v in res["single_models"].items():
        ax.scatter(
            v["usd_per_1k"], v["accuracy"], s=46, color=GRAY, edgecolor="white", lw=1.5, zorder=3
        )
        if m in SHORT:
            dx, dy = LABEL_AT.get(m, (7, 5))
            ax.annotate(
                SHORT[m],
                (v["usd_per_1k"], v["accuracy"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="right" if dx < 0 else "left",
                fontsize=10,
                color=MUTED,
            )
    for name, color, dash, label in (
        ("avengers-pro", BLUE, "-", "Avengers-Pro router"),
        ("knn", ORANGE, "--", "kNN router"),
    ):
        x, y = zip(*frontier(res["routers"][name]["curve"]))
        ax.plot(x, y, dash, color=color, lw=2.2, label=label, zorder=4)
    o = res["oracle"]
    ax.scatter(
        o["usd_per_1k"],
        o["accuracy"],
        marker="D",
        s=70,
        color=GREEN,
        edgecolor="white",
        lw=1.5,
        zorder=5,
        label="Perfect router (knows the answers)",
    )
    m = res["routers"]["avengers-pro"]["points"]["match_best_cheaper"]
    b = res["best_single"]
    ax.scatter(
        m["usd_per_1k"], m["accuracy"], s=60, color=BLUE, edgecolor="white", lw=1.5, zorder=6
    )
    ax.annotate(
        f"Router: {m['accuracy']:.1%} at ${m['usd_per_1k']:.0f}\n"
        f"Gemini 2.5 Pro: {b['accuracy']:.1%} at ${b['usd_per_1k']:.0f}",
        (m["usd_per_1k"], m["accuracy"]),
        xytext=(m["usd_per_1k"] * 0.55, 0.70),
        color=BLUE,
        fontsize=11,
        ha="center",
        arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 1.0},
    )
    ax.scatter([], [], s=46, color=GRAY, label="Single model")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ["$1", "$2", "$5", "$10", "$20", "$50", "$100"])
    ax.set_xlim(0.8, 150)
    ax.set_ylim(0.38, 0.86)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Cost per 1,000 questions (log scale)", color=MUTED, fontsize=11)
    ax.set_ylabel("Accuracy on held-out questions", color=MUTED, fontsize=11)
    ax.set_title(
        f"Twelve models, three routers, {res['n']['test']:,} held-out questions",
        color=INK,
        fontsize=14,
        loc="left",
        pad=12,
    )
    ax.legend(frameon=False, fontsize=10, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def chart_openrouter(res, path):
    import matplotlib.pyplot as plt

    o = res["routers"]["avengers-pro"]["points"]["at_openrouter_cost"]["vs_openrouter"]
    k = res["routers"]["knn"]["points"]["at_openrouter_cost"]["vs_openrouter"]
    bars = [
        ("OpenRouter auto router", o["openrouter_accuracy"], o["openrouter_usd_per_1k"], GRAY),
        ("Avengers-Pro router", o["router_accuracy"], o["router_usd_per_1k"], BLUE),
        ("kNN router", k["router_accuracy"], k["router_usd_per_1k"], ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
    style(ax)
    ax.grid(axis="y", visible=False)
    y = range(len(bars))[::-1]
    ax.barh(list(y), [b[1] for b in bars], color=[b[3] for b in bars], height=0.55)
    for yi, (_, acc, usd, _) in zip(y, bars):
        ax.text(
            acc + 0.004, yi, f"{acc:.1%}  at ${usd:.0f} per 1k", va="center", fontsize=11, color=INK
        )
    ax.set_yticks(list(y), [b[0] for b in bars], fontsize=11, color=INK)
    ax.set_xlim(0.4, 0.68)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Accuracy on held-out questions", color=MUTED, fontsize=11)
    ax.set_title(
        f"Against OpenRouter's auto router, at or under its cost ({o['n']} questions)",
        color=INK,
        fontsize=14,
        loc="left",
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default=str(HERE / "results.json"))
    p.add_argument("--out", default=str(FIGURES))
    args = p.parse_args()
    res = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chart_frontier(res, out / "model-router-frontier.png")
    chart_openrouter(res, out / "model-router-vs-openrouter.png")
    print(f"wrote {out / 'model-router-frontier.png'} and {out / 'model-router-vs-openrouter.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
