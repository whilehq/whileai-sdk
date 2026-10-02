"""The four figures on this recipe's page, drawn from results.json.

    python figures.py        # writes docs/figures/ptgs-*.svg

Each figure is a white card with its own background, so it reads on the
docs site in light and dark mode and on GitHub. 720 wide, 11 to 14 px type,
the guide figures' palette (scripts/gen_guide_figures.py).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "docs" / "figures"
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

W = 720
SANS = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Inter, sans-serif"
C = {
    "bg": "#FFFFFF",
    "ink": "#0B1220",
    "body": "#4B5563",
    "muted": "#6B7280",
    "line": "#D1D5DB",
    "surface": "#F4F6F9",
    "green": "#3F8F6B",
    "tint": "#F0FAF5",
    "warm": "#B5602A",
    "warm_tint": "#FBEDE3",
    "cool": "#3B6EA8",
    "cool_tint": "#E8F0FA",
    "grey": "#9CA3AF",
}
ARMS = [
    ("baseline", "Plain RL", "GRPO", C["grey"]),
    ("recipe", "Temperature fix", "PTGS", C["warm"]),
    ("ada", "Practice more", "Reinforce-Ada", C["green"]),
]


class Svg:
    def __init__(self, h: int) -> None:
        self.h = h
        self.parts = [
            f'<rect x="0.5" y="0.5" width="{W - 1}" height="{h - 1}" rx="12" '
            f'fill="{C["bg"]}" stroke="{C["line"]}"/>'
        ]

    def text(self, x, y, s, *, size=12, color="ink", weight=400, anchor="start"):
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="{size}" '
            f'font-weight="{weight}" fill="{C.get(color, color)}" text-anchor="{anchor}">{s}</text>'
        )

    def rect(self, x, y, w, h, fill, *, rx=4, stroke="none"):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{max(w, 0)}" height="{h}" rx="{rx}" '
            f'fill="{C.get(fill, fill)}" stroke="{C.get(stroke, stroke)}"/>'
        )

    def line(self, x1, y1, x2, y2, color="line", *, width=1, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{C.get(color, color)}" stroke-width="{width}"{d}/>'
        )

    def dot(self, x, y, r, fill, *, stroke="none"):
        self.parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{C.get(fill, fill)}" '
            f'stroke="{C.get(stroke, stroke)}" stroke-width="1.5"/>'
        )

    def save(self, name: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        body = "\n".join(self.parts)
        (OUT / name).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {self.h}" '
            f'width="{W}" height="{self.h}">\n{body}\n</svg>\n',
            encoding="utf-8",
        )


def title(s: Svg, head: str, sub: str) -> None:
    s.text(28, 38, head, size=17, weight=700)
    s.text(28, 60, sub, size=12.5, color="body")


def idea() -> None:
    """How each arm spends its practice on one easy and one hard problem."""
    s = Svg(330)
    title(
        s,
        "Three ways to practice",
        "Each problem gets attempts. Training only learns from a mix of right and wrong.",
    )
    rows = [
        ("Plain RL", "4 tries, same randomness", ["n"] * 4, ["n"] * 4, "0.9", "0.9"),
        (
            "Temperature fix",
            "4 tries, hotter when it keeps failing",
            ["c"] * 4,
            ["h"] * 4,
            "0.6",
            "1.35",
        ),
        (
            "Practice more",
            "up to 32 tries, trains on the 4 circled",
            ["n"] * 8,
            ["n"] * 24,
            "0.9",
            "0.9",
        ),
    ]
    s.text(330, 92, "EASY PROBLEM", size=11, color="muted", weight=600)
    s.text(500, 92, "HARD PROBLEM", size=11, color="muted", weight=600)
    fills = {"n": C["grey"], "c": C["cool"], "h": C["warm"]}
    for i, (name, how, easy, hard, te, th) in enumerate(rows):
        y = 125 + i * 72
        s.rect(20, y - 24, W - 40, 62, C["surface"], rx=8)
        s.text(36, y, name, size=14, weight=700)
        s.text(36, y + 20, how, size=12, color="body")
        for j, k in enumerate(easy):
            s.dot(338 + (j % 8) * 15, y - 4 + (j // 8) * 15, 5.5, fills[k])
        for j, k in enumerate(hard):
            s.dot(508 + (j % 8) * 15, y - 4 + (j // 8) * 15, 5.5, fills[k])
        s.text(338, y + 30, f"temperature {te}", size=11, color="muted")
        if len(hard) <= 8:
            s.text(508, y + 30, f"temperature {th}", size=11, color="muted")
        else:
            # The 4 it trains on: 2 right and 2 wrong once it has them.
            for j in range(4):
                s.dot(508 + j * 15, y - 4 + 2 * 15, 8, "none", stroke="ink")
                s.dot(338 + j * 15, y - 4, 8, "none", stroke="ink")
    s.save("ptgs-idea.svg")


def accuracy() -> None:
    """First-try and eight-try accuracy, both seeds, the untrained model as a line."""
    s = Svg(300)
    title(
        s,
        "How many math problems each model solves",
        "320 held-out MATH problems. Each dot is one training run; the bar is their average.",
    )
    base1 = R["arms"]["base"]["score"]
    base8 = R["arms"]["base"]["pass_at_k"]
    panels = [
        ("On the first try", "per_seed", base1, 28, 0.15, 0.45),
        ("Within 8 tries", "per_seed_pass_at_k", base8, 382, 0.40, 0.70),
    ]
    for head, key, base, x0, lo, hi in panels:
        px0, px1 = x0 + 118, x0 + 318
        sx = lambda v, px0=px0, px1=px1, lo=lo, hi=hi: px0 + (v - lo) / (hi - lo) * (px1 - px0)  # noqa: E731
        s.text(x0, 98, head.upper(), size=11, color="muted", weight=600)
        bx = sx(base)
        s.line(bx, 108, bx, 250, "muted", dash="3 3")
        s.text(bx, 266, f"untrained {base:.0%}", size=11, color="muted", anchor="middle")
        for i, (arm, name, _, col) in enumerate(ARMS):
            y = 132 + i * 44
            vals = R["arms"][arm][key]
            mean = statistics.fmean(vals)
            s.text(x0, y + 4, name, size=12.5, weight=600)
            s.rect(px0, y - 9, sx(mean) - px0, 18, col + "33", rx=4)
            for v in vals:
                s.dot(sx(v), y, 6, col, stroke="#FFFFFF")
            s.text(sx(max(vals)) + 12, y + 4, f"{mean:.0%}", size=12.5, weight=700, color=col)
    s.save("ptgs-accuracy.svg")


def waste_and_cost() -> None:
    """Share of training problems that taught nothing, and minutes per run."""
    s = Svg(270)
    title(
        s,
        "What each method spent",
        "Left: training problems where all 4 attempts agreed, so nothing was learned. Right: minutes per run.",
    )
    sampler = R["sampler"]
    flat_key = {
        "baseline": "frac_reward_zero_std",
        "recipe": "ptgs/zero_signal_frac",
        "ada": "ada/no_gradient",
    }
    wasted = {
        arm: statistics.fmean(statistics.fmean(t[flat_key[arm]]) for t in sampler[arm])
        for arm, *_ in ARMS
    }
    minutes = {arm: R["arms"][arm]["gpu_minutes"] for arm, *_ in ARMS}
    # GRPO's seed-17 container also ran the three untrained evals; it trains
    # the same 48 rollouts a step as PTGS, so it is drawn at PTGS's minutes.
    minutes["baseline"] = minutes["recipe"]
    for x0, head, vals, vmax, fmt in (
        (28, "PRACTICE THAT TAUGHT NOTHING", wasted, 1.0, "{:.0%}"),
        (382, "MINUTES TO TRAIN ONE RUN", minutes, 220.0, "{:.0f} min"),
    ):
        s.text(x0, 98, head, size=11, color="muted", weight=600)
        for i, (arm, name, _, col) in enumerate(ARMS):
            y = 128 + i * 42
            v = vals[arm]
            s.text(x0, y + 4, name, size=12.5, weight=600)
            w = v / vmax * 170
            s.rect(x0 + 118, y - 10, w, 20, col, rx=4)
            s.text(x0 + 118 + w + 8, y + 4, fmt.format(v), size=12.5, weight=700, color=col)
    s.text(
        382,
        252,
        "Plain RL takes the same time as the temperature fix: same 4 tries.",
        size=11,
        color="muted",
    )
    s.save("ptgs-cost.svg")


def tax() -> None:
    """Sharpening Tax at 8 tries per arm, with its 95% interval, around zero."""
    s = Svg(250)
    title(
        s,
        "Did training cost the model its range?",
        "Sharpening Tax at 8 tries, with the range it very likely sits in. Right of zero = range lost.",
    )
    lo, hi = -0.06, 0.06
    px0, px1 = 200, 660
    sx = lambda v: px0 + (v - lo) / (hi - lo) * (px1 - px0)  # noqa: E731
    s.rect(sx(0), 86, px1 - sx(0), 130, C["warm_tint"], rx=0)
    s.text(px1 - 8, 102, "range lost", size=11, color="warm", anchor="end", weight=600)
    s.text(px0 + 8, 102, "range kept", size=11, color="green", weight=600)
    s.line(sx(0), 86, sx(0), 216, "ink", width=1.5)
    for i, (arm, name, _, col) in enumerate(ARMS):
        y = 130 + i * 32
        t = R["arms"][arm]["sharpening_tax"]
        s.text(28, y + 4, name, size=12.5, weight=600)
        s.line(sx(t["ci"][0]), y, sx(t["ci"][1]), y, col, width=3)
        s.dot(sx(t["tax_s"]), y, 6.5, col, stroke="#FFFFFF")
    s.text(
        sx(0),
        236,
        "Every range crosses zero: no method paid the tax.",
        size=12,
        anchor="middle",
        weight=600,
    )
    s.save("ptgs-tax.svg")


if __name__ == "__main__":
    idea()
    accuracy()
    waste_and_cost()
    tax()
    print(f"wrote {OUT}/ptgs-{{idea,accuracy,cost,tax}}.svg")
