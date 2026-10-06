"""The three figures on this recipe's page, drawn from results.json.

    python figures.py        # writes docs/figures/context-lm-*.svg

Each figure is a white card with its own background, so it reads on the
docs site in light and dark mode, on GitHub, and as an image in a post. 560
wide with 13 to 21 px type, the same canvas as recipes/papers/ptgs.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "docs" / "figures"
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

W = 560
SANS = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Inter, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
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
}
# Arm, label, color. Plain GRPO is the neutral reference; every row is
# labeled, so color is never the only cue.
ARMS = [
    ("baseline", "Plain GRPO", C["muted"]),
    ("paper", "+ paper's Eq. 6", C["green"]),
    ("recipe", "+ Eq. 6, complete files", C["warm"]),
]
LABEL_W = 176
SEEDS = "17, 18, 19, 20"


class Svg:
    def __init__(self, h: int) -> None:
        self.h = h
        self.parts = [
            f'<rect x="0.5" y="0.5" width="{W - 1}" height="{h - 1}" rx="14" '
            f'fill="{C["bg"]}" stroke="{C["line"]}"/>'
        ]

    def text(self, x, y, s, *, size=15, color="ink", weight=400, anchor="start", mono=False):
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        fam = MONO if mono else SANS
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
            f'font-weight="{weight}" fill="{C.get(color, color)}" text-anchor="{anchor}">{s}</text>'
        )

    def rect(self, x, y, w, h, fill, *, rx=5, stroke="none"):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{max(w, 0)}" height="{h}" rx="{rx}" '
            f'fill="{C.get(fill, fill)}" stroke="{C.get(stroke, stroke)}"/>'
        )

    def line(self, x1, y1, x2, y2, color="line", *, width=1.5, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{C.get(color, color)}" stroke-width="{width}" stroke-linecap="round"{d}/>'
        )

    def dot(self, x, y, r, fill, *, stroke="none", sw=2):
        self.parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{C.get(fill, fill)}" '
            f'stroke="{C.get(stroke, stroke)}" stroke-width="{sw}"/>'
        )

    def save(self, name: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        body = "\n".join(self.parts)
        (OUT / name).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {self.h}" '
            f'width="{W}" height="{self.h}">\n{body}\n</svg>\n',
            encoding="utf-8",
            newline="\n",
        )


def title(s: Svg, head: str, sub: str) -> int:
    s.text(24, 42, head, size=21, weight=700)
    s.text(24, 68, sub, size=14, color="body")
    return 100


def idea() -> None:
    """What the model sees at each step: a normal chat context against a
    context file, over the five chunks of one log."""
    steps = 5
    files = [
        ["maple: 412", "river: 935"],
        ["maple: 412", "river: 162"],
        ["maple: 288", "river: 162"],
        ["maple: 177", "river: 162"],
        ["maple: 177", "river: 162"],
    ]
    s = Svg(100 + 150 + 104 + 60)
    y0 = title(
        s,
        "What the model sees before each answer",
        "A log in 5 chunks of 8 lines, then: final value of maple? (Illustration.)",
    )
    x0, gap = 24, 8
    cw = (W - 48 - gap * (steps - 1)) / steps
    s.text(24, y0 + 14, "NORMAL LLM: THE CONTEXT ONLY GROWS", size=12.5, color="muted", weight=700)
    base = y0 + 132
    for t in range(steps):
        x = x0 + t * (cw + gap)
        for k in range(t + 1):
            s.rect(x, base - (k + 1) * 20, cw, 17, "surface", rx=4, stroke="line")
            s.text(x + cw / 2, base - k * 20 - 6, f"chunk {k + 1}", size=11, color="muted",
                   anchor="middle")  # fmt: skip
    top = y0 + 150
    s.text(24, top + 14, "CONTEXT LM: IT REWRITES ONE FILE", size=12.5, color="muted", weight=700)
    for t in range(steps):
        x = x0 + t * (cw + gap)
        s.rect(x, top + 28, cw, 56, "tint", rx=6, stroke="green")
        for j, ln in enumerate(files[t]):
            s.text(x + 8, top + 46 + j * 15, ln, size=11, mono=True)
        s.text(x + 8, top + 77, "...", size=11, color="muted", mono=True)
        s.text(x + cw / 2, top + 102, f"after chunk {t + 1}", size=12, color="muted",
               anchor="middle")  # fmt: skip
    s.text(24, s.h - 18, "Each chunk is gone after its turn. The model keeps only what it wrote.",
           size=13, color="body")  # fmt: skip
    s.save("context-lm-idea.svg")


def dot_panel(s: Svg, y0: int, key: str, seeds_key: str, lo: float, hi: float, ticks, fmt) -> None:
    """One row per arm: four seed dots and the mean as a tick, on a shared
    axis with labeled ticks. No bars: the axis does not start at zero."""
    px0, px1 = 24 + LABEL_W, W - 96

    def sx(v):
        return px0 + (v - lo) / (hi - lo) * (px1 - px0)

    for t in ticks:
        s.line(sx(t), y0 + 6, sx(t), y0 + 120, "line", width=1)
        s.text(sx(t), y0 + 140, fmt(t), size=12.5, color="muted", anchor="middle")
    for i, (arm, name, col) in enumerate(ARMS):
        y = y0 + 24 + i * 38
        a = R["arms"][arm]
        s.text(24, y + 5, name, size=15, weight=600)
        for j, v in enumerate(a[seeds_key]):
            s.dot(sx(v), y + (-5 if j % 2 else 5), 6.5, col, stroke="#FFFFFF")
        m = sx(a[key])
        s.line(m, y - 14, m, y + 14, col, width=3.5)
        s.text(px1 + 14, y + 5, fmt(a[key]), size=15, weight=700, color=col)


def accuracy() -> None:
    """Held-out pass@1 per arm: four seeds and the pooled mean."""
    base = R["arms"]["base"]["score"]
    s = Svg(100 + 170)
    y0 = title(
        s,
        "Answered right on 200 held-out logs",
        f"Qwen2.5-1.5B, 60 steps. Untrained: {base:.2f}. Dots: 4 seeds. Tick: mean.",
    )
    dot_panel(
        s, y0, "score", "per_seed", 0.80, 1.00, [0.8, 0.85, 0.9, 0.95, 1.0], lambda v: f"{v:.2f}"
    )
    s.save("context-lm-accuracy.svg")


def tokens() -> None:
    """Tokens a trajectory per arm: four seeds and the pooled mean."""
    base = R["arms"]["base"]["cost_tokens"]
    s = Svg(100 + 170 + 26)
    y0 = title(
        s,
        "Tokens spent per held-out log",
        f"Prefix-reuse tokens over 6 steps. Untrained: {base:,}. Dots: 4 seeds.",
    )
    dot_panel(s, y0, "cost_tokens", "cost_per_seed", 800, 2200, [800, 1200, 1600, 2000],
              lambda v: f"{v:,.0f}")  # fmt: skip
    d = R["deltas"]["paper"]["cost"]["pooled"]["relative"]
    s.text(24, s.h - 18, f"Eq. 6 spends {abs(d):.0%} less than plain GRPO; most of that is one "
           "seed whose file bloated.", size=12.5, color="body")  # fmt: skip
    s.save("context-lm-tokens.svg")


if __name__ == "__main__":
    idea()
    accuracy()
    tokens()
    print(f"wrote {OUT}/context-lm-{{idea,accuracy,tokens}}.svg")
