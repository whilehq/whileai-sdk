"""The four figures on this recipe's page, drawn from results.json.

    python figures.py        # writes docs/figures/ptgs-*.svg

Each figure is a white card with its own background, so it reads on the
docs site in light and dark mode and on GitHub. 560 wide with 13 to 21 px
type: the docs column is about 520 px, so a wider canvas shrinks its text
below reading size. Panels stack instead of sitting side by side.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "docs" / "figures"
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

W = 560
SANS = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Inter, sans-serif"
C = {
    "bg": "#FFFFFF",
    "ink": "#0B1220",
    "body": "#4B5563",
    "muted": "#6B7280",
    "line": "#D1D5DB",
    "surface": "#F4F6F9",
    "green": "#3F8F6B",
    "warm": "#B5602A",
    "warm_tint": "#FBEDE3",
    "cool": "#3B6EA8",
    "grey": "#9CA3AF",
}
ARMS = [
    ("baseline", "GRPO", C["grey"]),
    ("recipe", "PTGS", C["warm"]),
    ("ada", "Reinforce-Ada", C["green"]),
]
LABEL_W = 150  # left column for arm names


class Svg:
    def __init__(self, h: int) -> None:
        self.h = h
        self.parts = [
            f'<rect x="0.5" y="0.5" width="{W - 1}" height="{h - 1}" rx="14" '
            f'fill="{C["bg"]}" stroke="{C["line"]}"/>'
        ]

    def text(self, x, y, s, *, size=15, color="ink", weight=400, anchor="start"):
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{SANS}" font-size="{size}" '
            f'font-weight="{weight}" fill="{C.get(color, color)}" text-anchor="{anchor}">{s}</text>'
        )

    def rect(self, x, y, w, h, fill, *, rx=5):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{max(w, 0)}" height="{h}" rx="{rx}" '
            f'fill="{C.get(fill, fill)}"/>'
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
    """How each arm samples rollouts for a high- and a low-pass-rate prompt."""
    rows = [
        ("GRPO", "4 rollouts, fixed T", ["n"] * 4, ["n"] * 4, "T = 0.9", "T = 0.9"),
        ("PTGS", "4 rollouts, T from posterior", ["c"] * 4, ["h"] * 4, "T = 0.6", "T = 1.35"),
        ("Reinforce-Ada", "up to 32 rollouts, keep 4", ["n"] * 8, ["n"] * 24, "", ""),
    ]
    s = Svg(110 + 32 + 3 * 92)
    y0 = title(
        s,
        "Rollout sampling per prompt",
        "A GRPO group has nonzero advantage only if its rewards differ.",
    )
    ex, hx = 238, 392
    s.text(ex, y0 + 14, "HIGH PASS RATE", size=12.5, color="muted", weight=700)
    s.text(hx, y0 + 14, "LOW PASS RATE", size=12.5, color="muted", weight=700)
    fills = {"n": C["grey"], "c": C["cool"], "h": C["warm"]}
    for i, (name, how, easy, hard, te, th) in enumerate(rows):
        top = y0 + 30 + i * 92
        s.rect(14, top, W - 28, 80, C["surface"], rx=10)
        s.text(28, top + 32, name, size=16.5, weight=700)
        s.text(28, top + 56, how, size=13.5, color="body")
        for xs, dots, label in ((ex, easy, te), (hx, hard, th)):
            for j, k in enumerate(dots):
                s.dot(xs + 6 + (j % 8) * 18, top + 20 + (j // 8) * 18, 6.5, fills[k])
            if label:
                s.text(xs, top + 66, label, size=13, color="muted")
        if len(hard) > 8:
            for j in range(4):  # the 4 kept for the update
                s.dot(ex + 6 + j * 18, top + 20, 9.5, "none", stroke="ink")
                s.dot(hx + 6 + j * 18, top + 56, 9.5, "none", stroke="ink")
    s.save("ptgs-idea.svg")


def accuracy() -> None:
    """pass@1 and pass@8 per arm: both seeds, their mean, the seed-17 CI, the base."""
    panels = [
        ("PASS@1", "per_seed", R["arms"]["base"]["score"], 0.15, 0.40),
        ("PASS@8", "per_seed_pass_at_k", R["arms"]["base"]["pass_at_k"], 0.45, 0.70),
    ]
    panel_h = 190
    s = Svg(100 + 2 * panel_h)
    y0 = title(
        s,
        "MATH-500 levels 3 to 5, n = 320, k = 8",
        "Dots: seeds 17, 18. Bar: mean. Whisker: seed-17 95% CI.",
    )
    px0, px1 = 24 + LABEL_W, W - 76
    for p, (head, key, base, lo, hi) in enumerate(panels):
        top = y0 + p * panel_h

        def sx(v, lo=lo, hi=hi):
            return px0 + (v - lo) / (hi - lo) * (px1 - px0)

        s.text(24, top + 14, head, size=12.5, color="muted", weight=700)
        bx = sx(base)
        s.line(bx, top + 28, bx, top + 150, "muted", dash="4 4")
        s.text(bx, top + 170, f"base {base:.3f}", size=13, color="muted", anchor="middle")
        for i, (arm, name, col) in enumerate(ARMS):
            y = top + 48 + i * 38
            vals = R["arms"][arm][key]
            mean = statistics.fmean(vals)
            s.text(24, y + 5, name, size=15, weight=600)
            s.rect(px0, y - 12, sx(mean) - px0, 24, col + "40", rx=5)
            right = max(vals)
            if key == "per_seed":
                ci_lo, ci_hi = R["arms"][arm]["ci"]
                s.line(sx(ci_lo), y, sx(ci_hi), y, col, width=2)
                right = max(right, ci_hi)
            for v in vals:
                s.dot(sx(v), y, 7.5, col, stroke="#FFFFFF")
            s.text(sx(right) + 12, y + 5, f"{mean:.3f}", size=15, weight=700, color=col)
    s.save("ptgs-accuracy.svg")


def waste_and_cost() -> None:
    """Zero-advantage groups per step and rollouts generated per prompt."""
    sampler = R["sampler"]
    flat_key = {
        "baseline": "frac_reward_zero_std",
        "recipe": "ptgs/zero_signal_frac",
        "ada": "ada/no_gradient",
    }
    flat = {
        arm: statistics.fmean(statistics.fmean(t[flat_key[arm]]) for t in sampler[arm])
        for arm, *_ in ARMS
    }
    drawn = {
        "baseline": 4.0,
        "recipe": 4.0,
        "ada": statistics.fmean(
            statistics.fmean(t["ada/drawn_per_prompt"]) for t in sampler["ada"]
        ),
    }
    panel_h = 160
    s = Svg(100 + 2 * panel_h + 10)
    y0 = title(
        s,
        "Training signal and sampling cost",
        "Mean over 80 steps and both seeds, 12 prompts per step.",
    )
    px0 = 24 + LABEL_W
    panels = (
        ("ZERO-ADVANTAGE GROUPS, SHARE OF PROMPTS", flat, 1.0, "{:.2f}"),
        ("ROLLOUTS GENERATED PER PROMPT", drawn, 26.0, "{:.1f}"),
    )
    for p, (head, vals, vmax, fmt) in enumerate(panels):
        top = y0 + p * panel_h
        s.text(24, top + 14, head, size=12.5, color="muted", weight=700)
        for i, (arm, name, col) in enumerate(ARMS):
            y = top + 48 + i * 38
            v = vals[arm]
            w = v / vmax * (W - px0 - 90)
            s.text(24, y + 5, name, size=15, weight=600)
            s.rect(px0, y - 12, w, 24, col, rx=5)
            s.text(px0 + w + 10, y + 5, fmt.format(v), size=15, weight=700, color=col)
    minutes = {arm: R["arms"][arm]["gpu_minutes"] for arm in ("recipe", "ada")}
    s.text(
        24,
        s.h - 16,
        f"Training time per seed on L40S: PTGS {minutes['recipe']:.1f} min, "
        f"Reinforce-Ada {minutes['ada']:.1f} min.",
        size=12.5,
        color="muted",
    )
    s.save("ptgs-cost.svg")


def tax() -> None:
    """Tax_S(8) per arm against the base model, with its 95% interval."""
    s = Svg(290)
    y0 = title(
        s,
        "Sharpening Tax, Tax_S(8), vs base",
        "Point estimate and 95% paired task-bootstrap CI.",
    )
    lo, hi = -0.05, 0.05
    px0, px1 = 24 + LABEL_W, W - 24

    def sx(v):
        return px0 + (v - lo) / (hi - lo) * (px1 - px0)

    s.rect(sx(0), y0, px1 - sx(0), 140, C["warm_tint"], rx=0)
    s.text(px1 - 8, y0 + 20, "coverage lost", size=13, color="warm", anchor="end", weight=700)
    s.text(px0 + 4, y0 + 20, "coverage kept", size=13, color="green", weight=700)
    s.line(sx(0), y0, sx(0), y0 + 140, "ink", width=2)
    for i, (arm, name, col) in enumerate(ARMS):
        y = y0 + 52 + i * 34
        t = R["arms"][arm]["sharpening_tax"]
        s.text(24, y + 5, name, size=15, weight=600)
        s.line(sx(t["ci"][0]), y, sx(t["ci"][1]), y, col, width=4)
        s.dot(sx(t["tax_s"]), y, 8, col, stroke="#FFFFFF")
    s.text(
        W / 2,
        s.h - 22,
        "Every CI includes 0: no arm paid a measurable tax.",
        size=15,
        anchor="middle",
        weight=700,
    )
    s.save("ptgs-tax.svg")


if __name__ == "__main__":
    idea()
    accuracy()
    waste_and_cost()
    tax()
    print(f"wrote {OUT}/ptgs-{{idea,accuracy,cost,tax}}.svg")
