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
    ("paper", "Paper's bonus", C["green"]),
    ("recipe", "Our stricter bonus", C["warm"]),
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
        "Normal LLM vs context LM",
        "Same log, read in 5 chunks. One piles it all up. One keeps notes.",
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
    s.text(24, s.h - 18, "Illustration. Once a chunk is read it's gone; only the notes remain.",
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
        "Trained, it gets almost every log right",
        f"Untrained: {base:.0%} right. Each dot is one training run; the line is the average.",
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
        "Tokens used per log (lower is cheaper)",
        f"Untrained: {base:,}. Each dot is one training run; the line is the average.",
    )
    dot_panel(s, y0, "cost_tokens", "cost_per_seed", 800, 2200, [800, 1200, 1600, 2000],
              lambda v: f"{v:,.0f}")  # fmt: skip
    d = R["deltas"]["paper"]["cost"]["pooled"]["relative"]
    s.text(24, s.h - 18, f"The paper's bonus uses {abs(d):.0%} fewer tokens, mostly because one "
           "plain run let its notes bloat.", size=12.5, color="body")  # fmt: skip
    s.save("context-lm-tokens.svg")


def before_after() -> None:
    """The last context.md the untrained model and the trained model (paper
    arm, seed 17) wrote for the same held-out log, each line checked
    against the log's true final state."""
    import whileai as wai

    env = wai.methods.KVLog()
    base, trained = R["samples"]["base"][0], R["samples"]["paper"][0]
    assert base["task"] == trained["task"]
    seed = int(base["task"].rsplit("-", 1)[1])
    task = env.task(seed)
    truth = env.final_state(task)

    def lines(sample):
        out = []
        for raw in sample["files"][-1].splitlines():
            ln = raw.strip().strip("|").strip()
            if ": " not in ln:
                continue
            k, v = (x.strip() for x in ln.split(": ", 1))
            out.append((k, v, truth.get(k) == v, truth.get(k)))
        return out

    cols = [
        ("Untrained", lines(base), env.reward(task, base["answer"])),
        ("Trained, 60 steps", lines(trained), env.reward(task, trained["answer"])),
    ]
    n = max(len(c[1]) for c in cols)
    s = Svg(100 + 40 + n * 26 + 62)
    y0 = title(
        s,
        "Its notes after reading the whole log",
        f"Same unseen log, checked against the true values. Asked: what is {task['ask']}?",
    )
    cw = (W - 48 - 16) / 2
    for c, (head, rows, right) in enumerate(cols):
        x = 24 + c * (cw + 16)
        s.text(x, y0 + 14, head.upper(), size=12.5, color="muted", weight=700)
        s.rect(x, y0 + 26, cw, n * 26 + 18, "surface", rx=8, stroke="line")
        for i, (k, v, ok, real) in enumerate(rows):
            y = y0 + 50 + i * 26
            col = "green" if ok else "warm"
            s.text(x + 14, y, f"{k}: {v}", size=13.5, mono=True,
                   weight=700 if k == task["ask"] else 400)  # fmt: skip
            note = "current" if ok else (f"stale, now {real}" if real else "not in the log")
            s.text(x + cw - 14, y, note, size=12, color=col, anchor="end", weight=600)
        y = y0 + 26 + n * 26 + 18 + 32
        verdict = f"answered {task['gold']}, right" if right else "answered wrong"
        s.text(x, y, verdict, size=15, weight=700, color="green" if right else "warm")
        kept = sum(ok for _, _, ok, _ in rows)
        s.text(x + cw, y, f"{kept} of {len(truth)} current", size=13, color="body", anchor="end")
    s.save("context-lm-before-after.svg")


def learning() -> None:
    """Right answers during training, paper arm: each seed and their mean,
    smoothed over 5 steps."""
    traces = [t["reward"] for t in R["train_trace"]["paper"]]
    steps = len(traces[0])

    def smooth(xs, k=5):
        return [
            sum(xs[max(0, i - k + 1) : i + 1]) / len(xs[max(0, i - k + 1) : i + 1])
            for i in range(len(xs))
        ]

    s = Svg(100 + 230)
    y0 = title(
        s,
        "It learns to keep notes in about 15 steps",
        "Practice logs answered right during training. Faint: each run. Bold: average.",
    )
    px0, px1, py0, py1 = 70, W - 30, y0 + 10, y0 + 180

    def sx(i):
        return px0 + i / (steps - 1) * (px1 - px0)

    def sy(v):
        return py1 - v * (py1 - py0)

    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        s.line(px0, sy(v), px1, sy(v), "line", width=1)
        s.text(px0 - 10, sy(v) + 4, f"{v:.2f}", size=12, color="muted", anchor="end")
    for i in (0, 20, 40, steps - 1):
        anchor = "end" if i == steps - 1 else "middle"
        s.text(sx(i), py1 + 22, f"step {i + 1}", size=12, color="muted", anchor=anchor)

    def path(ys):
        return " ".join(
            f"{'M' if i == 0 else 'L'}{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(ys)
        )

    for t in traces:
        s.parts.append(
            f'<path d="{path(smooth(t))}" fill="none" stroke="{C["green"]}" stroke-opacity="0.35" '
            f'stroke-width="1.5" stroke-linejoin="round"/>'
        )
    mean = [sum(t[i] for t in traces) / len(traces) for i in range(steps)]
    s.parts.append(
        f'<path d="{path(smooth(mean))}" fill="none" stroke="{C["green"]}" stroke-width="3.5" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
    )
    s.text(px1, sy(0.62), "all 4 seeds end near 1.0", size=13, color="green",
           anchor="end", weight=700)  # fmt: skip
    s.save("context-lm-learning.svg")


if __name__ == "__main__":
    idea()
    accuracy()
    tokens()
    before_after()
    learning()
    print(f"wrote {OUT}/context-lm-{{idea,accuracy,tokens,before-after,learning}}.svg")
