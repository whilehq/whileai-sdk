#!/usr/bin/env python
"""Draw the guide figures, one light and one dark SVG each, into docs/figures/.

Each figure is one mechanism from one guide, drawn with the names the code
uses. The palette is docs/style.css; the dark set is the same tokens on the
docs.json dark background. Rerun after editing and commit both files:

    python scripts/gen_guide_figures.py
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

LIGHT = {
    "bg": "#FFFFFF",
    "ink": "#0B1220",
    "body": "#4B5563",
    "muted": "#6B7280",
    "line": "#D1D5DB",
    "surface": "#F4F6F9",
    "green": "#3F8F6B",
    "green_soft": "#5CB08A",
    "tint": "#F0FAF5",
    "tint_strong": "#D7F0E3",
    "warm": "#B5602A",
    "warm_tint": "#FBEDE3",
}
DARK = {
    "bg": "#0B1220",
    "ink": "#E5E7EB",
    "body": "#CBD5E1",
    "muted": "#9CA3AF",
    "line": "#334155",
    "surface": "#121A2B",
    "green": "#5CB08A",
    "green_soft": "#3F8F6B",
    "tint": "#12261D",
    "tint_strong": "#1D3A2C",
    "warm": "#E09A6A",
    "warm_tint": "#3A2418",
}
SANS = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Inter, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


class Canvas:
    def __init__(self, w: int, h: int, p: dict, label: str):
        self.w, self.h, self.p = w, h, p
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
            f'height="{h}" role="img" aria-label="{label}">',
            "<defs>"
            f'<marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" '
            f'markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{p["muted"]}"/></marker>'
            f'<marker id="arrow-green" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" '
            f'markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{p["green"]}"/></marker>'
            "</defs>",
            f'<rect width="{w}" height="{h}" rx="6" fill="{p["bg"]}"/>',
        ]

    def box(
        self, x, y, w, h, title, sub=None, *, fill="surface", stroke="line", mono=False, small=False
    ):
        p = self.p
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{p[fill]}" stroke="{p[stroke]}"/>'
        )
        fam = MONO if mono else SANS
        size = 11 if small else 13
        cy = y + h / 2
        if sub:
            self.text(x + w / 2, cy - 5, title, size=size, fam=fam, weight=600, anchor="middle")
            self.text(x + w / 2, cy + 12, sub, size=10.5, fam=MONO, color="muted", anchor="middle")
        else:
            self.text(x + w / 2, cy + 5, title, size=size, fam=fam, weight=600, anchor="middle")

    def text(self, x, y, s, *, size=12, fam=SANS, color="ink", weight=400, anchor="start"):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
            f'fill="{self.p[color]}" text-anchor="{anchor}">{s}</text>'
        )

    def arrow(self, x1, y1, x2, y2, *, green=False, dashed=False):
        c = self.p["green" if green else "muted"]
        m = "arrow-green" if green else "arrow"
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="1.5" '
            f'marker-end="url(#{m})"{dash}/>'
        )

    def path(self, d, *, green=False, dashed=False, arrow=True):
        c = self.p["green" if green else "muted"]
        m = "arrow-green" if green else "arrow"
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        head = f' marker-end="url(#{m})"' if arrow else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{c}" stroke-width="1.5"{head}{dash}/>'
        )

    def dot(self, x, y, r, *, fill="green", stroke=None):
        s = f' stroke="{self.p[stroke]}" stroke-width="1.5"' if stroke else ""
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{self.p[fill]}"{s}/>')

    def render(self) -> str:
        return "\n".join([*self.parts, "</svg>"]) + "\n"


def simulations_pipeline(p):
    c = Canvas(960, 250, p, "How simulate() turns an agent definition into graded rows and cuts")
    c.text(16, 24, "IN", size=10, color="muted", weight=600)
    c.box(16, 34, 150, 44, "describe the behavior", "system_prompt, tools", small=True)
    c.box(16, 92, 150, 44, "or its traces", "traces=, OTel spans", small=True)
    c.arrow(166, 56, 196, 78)
    c.arrow(166, 114, 196, 92)
    c.box(200, 60, 130, 50, "six axes", "pairwise grid", fill="tint", stroke="green")
    c.arrow(330, 85, 360, 85)
    c.box(364, 60, 120, 50, "world", "seeded, says no")
    c.arrow(484, 85, 514, 85)
    c.box(518, 60, 120, 50, "agent", "agent= or callable")
    c.arrow(638, 85, 668, 85)
    c.box(672, 60, 120, 50, "rows", "ungraded", fill="tint", stroke="green")
    c.arrow(792, 85, 822, 85)
    c.box(826, 60, 118, 50, "your judge", "grade(judge=)")
    c.text(200, 140, "what varies", size=10, color="muted", weight=600)
    for i, ax in enumerate(
        ["tool", "policy rule", "user stance", "world state", "tool condition", "history"]
    ):
        c.text(200, 158 + i * 15, ax, size=11, fam=MONO, color="body")
    c.text(364, 140, "records shaped like the schema", size=10, color="muted", weight=600)
    c.text(364, 158, "unknown id: not found", size=11, fam=MONO, color="body")
    c.text(364, 174, "schema echo: refused", size=11, fam=MONO, color="body")
    c.text(672, 140, "OUT, from the graded rows", size=10, color="muted", weight=600)
    for i, (name, call) in enumerate(
        [
            ("SFT", "select_for_sft"),
            ("DPO pairs", "build_preference_pairs"),
            ("GRPO groups", "select_for_rl"),
            ("RL env", "export_environment"),
        ]
    ):
        y = 158 + i * 18
        c.text(672, y, name, size=11, color="ink", weight=600)
        c.text(760, y, call, size=11, fam=MONO, color="body")
    c.path("M 885 110 L 885 128 L 740 128 L 740 146", green=True)
    return c.render()


def engine_eight_steps(p):
    c = Canvas(960, 230, p, "The eight steps of the engine, with Delta feeding the next run")
    steps = [
        ("01 Axes", "what varies"),
        ("02 Cover", "pairwise array"),
        ("03 Search", "five arms"),
        ("04 World", "seeded sandbox"),
        ("05 Rollout", "N x n x k"),
        ("06 Grade", "rules, then judge"),
        ("07 Cut", "SFT / DPO / GRPO"),
        ("08 Delta", "paired, bootstrap"),
    ]
    w, h, gap = 104, 54, 14
    x0, y0 = 16, 40
    for i, (t, s) in enumerate(steps):
        x = x0 + i * (w + gap)
        fill = "tint" if i in (1, 2, 3, 6) else "surface"
        stroke = "green" if i in (1, 2, 3, 6) else "line"
        c.box(x, y0, w, h, t, s, fill=fill, stroke=stroke, small=True)
        if i < 7:
            c.arrow(x + w, y0 + h / 2, x + w + gap, y0 + h / 2)
    c.text(16, 24, "green: ours", size=10, color="green", weight=600)
    c.text(90, 24, "the rest: from the literature", size=10, color="muted")
    c.text(
        16,
        128,
        "01, 04, 05: the run.  02, 03: which situations.  06: reward.  07: training rows.  08: the proof.",
        size=11,
        color="body",
    )
    x_last = x0 + 7 * (w + gap) + w / 2
    x_first = x0 + 4 * (w + gap) + w / 2
    c.path(
        f"M {x_last} {y0 + h} L {x_last} 150 L {x_first} 150 L {x_first} {y0 + h + 4}",
        green=True,
        dashed=True,
    )
    c.text(
        (x_last + x_first) / 2,
        166,
        "after training: re-run the held-out tasks",
        size=10.5,
        color="green",
        anchor="middle",
    )
    c.text(
        16,
        200,
        "pass@1 [lo..hi]  ->  delta per task  ->  ship when the interval excludes zero",
        size=11,
        fam=MONO,
        color="body",
    )
    return c.render()


def evals_loop(p):
    c = Canvas(
        960, 210, p, "The eval loop: wrap, simulate with repeats, judge, read pass rates, gate CI"
    )
    boxes = [
        ("agent(message)", "your callable", False),
        ("simulate", "seeds=, repeats=4, fixed", True),
        ("evaluate", "judge(row) -> reward", False),
        ("pass_at", "pass@1 [lo..hi] pass^4", True),
        ("run.py --gate 0.9", "exit 1 low, exit 2 hollow", False),
    ]
    w, h, gap, y = 168, 54, 22, 40
    for i, (t, s, hl) in enumerate(boxes):
        x = 16 + i * (w + gap)
        c.box(
            x,
            y,
            w,
            h,
            t,
            s,
            mono=True,
            small=True,
            fill="tint" if hl else "surface",
            stroke="green" if hl else "line",
        )
        if i < 4:
            c.arrow(x + w, y + h / 2, x + w + gap, y + h / 2)
    c.text(
        16,
        122,
        "read scored.warnings first: no tool called, a tool never touched, a marker on no row",
        size=11,
        color="warm",
    )
    # judge check branch
    jx = 16 + 2 * (w + gap)
    c.path(f"M {jx + w / 2} {y + h} L {jx + w / 2} 150", dashed=True)
    c.box(
        jx - 60,
        152,
        300,
        40,
        'attach_labels(kind="human")  ->  judge_trust',
        None,
        mono=True,
        small=True,
    )
    c.text(jx + 254, 168, "agreement, kappa, Wilson lower bound", size=11, color="body")
    c.text(jx + 254, 184, "16 labels to clear 0.8 at perfect agreement", size=11, color="muted")
    return c.render()


def evals_pass_at_k(p):
    c = Canvas(
        960, 250, p, "pass@1, pass^k and pass@k read off one grid of five tasks by four tries"
    )
    grid = [[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]]
    x0, y0, cell = 120, 40, 34
    c.text(x0 - 8, 28, "task", size=10, color="muted", anchor="end")
    for j in range(4):
        c.text(
            x0 + j * cell + cell / 2, 28, f"try {j + 1}", size=10, color="muted", anchor="middle"
        )
    c.text(x0 + 4 * cell + 24, 28, "per task", size=10, color="muted")
    c.text(x0 + 4 * cell + 96, 28, "all 4", size=10, color="muted")
    c.text(x0 + 4 * cell + 150, 28, "any of 4", size=10, color="muted")
    for i, row in enumerate(grid):
        y = y0 + i * cell + cell / 2
        c.text(x0 - 8, y + 4, f"t{i + 1}", size=11, fam=MONO, color="body", anchor="end")
        for j, v in enumerate(row):
            x = x0 + j * cell + cell / 2
            if v:
                c.dot(x, y, 9)
            else:
                c.dot(x, y, 9, fill="warm_tint", stroke="warm")
        k = sum(row)
        c.text(x0 + 4 * cell + 24, y + 4, f"{k}/4", size=11, fam=MONO, color="body")
        c.text(x0 + 4 * cell + 96, y + 4, "1" if k == 4 else "0", size=11, fam=MONO, color="body")
        c.text(x0 + 4 * cell + 150, y + 4, "1" if k > 0 else "0", size=11, fam=MONO, color="body")
    tx = 520
    lines = [
        ("pass@1", "mean of the per-task rates", "(1.00+0.75+0.50+0+0.25)/5 = 0.50"),
        ("pass^4", "tasks that passed all four tries", "1/5 = 0.20"),
        ("pass@4", "tasks that passed at least once", "4/5 = 0.80"),
        ("headroom", "pass@4 - pass@1, what training can win", "0.80 - 0.50 = 0.30"),
    ]
    for i, (name, what, arith) in enumerate(lines):
        y = 52 + i * 46
        c.text(tx, y, name, size=13, fam=MONO, color="green", weight=600)
        c.text(tx + 90, y, what, size=11, color="body")
        c.text(tx + 90, y + 16, arith, size=11, fam=MONO, color="muted")
    c.text(
        16,
        232,
        "the interval is a bootstrap over tasks, not tries: five tasks is a wide one",
        size=11,
        color="muted",
    )
    return c.render()


def reward_hacking_curve(p):
    c = Canvas(
        960,
        270,
        p,
        "Proxy reward keeps climbing while gold reward turns over as KL grows; the five checks placed before, during and after",
    )
    ox, oy, W, H = 60, 200, 520, 150
    c.arrow(ox, oy, ox + W + 20, oy)
    c.arrow(ox, oy, ox, oy - H - 10)
    c.text(
        ox + W / 2, oy + 40, "KL from the reference policy", size=11, color="muted", anchor="middle"
    )
    c.text(ox - 6, oy - H - 14, "reward", size=11, color="muted", anchor="end")
    # proxy: monotone rise
    c.path(
        f"M {ox} {oy - 10} C {ox + 120} {oy - 90}, {ox + 260} {oy - 120}, {ox + W} {oy - 140}",
        green=True,
        arrow=False,
    )
    c.text(ox + W + 6, oy - 138, "proxy", size=12, fam=MONO, color="green", weight=600)
    c.text(ox + W + 6, oy - 124, "training reward", size=10, color="muted")
    # gold: rise then fall
    c.parts.append(
        f'<path d="M {ox} {oy - 10} C {ox + 110} {oy - 85}, {ox + 200} {oy - 105}, {ox + 260} {oy - 100} '
        f'S {ox + 420} {oy - 40}, {ox + W} {oy - 15}" fill="none" stroke="{p["warm"]}" stroke-width="2"/>'
    )
    c.text(ox + W + 6, oy - 34, "gold", size=12, fam=MONO, color="warm", weight=600)
    c.text(ox + W + 6, oy - 20, "held-out scorer", size=10, color="muted")
    # gap
    c.parts.append(
        f'<line x1="{ox + 430}" y1="{oy - 126}" x2="{ox + 430}" y2="{oy - 36}" stroke="{p["muted"]}" '
        f'stroke-width="1" stroke-dasharray="3 3"/>'
    )
    c.text(ox + 438, oy - 80, "the gap", size=11, color="body")
    # phases
    for x, label in [(ox + 40, "before"), (ox + 250, "during"), (ox + 470, "after")]:
        c.text(x, oy + 22, label, size=10.5, color="muted", weight=600, anchor="middle")
    c.parts.append(
        f'<line x1="{ox}" y1="{oy + 10}" x2="{ox + W}" y2="{oy + 10}" stroke="{p["line"]}" stroke-width="1"/>'
    )
    # checks
    checks = [
        ("before", ["hack_scan(endorsed=)", "judge_probes", "trace_flag_report"]),
        ("during", ["HackMonitor(holdout=, gold=)"]),
        ("after", ["delta_report(proxy=)", "hack_scan_diff"]),
    ]
    x = 700
    for i, (phase, calls) in enumerate(checks):
        y = 46 + i * 68
        c.text(x, y, phase, size=10.5, color="muted", weight=600)
        for j, call in enumerate(calls):
            c.text(x, y + 17 + j * 15, call, size=11, fam=MONO, color="ink")
    c.text(700, 246, "shape after Gao et al. 2023; not measured here", size=10, color="muted")
    return c.render()


def safety_channels(p):
    c = Canvas(
        960,
        260,
        p,
        "Three ways an instruction reaches a tool-using agent, three ways data leaves, one marker per exit",
    )
    c.text(16, 24, "WHERE THE INSTRUCTION COMES FROM", size=10, color="muted", weight=600)
    ins = [
        ("the ask", "prompt_injection, social_engineering"),
        ("a tool result", "indirect_injection: planted in a record"),
        ("public text", "a review, a listing (marketplace)"),
    ]
    for i, (t, s) in enumerate(ins):
        y = 40 + i * 60
        c.box(16, y, 250, 46, t, s, small=True)
        c.arrow(266, y + 23, 330, 130)
    c.box(334, 90, 180, 80, "agent", "reads data, acts, sends", fill="tint", stroke="green")
    c.text(560, 24, "WHERE DATA CAN LEAVE", size=10, color="muted", weight=600)
    outs = [
        ("the reply", "no_secret_leak"),
        ("an outbound send", "no_external_send"),
        ("a write", "no_unauthorized_write"),
    ]
    for i, (t, m) in enumerate(outs):
        y = 40 + i * 60
        c.arrow(514, 130, 556, y + 23)
        c.box(560, y, 200, 46, t, m, small=True)
        c.text(772, y + 28, "1.0 when it held", size=10.5, color="muted")
    c.text(
        16,
        232,
        "controls: benign asks in the same set, marker helpful_on_benign. A refusal that passes them is a reward the policy can collect.",
        size=11,
        color="body",
    )
    c.text(
        16,
        248,
        "reward = 1 only when every applicable marker holds; the judge reads steps and final_text, not the prose alone",
        size=11,
        color="muted",
    )
    return c.render()


def character_pipeline(p):
    c = Canvas(
        960,
        250,
        p,
        "Character training as a data pipeline: constitution, prompts, replies under the deployment prompt, a judge that alone sees the principle, pairs and SFT, then a paired delta",
    )
    boxes = [
        ("constitution", "one principle per trait"),
        ("prompts", "where the trait matters"),
        ("k replies", "deployment prompt only"),
        ("judge", "alone sees the principle"),
        ("markers", "trait, on_task, no_filler"),
    ]
    w, h, gap, y = 168, 54, 20, 40
    for i, (t, s) in enumerate(boxes):
        x = 16 + i * (w + gap)
        hl = i in (2, 3)
        c.box(
            x,
            y,
            w,
            h,
            t,
            s,
            small=True,
            fill="tint" if hl else "surface",
            stroke="green" if hl else "line",
        )
        if i < 4:
            c.arrow(x + w, y + h / 2, x + w + gap, y + h / 2)
    c.text(
        16,
        118,
        "if the principle is in the sampling prompt you measure prompting, not character",
        size=11,
        color="warm",
    )
    x_m = 16 + 4 * (w + gap) + w / 2
    c.path(f"M {x_m} {y + h} L {x_m} 140", green=True)
    c.box(560, 142, 184, 44, "build_preference_pairs", "length_match=True", mono=True, small=True)
    c.box(
        760,
        142,
        184,
        44,
        "export_training",
        "passes, loss mask on agent turn",
        mono=True,
        small=True,
    )
    c.arrow(652, 186, 652, 206)
    c.box(
        560,
        208,
        384,
        40,
        'train(method="dpo")  ->  delta_report',
        'target="marker:trait", must_not_regress=[on_task, no_filler]',
        mono=True,
        small=True,
    )
    c.text(16, 160, "judge check first:", size=11, color="ink", weight=600)
    c.text(
        16,
        176,
        "grade the spec's GOOD/BAD replies, judge_agreement >= 0.8, kappa >= 0.6",
        size=11,
        fam=MONO,
        color="body",
    )
    c.text(16, 200, "holdout:", size=11, color="ink", weight=600)
    c.text(
        16,
        216,
        'same prompts + "drop the act", plus plain tasks the persona must not distort',
        size=11,
        color="body",
    )
    return c.render()


FIGURES = {
    "simulations-pipeline": simulations_pipeline,
    "engine-eight-steps": engine_eight_steps,
    "evals-loop": evals_loop,
    "evals-pass-at-k": evals_pass_at_k,
    "reward-hacking-curve": reward_hacking_curve,
    "safety-channels": safety_channels,
    "character-pipeline": character_pipeline,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, draw in FIGURES.items():
        for variant, palette in (("light", LIGHT), ("dark", DARK)):
            path = OUT / f"{name}-{variant}.svg"
            path.write_text(draw(palette), encoding="utf-8", newline="\n")
            print(path.relative_to(OUT.parent.parent))


if __name__ == "__main__":
    main()
