#!/usr/bin/env python
"""Draw the guide figures, one light and one dark SVG each, into docs/figures/.

Each figure is one mechanism from one guide, drawn with the names the code
uses, 720 px wide so it fills the docs content column without shrinking the
type. The palette is docs/style.css; the dark set is the same tokens on the
docs.json dark background. Rerun after editing and commit both files:

    python scripts/gen_guide_figures.py
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
W = 720

LIGHT = {
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
DARK = {
    "bg": "#0B1220",
    "ink": "#E5E7EB",
    "body": "#CBD5E1",
    "muted": "#9CA3AF",
    "line": "#334155",
    "surface": "#121A2B",
    "green": "#5CB08A",
    "tint": "#12261D",
    "warm": "#E09A6A",
    "warm_tint": "#3A2418",
}
SANS = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Inter, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


class Canvas:
    def __init__(self, h: int, p: dict, label: str):
        self.p = p
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" '
            f'height="{h}" role="img" aria-label="{label}">',
            "<defs>"
            f'<marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" '
            f'markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{p["muted"]}"/></marker>'
            f'<marker id="arrow-green" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" '
            f'markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{p["green"]}"/></marker>'
            "</defs>",
            f'<rect width="{W}" height="{h}" rx="6" fill="{p["bg"]}"/>',
        ]

    def box(self, x, y, w, h, title, sub=None, *, hl=False, mono=False):
        p = self.p
        fill, stroke = ("tint", "green") if hl else ("surface", "line")
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{p[fill]}" stroke="{p[stroke]}"/>'
        )
        fam = MONO if mono else SANS
        cx, cy = x + w / 2, y + h / 2
        subs = [sub] if isinstance(sub, str) else list(sub or [])
        if not subs:
            self.text(cx, cy + 5, title, size=13, fam=fam, weight=600, anchor="middle")
            return
        top = cy - 6 * len(subs) + 1
        self.text(cx, top, title, size=13, fam=fam, weight=600, anchor="middle")
        for i, s in enumerate(subs):
            self.text(cx, top + 16 + i * 14, s, size=11, fam=MONO, color="muted", anchor="middle")

    def text(self, x, y, s, *, size=12, fam=SANS, color="ink", weight=400, anchor="start"):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
            f'fill="{self.p[color]}" text-anchor="{anchor}">{s}</text>'
        )

    def arrow(self, x1, y1, x2, y2, *, green=False, dashed=False):
        self.path(f"M {x1} {y1} L {x2} {y2}", green=green, dashed=dashed)

    def path(self, d, *, green=False, dashed=False, arrow=True, width=1.5):
        c = self.p["green" if green else "muted"]
        m = "arrow-green" if green else "arrow"
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        head = f' marker-end="url(#{m})"' if arrow else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{width}"{head}{dash}/>'
        )

    def dot(self, x, y, r, *, fill="green", stroke=None):
        s = f' stroke="{self.p[stroke]}" stroke-width="1.5"' if stroke else ""
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{self.p[fill]}"{s}/>')

    def render(self) -> str:
        return "\n".join([*self.parts, "</svg>"]) + "\n"


def row(c, y, boxes, *, x0=16, w=200, h=56, gap=32, mono=False):
    """Boxes left to right with arrows between them; returns the x of each."""
    xs = []
    for i, (title, sub, hl) in enumerate(boxes):
        x = x0 + i * (w + gap)
        c.box(x, y, w, h, title, sub, hl=hl, mono=mono)
        if i < len(boxes) - 1:
            c.arrow(x + w, y + h / 2, x + w + gap, y + h / 2)
        xs.append(x)
    return xs


def simulations_pipeline(p):
    c = Canvas(344, p, "How simulate() turns an agent definition into graded rows and cuts")
    c.text(16, 24, "IN", size=11, color="muted", weight=600)
    c.box(16, 32, 200, 56, "describe the behavior", "system_prompt, tools")
    c.text(232, 64, "or", size=12, color="muted")
    c.box(256, 32, 200, 56, "point at its traces", "traces=, OTel spans")
    c.text(480, 56, "either way, the same engine", size=11.5, color="muted")
    c.path("M 116 88 L 116 110", green=True)
    c.path("M 356 88 L 356 110", green=True)
    row(
        c,
        112,
        [
            ("six axes", "a pairwise grid", True),
            ("world", "seeded, says no", False),
            ("agent", "agent= or a callable", False),
        ],
    )
    c.path("M 580 168 L 580 184 L 116 184 L 116 200", green=True)
    xs = row(c, 202, [("rows", "ungraded", True), ("your judge", "grade(judge=)", False)])
    ox = xs[1] + 232
    c.text(ox, 214, "cuts, from the graded rows", size=11, color="muted", weight=600)
    for i, (name, call) in enumerate(
        [
            ("SFT", "select_for_sft"),
            ("DPO", "build_preference_pairs"),
            ("GRPO", "select_for_rl"),
            ("RL env", "export_environment"),
        ]
    ):
        y = 232 + i * 16
        c.text(ox, y, name, size=11.5, weight=600)
        c.text(ox + 48, y, call, size=11.5, fam=MONO, color="body")
    c.arrow(xs[1] + 200, 230, ox - 6, 230)
    c.text(
        16,
        306,
        "axes: tool, policy rule, user stance, world state, tool condition, history",
        size=11.5,
        fam=MONO,
        color="body",
    )
    c.text(
        16,
        324,
        "world: records shaped like the schema; unknown id not found; schema echo refused",
        size=11.5,
        fam=MONO,
        color="body",
    )
    return c.render()


def engine_eight_steps(p):
    c = Canvas(300, p, "The eight steps of the engine, with Delta feeding the next run")
    steps = [
        ("01 Axes", "what varies", True),
        ("02 Cover", "pairwise array", True),
        ("03 Search", "five arms", True),
        ("04 World", "seeded sandbox", True),
        ("05 Rollout", "N x n x k", False),
        ("06 Grade", "rules, then judge", False),
        ("07 Cut", "SFT / DPO / GRPO", True),
        ("08 Delta", "paired, bootstrap", False),
    ]
    # ours: 02, 03, 04, 07 (green). 01 is the declaration the rest hangs on.
    steps[0] = ("01 Axes", "what varies", False)
    w, h, gap = 160, 56, 16
    r1 = row(c, 40, steps[:4], w=w, h=h, gap=gap)
    r2 = row(c, 150, steps[4:], w=w, h=h, gap=gap)
    c.path(
        f"M {r1[3] + w / 2} 96 L {r1[3] + w / 2} 120 L {r2[0] + w / 2} 120 L {r2[0] + w / 2} 148",
        green=False,
    )
    c.text(16, 24, "green: ours", size=11, color="green", weight=600)
    c.text(100, 24, "the rest: from the literature", size=11, color="muted")
    c.path(
        f"M {r2[3] + w / 2} 206 L {r2[3] + w / 2} 236 L {r2[0] + w / 2} 236 L {r2[0] + w / 2} 208",
        green=True,
        dashed=True,
    )
    c.text(
        (r2[0] + r2[3] + w) / 2,
        254,
        "after training: the same held-out tasks again, paired per task",
        size=11.5,
        color="green",
        anchor="middle",
    )
    c.text(
        16,
        286,
        "pass@1 [lo..hi]  ->  delta per task  ->  ship when the interval excludes zero",
        size=11.5,
        fam=MONO,
        color="body",
    )
    return c.render()


def evals_loop(p):
    c = Canvas(
        300, p, "The eval loop: wrap, simulate with repeats, judge, read pass rates, gate CI"
    )
    row(
        c,
        32,
        [
            ("agent(message)", "your callable", False),
            ("simulate", "seeds=, repeats=4, fixed", True),
            ("evaluate", "judge(row) -> reward", False),
        ],
        mono=True,
    )
    c.path("M 580 88 L 580 108 L 116 108 L 116 130", green=True)
    xs = row(
        c,
        132,
        [
            ("pass_at", "pass@1 [lo..hi]  pass^4", True),
            ("run.py --gate 0.9", ["exit 1 under the floor", "exit 2 when hollow"], False),
        ],
        mono=True,
    )
    c.text(xs[1] + 232, 150, "hollow: no tool called, a tool", size=11.5, color="warm")
    c.text(xs[1] + 232, 166, "never touched, a marker on no", size=11.5, color="warm")
    c.text(xs[1] + 232, 182, "row. Not a result.", size=11.5, color="warm")
    c.path("M 116 188 L 116 220", dashed=True)
    c.box(16, 222, 380, 40, 'attach_labels(kind="human")  ->  judge_trust', None, mono=True)
    c.text(412, 238, "agreement, kappa, Wilson lower bound;", size=11.5, color="body")
    c.text(412, 254, "16 labels to clear 0.8 at perfect agreement", size=11.5, color="muted")
    c.text(16, 286, "check the judge before you read the number", size=11.5, color="muted")
    return c.render()


def evals_pass_at_k(p):
    c = Canvas(270, p, "pass@1, pass^k and pass@k read off one grid of five tasks by four tries")
    grid = [[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]]
    x0, y0, cell = 60, 44, 32
    cols = [
        (x0 + 4 * cell + 22, "per task"),
        (x0 + 4 * cell + 88, "all 4"),
        (x0 + 4 * cell + 140, "any of 4"),
    ]
    for j in range(4):
        c.text(
            x0 + j * cell + cell / 2, 30, f"try {j + 1}", size=11, color="muted", anchor="middle"
        )
    for x, name in cols:
        c.text(x, 30, name, size=11, color="muted")
    for i, r in enumerate(grid):
        y = y0 + i * cell + cell / 2
        c.text(x0 - 10, y + 4, f"t{i + 1}", size=12, fam=MONO, color="body", anchor="end")
        for j, v in enumerate(r):
            x = x0 + j * cell + cell / 2
            if v:
                c.dot(x, y, 9)
            else:
                c.dot(x, y, 9, fill="warm_tint", stroke="warm")
        k = sum(r)
        for (x, _), val in zip(cols, (f"{k}/4", "1" if k == 4 else "0", "1" if k else "0")):
            c.text(x, y + 4, val, size=12, fam=MONO, color="body")
    tx = 400
    lines = [
        ("pass@1", "mean of the per-task rates", "(1 + .75 + .5 + 0 + .25) / 5 = 0.50"),
        ("pass^4", "tasks that passed all four tries", "1 / 5 = 0.20"),
        ("pass@4", "tasks that passed at least once", "4 / 5 = 0.80"),
        ("headroom", "pass@4 - pass@1: what training can win", "0.80 - 0.50 = 0.30"),
    ]
    for i, (name, what, arith) in enumerate(lines):
        y = 46 + i * 50
        c.text(tx, y, name, size=13, fam=MONO, color="green", weight=600)
        c.text(tx + 84, y, what, size=11.5, color="body")
        c.text(tx + 84, y + 17, arith, size=11.5, fam=MONO, color="muted")
    c.text(
        16,
        250,
        "the interval is a bootstrap over tasks, not tries; five tasks makes a wide one",
        size=11.5,
        color="muted",
    )
    return c.render()


def reward_hacking_curve(p):
    c = Canvas(
        300,
        p,
        "Proxy reward keeps climbing while gold reward turns over as KL grows; the five checks placed before, during and after",
    )
    ox, oy, cw, ch = 50, 220, 400, 170
    c.arrow(ox, oy, ox + cw + 16, oy)
    c.arrow(ox, oy, ox, oy - ch - 10)
    c.text(
        ox + cw / 2,
        oy + 44,
        "KL from the reference policy",
        size=11.5,
        color="muted",
        anchor="middle",
    )
    c.text(ox - 6, oy - ch - 14, "reward", size=11.5, color="muted", anchor="end")
    c.path(
        f"M {ox} {oy - 10} C {ox + 90} {oy - 100}, {ox + 200} {oy - 135}, {ox + cw} {oy - 155}",
        green=True,
        arrow=False,
        width=2,
    )
    c.text(ox + cw + 6, oy - 150, "proxy", size=12.5, fam=MONO, color="green", weight=600)
    c.text(ox + cw + 6, oy - 136, "training", size=10.5, color="muted")
    c.text(ox + cw + 6, oy - 124, "reward", size=10.5, color="muted")
    c.parts.append(
        f'<path d="M {ox} {oy - 10} C {ox + 85} {oy - 95}, {ox + 150} {oy - 118}, {ox + 200} {oy - 112} '
        f'S {ox + 320} {oy - 45}, {ox + cw} {oy - 18}" fill="none" stroke="{p["warm"]}" stroke-width="2"/>'
    )
    c.text(ox + cw + 6, oy - 40, "gold", size=12.5, fam=MONO, color="warm", weight=600)
    c.text(ox + cw + 6, oy - 26, "held-out", size=10.5, color="muted")
    c.text(ox + cw + 6, oy - 14, "scorer", size=10.5, color="muted")
    c.parts.append(
        f'<line x1="{ox + 330}" y1="{oy - 142}" x2="{ox + 330}" y2="{oy - 40}" stroke="{p["muted"]}" stroke-width="1" stroke-dasharray="3 3"/>'
    )
    c.text(ox + 338, oy - 88, "the gap", size=11.5, color="body")
    c.parts.append(
        f'<line x1="{ox}" y1="{oy + 12}" x2="{ox + cw}" y2="{oy + 12}" stroke="{p["line"]}" stroke-width="1"/>'
    )
    for x, label in [(ox + 40, "before"), (ox + 200, "during"), (ox + 360, "after")]:
        c.text(x, oy + 26, label, size=11.5, color="muted", weight=600, anchor="middle")
    checks = [
        ("before", ["hack_scan(endorsed=)", "judge_probes", "trace_flag_report"]),
        ("during", ["HackMonitor(", "  holdout=, gold=)"]),
        ("after", ["delta_report(proxy=)", "hack_scan_diff"]),
    ]
    x = 530
    y = 40
    for phase, calls in checks:
        c.text(x, y, phase, size=11.5, color="muted", weight=600)
        for j, call in enumerate(calls):
            c.text(x, y + 18 + j * 16, call, size=12, fam=MONO, color="ink")
        y += 18 + len(calls) * 16 + 14
    c.text(530, 286, "shape after Gao et al. 2023", size=10.5, color="muted")
    return c.render()


def safety_channels(p):
    c = Canvas(
        320,
        p,
        "Three ways an instruction reaches a tool-using agent, three ways data leaves, one marker per exit",
    )
    c.text(16, 24, "WHERE THE INSTRUCTION COMES FROM", size=10.5, color="muted", weight=600)
    ins = [
        ("the ask", ["prompt_injection", "social_engineering"]),
        ("a tool result", ["indirect_injection", "planted in a record"]),
        ("public text", ["a review, a listing", "(marketplace)"]),
    ]
    for i, (t, s) in enumerate(ins):
        y = 34 + i * 72
        c.box(16, y, 200, 60, t, s)
        c.arrow(216, y + 30, 268, 142)
    c.box(272, 112, 176, 60, "agent", ["reads private data,", "acts on state, sends"], hl=True)
    c.text(504, 24, "WHERE DATA CAN LEAVE", size=10.5, color="muted", weight=600)
    outs = [
        ("the reply", "no_secret_leak"),
        ("an outbound send", "no_external_send"),
        ("a write", "no_unauthorized_write"),
    ]
    for i, (t, m) in enumerate(outs):
        y = 34 + i * 72
        c.arrow(448, 142, 500, y + 30)
        c.box(504, y, 200, 60, t, [m, "1.0 when it held"])
    c.text(
        16,
        272,
        "controls: benign asks in the same set, marker helpful_on_benign. A refusal that",
        size=11.5,
        color="body",
    )
    c.text(
        16,
        288,
        "passes them is a reward the policy can collect by refusing everything.",
        size=11.5,
        color="body",
    )
    c.text(
        16,
        308,
        "reward = 1 only when every applicable marker holds; the judge reads steps and final_text",
        size=11.5,
        color="muted",
    )
    return c.render()


def character_pipeline(p):
    c = Canvas(
        340,
        p,
        "Character training as a data pipeline: constitution, prompts, replies under the deployment prompt, a judge that alone sees the principle, pairs and SFT, then a paired delta",
    )
    row(
        c,
        32,
        [
            ("constitution", "one principle per trait", False),
            ("prompts", "where the trait matters", False),
            ("k replies", "deployment prompt only", True),
        ],
    )
    c.path("M 580 88 L 580 100 L 116 100 L 116 130", green=True)
    c.text(
        140,
        122,
        "the principle is not in the sampling prompt, or you measure prompting, not character",
        size=11,
        color="warm",
    )
    row(
        c,
        132,
        [
            ("judge", "alone sees the principle", True),
            ("markers", "trait, on_task, no_filler", False),
            ("pairs and SFT", ["build_preference_pairs", "length_match=True"], False),
        ],
    )
    c.text(
        16,
        214,
        "judge check first: grade the spec's GOOD/BAD replies; agreement >= 0.8, kappa >= 0.6",
        size=11,
        fam=MONO,
        color="body",
    )
    c.path("M 580 188 L 580 236 L 116 236 L 116 250", green=True)
    c.box(16, 252, 200, 56, 'train(method="dpo")', "or any DPO trainer", mono=True)
    c.arrow(216, 280, 248, 280)
    c.box(
        248,
        252,
        456,
        56,
        "delta_report",
        ['target="marker:trait"', "must_not_regress=[on_task, no_filler]"],
        hl=True,
        mono=True,
    )
    c.text(
        16,
        330,
        'holdout: the same prompts + "drop the act", plus plain tasks the persona must not distort',
        size=11.5,
        color="muted",
    )
    return c.render()


def distillation_paths(p):
    c = Canvas(
        318,
        p,
        "Three ways to a per-token signal: a reward (GRPO), a frozen teacher (OPD), the same model with a hint (OPSD)",
    )
    row(
        c,
        32,
        [
            ("student samples", "k replies per prompt, T=1.0", True),
            ("score every token", "who scores decides the method", False),
            ("update", "prime-rl, your GPUs", False),
        ],
        mono=True,
    )
    c.path("M 348 88 L 348 108", green=True)
    xs = row(
        c,
        118,
        [
            ('"grpo"', ["reward on the reply", "advantage = r - group mean"], False),
            ("wai.OPD(teacher)", ["frozen server, logprobs", "A_t = log p_T - log p_S"], True),
            (
                'wai.OPSD("answer")',
                ["same model + hint", "A_t = log p(y|x,hint) - log p(y|x)"],
                True,
            ),
        ],
        mono=True,
    )
    c.text(16, 206, "needs a verifier or a judge;", size=11.5, color="body")
    c.text(16, 222, "zero gradient when all k agree", size=11.5, color="muted")
    c.text(xs[1], 206, "needs a stronger model that", size=11.5, color="body")
    c.text(xs[1], 222, "shares the tokenizer", size=11.5, color="muted")
    c.text(xs[2], 206, "needs a hint the model can use;", size=11.5, color="body")
    c.text(xs[2], 222, "costs points on thinking models", size=11.5, color="muted")
    c.path("M 116 244 L 116 262", dashed=True)
    c.box(
        16,
        264,
        688,
        40,
        "wai.prime_rl_config(taskset, method, model=, out=)  ->  reads / ignores / uv run rl @ file",
        None,
        mono=True,
    )
    return c.render()


def learn_two_scores(p):
    c = Canvas(300, p, "One score for the whole reply, or one score for every word")
    tokens = ["t", "l", "i", "u", "b"]
    student = [0.5, 0.6, 0.7, 0.2, 0.9]
    teacher = [0.9, 0.8, 0.7, 0.8, 0.6]
    adv = ["+0.59", "+0.29", "+0.00", "+1.39", "-0.41"]
    c.text(16, 46, "a reward (GRPO)", size=12, color="muted", fam=MONO)
    for i, tok in enumerate(tokens):
        c.box(16 + i * 64, 56, 52, 40, tok, None, mono=True)
    c.arrow(16 + 5 * 64 - 12, 76, 372, 76)
    c.box(376, 56, 328, 40, "0.8 for the whole reply", None, hl=True, mono=True)
    c.text(376, 112, "one number, after the reply is finished; every word", size=11.5, color="body")
    c.text(376, 128, "shares it, the right ones and the wrong ones alike", size=11.5, color="body")
    c.text(16, 166, "a teacher (OPD, OPSD)", size=12, color="muted", fam=MONO)
    for i, tok in enumerate(tokens):
        x = 16 + i * 128
        good = not adv[i].startswith("-")
        c.text(x + 8, 184, adv[i], size=13, fam=MONO, color="green" if good else "warm", weight=700)
        c.box(x, 192, 116, 56, tok, [f"student {student[i]}", f"teacher {teacher[i]}"], mono=True)
    c.text(
        16,
        274,
        "one number per word: how much more likely the teacher was to write it.",
        size=11.5,
        color="body",
    )
    c.text(
        16,
        290,
        "log teacher - log student. Above zero: do more of that. Below: less.",
        size=11.5,
        color="muted",
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
    "distillation-paths": distillation_paths,
    "learn-two-scores": learn_two_scores,
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
