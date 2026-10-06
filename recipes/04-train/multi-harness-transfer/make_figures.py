"""Shareable PNG figures for Phase 0. Numbers are copied from results.json, sensitivity.json and
PREREGISTRATION.md amendment 5 (paired bootstrap over 250 tasks, 3 tries per task, 95% bands)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # validated pair (dataviz validate_palette.js, light)
SOURCE = "while.ai re-score of FineEnvs' LFM2.5-2.6B checkpoints · SmolDataEnvs test, 250 tasks x 3 tries"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def frame(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def header(fig, title, subtitle):
    fig.text(0.03, 0.965, title, fontsize=16, fontweight="bold", va="top")
    fig.text(0.03, 0.905, subtitle, fontsize=11, color=INK2, va="top")
    fig.text(0.03, 0.02, SOURCE, fontsize=8.5, color=MUTED)


def dot(ax, y, v, lo, hi, color, label=None):
    ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
    ax.plot(v, y, "o", color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    ax.text(hi + 0.6, y, f"{v:+.1f}", va="center", fontsize=10, color=INK)


# 1. Gain over the untrained model, in the agents each model trained in vs agents it never saw.
gains = [  # model, trained four (v, lo, hi), unseen six (v, lo, hi)
    ("RL, trained in 1 tool", (8.1, 6.1, 10.0), (5.2, 3.6, 6.8)),
    ("RL, trained in 4 tools", (13.8, 11.5, 16.0), (4.4, 3.1, 5.7)),
    ("SFT, trained in 1 tool", (3.4, 1.2, 5.6), (4.7, 2.4, 6.9)),
    ("SFT, trained in 4 tools", (17.1, 14.0, 20.4), (6.9, 4.4, 9.7)),
]
fig, ax = plt.subplots(figsize=(10, 5.6))
fig.subplots_adjust(left=0.22, right=0.95, top=0.76, bottom=0.17)
for i, (name, tr, un) in enumerate(gains):
    y = len(gains) - 1 - i
    dot(ax, y + 0.15, *tr, BLUE, "Tools it trained in" if i == 0 else None)
    dot(ax, y - 0.15, *un, ORANGE, "6 new tools it never saw" if i == 0 else None)
ax.set_yticks(range(len(gains)), [g[0] for g in reversed(gains)], fontsize=11, color=INK)
ax.axvline(0, color=MUTED, linewidth=1)
ax.set_xlim(-1, 23)
ax.set_xlabel("Points gained over the untrained model (line = 95% range)")
frame(ax)
fig.legend(loc="upper left", bbox_to_anchor=(0.215, 0.85), ncol=2, frameon=False, fontsize=10.5, handletextpad=0.3, columnspacing=1.6)
header(fig, "Training in more coding tools only helps in those tools",
       "In new coding tools, every trained model improves about 5 points, whether it trained in 1 tool or 4")
fig.savefig(OUT / "1-gain-trained-vs-unseen.png", dpi=200)
plt.close(fig)

# 2. Four-agent RL minus OpenCode-only RL, per agent.
rows = [  # label, v, lo, hi, group
    ("Codex", 20.1, 16.3, 24.2, "t"), ("Claude Code", 6.4, 3.1, 9.7, "t"),
    ("Mini-SWE-Agent", 5.6, 2.3, 9.1, "t"), ("OpenCode", -9.6, -13.3, -5.9, "t"),
    ("Qwen Code", 6.0, 2.8, 9.2, "u"), ("OpenHands SDK", 3.3, 0.0, 6.5, "u"),
    ("Gemini CLI", -0.1, -0.4, 0.0, "u"), ("Terminus 2", -0.9, -3.8, 2.0, "u"),
    ("Vibe", -3.7, -7.2, -0.5, "u"), ("Pi", -9.3, -12.9, -5.5, "u"),
    ("All 6 new tools", -0.8, -2.0, 0.5, "p"),
]
fig, ax = plt.subplots(figsize=(10, 7.2))
fig.subplots_adjust(left=0.22, right=0.95, top=0.83, bottom=0.12)
ys = []
y = 0
prev = None
tops = {}
for label, v, lo, hi, g in reversed(rows):
    if prev is not None and g != prev:
        y += 1.0 if prev != "p" else 0.6
    color = {"t": BLUE, "u": ORANGE, "p": INK}[g]
    dot(ax, y, v, lo, hi, color)
    ys.append((y, label, g))
    tops[g] = y
    prev = g
    y += 1
ax.set_yticks([p[0] for p in ys], [p[1] for p in ys], fontsize=11)
for tick, (_, _, g) in zip(ax.get_yticklabels(), ys):
    tick.set_color(INK)
    if g == "p":
        tick.set_fontweight("bold")
ax.axvline(0, color=MUTED, linewidth=1)
ax.set_xlim(-16, 27)
ax.set_xlabel("Score difference in points (line = 95% range)")
ax.text(-15.5, tops["t"] + 0.75, "Tools both models trained in", color=BLUE, fontsize=10.5, fontweight="bold")
ax.text(-15.5, tops["u"] + 0.75, "New tools neither model saw", color=ORANGE, fontsize=10.5, fontweight="bold")
ax.set_ylim(-0.7, tops["t"] + 1.3)
ax.text(26.5, ys[0][0], "no real difference:\nrange crosses zero", ha="right", va="center", fontsize=9, color=INK2)
frame(ax)
header(fig, "In new coding tools, training in 4 tools is no better than training in 1",
       "Right of zero: the 4-tool model did better. Left: the 1-tool (OpenCode) model did better")
fig.savefig(OUT / "2-four-agents-minus-opencode-per-agent.png", dpi=200)
plt.close(fig)

# 3. The agent moves the score more than the model.
models = ["Base", "RL, 1 tool", "RL, 4 tools", "SFT, 1 tool", "SFT, 4 tools"]
grid = {  # base, oc-rl, mh-rl, oc-sft, mh-sft (211 tasks shared by every cell)
    "Terminus 2": (60.3, 63.7, 62.8, 66.0, 60.7), "Mini-SWE-Agent": (52.9, 54.0, 58.9, 46.6, 41.5),
    "Pi": (42.4, 56.3, 46.1, 55.6, 51.2), "OpenHands SDK": (44.5, 50.6, 53.9, 45.5, 45.0),
    "Vibe": (41.4, 51.3, 46.3, 44.1, 43.1), "OpenCode": (10.5, 43.5, 32.9, 43.2, 43.7),
    "Codex": (15.8, 14.5, 36.8, 10.3, 44.6), "Claude Code": (13.6, 17.3, 24.6, 8.0, 41.6),
    "Qwen Code": (8.7, 10.7, 17.9, 16.7, 28.8), "Gemini CLI": (0.2, 0.2, 0.0, 7.4, 20.9),
}
seq = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
names = list(grid)
fig, ax = plt.subplots(figsize=(10, 7.4))
fig.subplots_adjust(left=0.2, right=0.97, top=0.8, bottom=0.1)
vals = [grid[n] for n in names]
ax.imshow(vals, cmap=seq, vmin=0, vmax=70, aspect="auto")
for i, row in enumerate(vals):
    for j, v in enumerate(row):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10.5, color="#ffffff" if v > 40 else INK)
ax.set_xticks(range(len(models)), models, fontsize=10.5, color=INK)
ax.set_yticks(range(len(names)), names, fontsize=11, color=INK)
ax.xaxis.tick_top()
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_xticks([x - 0.5 for x in range(1, len(models))], minor=True)
ax.set_yticks([y - 0.5 for y in range(1, len(names))], minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
header(fig, "Which coding tool you use matters more than which model",
       "Percent of tasks solved. The tool explains 81% of the differences, the model only 5%")
fig.savefig(OUT / "3-agent-vs-model-heatmap.png", dpi=200)
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
