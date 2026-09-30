"""Print the README's climb table from results.json, and write it into README.md.

    python climb_table.py            # prints the table
    python climb_table.py --write    # replaces the block between the climb markers in README.md

Seed 1 per round, correctness at the round's threshold, points out of 100;
the Wilson half-widths are in results.json. Nothing here computes a number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLS = [
    ("agentdojo_docs", "AgentDojo docs (100)"),
    ("llmail_inject", "LLMail recall (600)"),
    ("multilingual_direct", "multilingual recall (974)"),
    ("llm_heldout_domain", "llm (422)"),
    ("hard", "hard (316)"),
    ("paste", "paste (300)"),
    ("deepset", "deepset (662)"),
    ("sim_tool", "sim_tool (91)"),
    ("notinject", "NotInject (339)"),
    ("indirect_heldout_family", "held-out fam. (250)"),
]
CHANGES = {
    "protectai-v2": "served baseline, 184M",
    "v1-templates": "planted vs clean template carriers",
    "v3-twins-dedupe": "matched twins, 8-gram dedupe, oasst1 turns",
    "v4-spml-direct": "+ SPML direct set (MIT)",
    "v5-channels": "SPML out; paste channel + six-domain traffic",
    "v6-twin-margin": "v5 rows + pairwise twin hinge",
    "v7-llm-carriers": "model-written carriers only, all six businesses",
    "v7-mined-weights": "v5 rows, wrong or uncertain rows weighted x3",
    "v7-random-weights-control": "v5 rows, the same mass of x3 weights at random",
    "v7b-llm-heldout-domain": "model-written carriers, legal held out",
    "v8-union": "v5 rows + v7b rows",
    "v9-seeded-world": "+ seeded runs, the world plants and labels",
    "v10-multilingual": "+ translations, multilingual MiniLM-L12 base (118 MB)",
}


def main() -> None:
    r = json.loads((HERE / "results.json").read_text())
    base = r["arms"]["protectai-v2"]["headline_points"]
    lines = [
        "| round | change | " + " | ".join(c for _, c in COLS) + " |",
        "|---|---|" + "---|" * len(COLS),
    ]
    for name, arm in r["arms"].items():
        h = arm.get("headline_points") or {}
        cells = []
        for key, _ in COLS:
            if key not in h:
                cells.append("n/a")
                continue
            v = h[key]["points"]
            lo, hi = v - h[key]["ci95_half"], v + h[key]["ci95_half"]
            b = base.get(key)
            mark = ""
            if b and name != "protectai-v2":
                if lo > b["points"] + b["ci95_half"]:
                    mark = "**"
                elif hi < b["points"] - b["ci95_half"]:
                    mark = "_"
            cells.append(f"{mark}{v:.0f}{mark}")
        lines.append(
            f"| {name} | {CHANGES.get(name, arm.get('note', '')[:40])} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")
    lines.append(
        "Bold clears the baseline's Wilson 95% interval on that slice; italics sit below it; plain is inside it. "
        f"Verdict: {r['verdict']}."
    )
    table = "\n".join(lines)
    print(table)
    if "--write" in sys.argv:
        p = HERE / "README.md"
        s = p.read_text()
        start, end = "<!--climb-table-->", "<!--/climb-table-->"
        if end in s:
            i, j = s.index(start), s.index(end) + len(end)
            s = s[:i] + start + "\n" + table + "\n" + end + s[j:]
        else:
            s = s.replace(start, start + "\n" + table + "\n" + end)
        p.write_text(s)


if __name__ == "__main__":
    main()
