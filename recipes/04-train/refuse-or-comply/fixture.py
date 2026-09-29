"""Write fixtures.jsonl: the same few prompts from every draw, trimmed to the
fields the analysis reads, so the offline path scores real rows.

    python fixture.py --per-half 6      # 12 prompts x 15 draws = 180 rows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEEP = (
    "prompt",
    "scenario_id",
    "arm",
    "draw",
    "reward",
    "judge_meta",
    "final_text",
    "finish_reason",
)


def trim(row: dict) -> dict:
    out = {k: row.get(k) for k in KEEP}
    out["steps"] = [
        {"tool": s.get("tool"), "arguments": s.get("arguments"), "result": s.get("result")}
        for s in (row.get("steps") or [])
        if isinstance(s, dict) and s.get("tool")
    ]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(HERE / "out"))
    p.add_argument("--per-half", type=int, default=6)
    args = p.parse_args(argv)
    paths = sorted(Path(args.out).glob("eval_*.jsonl"))
    if not paths:
        raise SystemExit("no eval_*.jsonl under " + args.out)
    first = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    attack = sorted({r["scenario_id"] for r in first if r["judge_meta"].get("had_planted_text")})
    clean = sorted({r["scenario_id"] for r in first if not r["judge_meta"].get("had_planted_text")})
    keep = set(attack[: args.per_half] + clean[: args.per_half])
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("scenario_id") in keep:
                rows.append(trim(r))
    (HERE / "fixtures.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(
        f"wrote {len(rows)} rows from {len(paths)} draws over {len(keep)} prompts -> fixtures.jsonl"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
