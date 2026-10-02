"""Turn a smithtune pull into SFT rows with smithtune's own converter and split.

Runs inside the smithtune image (it imports smithtune), because `smithtune
prepare` refuses to run without a Baseten Loops or Fireworks account. What it
skips is only that provider preflight; the conversion (`prepare_sft_rows`,
reasoning omitted, the default) and the 80/10/10 split (`split_rows`, fixed
hash ranges) are smithtune's functions, so both arms train on exactly the rows
smithtune would have.

    docker run ... smithtune:0.1.0 python /recipe/export_sft.py council /work/sft-without
    docker run ... smithtune:0.1.0 python /recipe/export_sft.py while   /work/sft-with

A council directory keeps only trajectories its labels.jsonl marks keep=1.
The rerun (PREREGISTRATION.md) adds two options: `--also DIR` adds another
pull (Experiment B's pools span two LangSmith projects), and `--keep FILE`
keeps only the trace ids listed in FILE (equalize.py's equal-size cut).

    python /recipe/export_sft.py council-v2 /work/a-without --keep /work/keep/a-without.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smithtune.bindings import read_bindings
from smithtune.dataset import prepare_sft_rows, split_rows
from smithtune.providers.baseten import MODEL_SPECS
from smithtune.rendering import load_training_renderer, render_row_tokens

# The Baseten path's renderer for this model ("hf_assistant"): every assistant
# turn is one datum, its history at zero loss, itself at loss one.
MODEL = MODEL_SPECS["qwen3p8-27b"]


def load(pull: Path, only: set[str] | None = None) -> list[dict]:
    keep = None
    labels = pull / "labels.jsonl"
    if labels.exists():
        keep = {
            r["trajectory_id"]
            for r in map(json.loads, labels.read_text().splitlines())
            if r["keep"] == 1
        }
    examples = []
    for f in sorted((pull / "conversations").glob("*.json")):
        ex = json.loads(f.read_text())["example"]
        trace = ex["metadata"].get("source_trace_id")
        if (keep is None or ex["id"] in keep) and (only is None or trace in only):
            examples.append(ex)
    return examples


def tools_of(example: dict) -> list[dict]:
    """The tool schemas the agent saw on its first call (the same on every call)."""
    bindings = read_bindings(example.get("metadata"), example["inputs"]["messages"])
    return next(b["tools"] for b in bindings.values() if b.get("tools"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("pull", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--also", type=Path, action="append", default=[])
    p.add_argument("--keep", type=Path, help="JSON list of source trace ids to keep")
    args = p.parse_args()
    pull, out = args.pull, args.out
    only = set(json.loads(args.keep.read_text())) if args.keep else None
    examples = [ex for d in (pull, *args.also) for ex in load(d, only)]
    if only is not None and len(examples) != len(only):
        raise SystemExit(f"--keep lists {len(only)} traces but {len(examples)} were found")
    warnings: list[dict] = []
    rows = prepare_sft_rows(
        examples,
        reasoning_policy="omit",
        exclusion_warnings=warnings,
        workspace_id=examples[0]["metadata"]["source_workspace_id"],
    )
    train, val, test = split_rows(rows)
    out.mkdir(parents=True, exist_ok=True)
    tools = tools_of(examples[0])
    renderer = load_training_renderer(MODEL)
    counts = {}
    for name, part in (("train", train), ("validation", val), ("test", test)):
        n = 0
        with (out / f"{name}.jsonl").open("w") as f, (out / f"{name}.tokens.jsonl").open("w") as g:
            for r in part:
                f.write(
                    json.dumps(
                        {
                            "messages": r["messages"],
                            "tools": tools,
                            "example_id": r["_source"]["example_id"],
                        }
                    )
                    + "\n"
                )
                for d in render_row_tokens(r, MODEL, renderer=renderer):
                    g.write(
                        json.dumps(
                            {
                                "example_id": r["_source"]["example_id"],
                                "ids": list(d.token_ids),
                                "w": [int(x) for x in d.token_weights],
                            }
                        )
                        + "\n"
                    )
                    n += 1
        counts[f"{name}_datums"] = n
    summary = {
        **counts,
        "pull": [str(d) for d in (pull, *args.also)],
        "keep": str(args.keep) if args.keep else None,
        "examples": len(examples),
        "rows": len(rows),
        "excluded": len(warnings),
        "train": len(train),
        "validation": len(val),
        "test": len(test),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
