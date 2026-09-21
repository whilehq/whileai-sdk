"""Hugging Face round trip: import a public split and read its numbers,
optionally push one of your own sets (or a finished run's adapter) and
print the tagged commit.

    python roundtrip.py
    python roundtrip.py --repo-in tatsu-lab/alpaca --split train --keep
    python roundtrip.py --push ds_0123 --repo my-airline-set --private
    python roundtrip.py --push-run run_0123 --repo my-airline-lora

Needs a platform key (``wai login`` or WHILEAI_API_KEY). The push
halves also need a Hugging Face account connected on the platform.
"""

from __future__ import annotations

import argparse
import sys

import whileai.simulations as wai
from whileai.config import provenance
from whileai.simulations.ingest.platform import PlatformError


def fmt_pct(v: float | None) -> str:
    return "–" if v is None else f"{round(v * 100)}%"


def fmt_num(v: float | None) -> str:
    return "–" if v is None else f"{v:.2f}"


def import_half(repo: str, split: str, keep: bool) -> None:
    row = wai.import_hf(
        repo, split=split, purpose="eval", name=f"{repo.split('/')[-1]}:{split} (example)"
    )
    print(f"imported {repo}:{split} -> {row['datasetId']} ({row.get('rows') or '?'} rows)")
    p = wai.profile(row["datasetId"])
    print(
        f"  rows {p['rows']} · prompts {p['tasks']} · graded {p['graded']} · "
        f"pass {fmt_pct(p.get('pass_rate'))} · support {fmt_num(p.get('support'))}"
    )
    if not p["graded"]:
        print("  (no reward field: profile it after grading, or use it as an eval set)")
    if keep:
        print(f"  kept as {row['datasetId']}")
    else:
        wai.delete_dataset(row["datasetId"])
        print("  deleted")


def connected_username() -> str:
    me = wai.hf_status()
    if not me["connected"]:
        sys.exit(
            "Connect a Hugging Face account first: any dataset page under Platform → Datasets."
        )
    return me["username"]


def push_half(dataset_id: str, repo: str | None, private: bool) -> None:
    print(f"pushing {dataset_id} as {connected_username()} ...")
    hf = wai.hf_publish(dataset_id, repo=repo, private=private, wait=True)
    print(f"  {hf['url']}")
    print(f"  split {hf['split']} · commit {hf['commit'][:7]} · tag {hf['tag']}")
    if hf.get("previous"):
        prev = hf["previous"]
        print(
            f"  replaced {prev['datasetId']} (rows {prev['rows']}, pass {fmt_pct(prev.get('pass_rate'))})"
        )
    print(f'  load_dataset("{hf["repo"]}", split="{hf["split"]}", revision="{hf["tag"]}")')


def push_run_half(run_id: str, repo: str | None) -> None:
    """A finished training run's LoRA adapter becomes a model repo. Always
    private from here: it is a checkpoint, not a release."""
    print(f"pushing adapter of {run_id} as {connected_username()} ...")
    hf = wai.hf_publish_run(run_id, repo=repo, private=True, wait=True)
    print(f"  {hf['url']}")
    print(f"  commit {hf['commit'][:7]} · tag {hf['tag']}")
    print(f'  PeftModel.from_pretrained(base, "{hf["repo"]}", revision="{hf["tag"]}")')


def main(argv: list[str] | None = None) -> None:
    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--repo-in",
        default="cornell-movie-review-data/rotten_tomatoes",
        help="Hub dataset to import",
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--keep", action="store_true", help="keep the imported set on the account")
    ap.add_argument(
        "--push", metavar="DATASET_ID", help="also push one of your sets to Hugging Face"
    )
    ap.add_argument(
        "--push-run", metavar="RUN_ID", help="also push a finished run's adapter as a model repo"
    )
    ap.add_argument("--repo", help="repo name for the push (default: a slug of the set's name)")
    ap.add_argument(
        "--private",
        action="store_true",
        help="make the pushed dataset repo private (adapter repos always are)",
    )
    args = ap.parse_args(argv)

    # Every call here needs a platform credential. PlatformError already
    # says which of the three ways to supply one; without this it arrives
    # as a ten-frame traceback with that sentence at the bottom.
    try:
        import_half(args.repo_in, args.split, args.keep)
        if args.push:
            push_half(args.push, args.repo, args.private)
        if args.push_run:
            push_run_half(args.push_run, args.repo)
    except PlatformError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
