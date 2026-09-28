"""Build the review tasks: one issue, one candidate patch, a known verdict.

Source is public. `nebius/SWE-agent-trajectories` holds 80k SWE-agent attempts at
real GitHub issues, each with the patch it produced and whether the hidden tests
passed. `nebius/SWE-bench-extra` holds the issue text and base commit. For every
issue that has at least one passing and one failing attempt we keep one of each,
so every issue contributes an "approve" and a "reject" and the labels are 50/50
by construction.

Splits are by repository, not by row: a holdout repo is never seen in training
traces, so the holdout measures review skill, not memory of a codebase.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / ".cache"
TASKS = HERE / "tasks.jsonl"
DECONTAM = HERE / "decontam.json"
UNAVAILABLE = HERE / "unavailable.json"  # repos or commits since deleted from GitHub

# Dataset revisions the published numbers were built from.
TRAJECTORIES = "nebius/SWE-agent-trajectories@68195a1450865274106246d0d0296a1d6807b88e"
ISSUES = "nebius/SWE-bench-extra@11dcbfb30e19552df2a2f8030bd764adc95c92a5"

MAX_PATCH_CHARS = 12_000  # longer diffs blow the context of a 27B reviewer
SEED = 0


def split_of(repo: str) -> str:
    """Stable repo-level split: 20% holdout, 10% dev (harness search), rest train."""
    h = int(hashlib.sha256(repo.encode()).hexdigest(), 16) % 100
    if h < 20:
        return "holdout"
    if h < 30:
        return "dev"
    return "train"


def _read(path: str, columns: list[str]):
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    tables = []
    for f in sorted(fs.glob(f"datasets/{path}/**/*.parquet")):
        with fs.open(f) as fh:
            tables.append(pq.read_table(fh, columns=columns).to_pandas())
    return pd.concat(tables, ignore_index=True)


def build() -> list[dict]:
    CACHE.mkdir(exist_ok=True)
    meta = CACHE / "traj_meta.parquet"
    if meta.exists():
        import pandas as pd

        attempts = pd.read_parquet(meta)
    else:
        attempts = _read(
            TRAJECTORIES,
            ["instance_id", "model_name", "target", "exit_status", "generated_patch"],
        )
        attempts.to_parquet(meta)

    attempts["passed"] = attempts.target.astype(str) == "True"
    patch = attempts.generated_patch.fillna("").str.strip()
    attempts = attempts[(patch.str.len() > 0) & (patch.str.len() < MAX_PATCH_CHARS)]

    issues = _read(
        ISSUES,
        ["instance_id", "repo", "base_commit", "problem_statement"],
    ).set_index("instance_id")

    rng = random.Random(SEED)
    tasks = []
    for iid, group in sorted(attempts.groupby("instance_id"), key=lambda kv: kv[0]):
        good = group[group.passed].generated_patch.tolist()
        bad = group[~group.passed].generated_patch.tolist()
        if not good or not bad or iid not in issues.index:
            continue
        issue = issues.loc[iid]
        for verdict, pool in (("approve", good), ("reject", bad)):
            tasks.append(
                {
                    "id": f"{iid}:{verdict}",
                    "instance_id": iid,
                    "repo": issue.repo,
                    "base_commit": issue.base_commit,
                    "issue": issue.problem_statement,
                    "patch": rng.choice(sorted(pool)),
                    "label": verdict,
                    "split": split_of(issue.repo),
                }
            )
    TASKS.write_text("\n".join(json.dumps(t) for t in tasks) + "\n", encoding="utf-8")
    return tasks


def decontaminate(tasks: list[dict]) -> dict:
    """wai.decontaminate on top of the repo split: drop train and dev reviews whose
    issue-plus-patch text near-copies a holdout one (8-grams over 80% of tokens).
    Disjoint repos should leave nothing to drop; this is the check that says so."""
    import whileai as wai

    def as_rows(ts):
        # task_id = the GitHub issue, so the same_task rule runs too.
        return [
            {**t, "task_id": t["instance_id"], "prompt": t["issue"] + "\n" + t["patch"]} for t in ts
        ]

    holdout = as_rows([t for t in tasks if t["split"] == "holdout"])
    rest = as_rows([t for t in tasks if t["split"] != "holdout"])
    clean, rep = wai.decontaminate(rest, against=holdout)
    keep = {r["id"] for r in clean}
    dropped = sorted(t["id"] for t in rest if t["id"] not in keep)
    out = {
        "n_checked": len(rest),
        "n_dropped": len(dropped),
        "dropped": dropped,
        "rules_skipped": rep.get("rules_skipped"),
        "notes": rep.get("notes"),
    }
    DECONTAM.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


PUBLIC = HERE / "public_bench.jsonl"  # SWE-bench Verified reviews, see public_bench.py


def load(split: str | None = None, limit: int | None = None) -> list[dict]:
    if split == "public":
        rows = [json.loads(line) for line in PUBLIC.read_text(encoding="utf-8").splitlines()]
        return rows[:limit] if limit else rows
    rows = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines()]
    if DECONTAM.exists():
        dropped = set(json.loads(DECONTAM.read_text(encoding="utf-8"))["dropped"])
        rows = [r for r in rows if r["id"] not in dropped]
    if UNAVAILABLE.exists():
        gone = set(json.loads(UNAVAILABLE.read_text(encoding="utf-8"))["tasks"])
        rows = [r for r in rows if r["id"] not in gone]
    if split:
        rows = [r for r in rows if r["split"] == split]
    return rows[:limit] if limit else rows


if __name__ == "__main__":
    import collections

    tasks = build()
    print("decontaminate:", {k: v for k, v in decontaminate(tasks).items() if k != "dropped"})
    by = collections.Counter((t["split"], t["label"]) for t in tasks)
    repos = collections.Counter(
        t["split"] for t in {(t["repo"], t["split"]): t for t in tasks}.values()
    )
    print(f"{len(tasks)} tasks", dict(by), "repos per split:", dict(repos))
