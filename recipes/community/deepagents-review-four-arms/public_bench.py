"""The public benchmark: patch review on SWE-bench Verified, built from the official leaderboard.

Nothing here is ours. The issues and base commits are SWE-bench Verified
(princeton-nlp/SWE-bench_Verified); the candidate patches are six public
leaderboard submissions; whether each patch resolved its issue is the
official evaluation's `report.json`, published with the submission. For every
issue with at least one resolved and one unresolved patch among those six,
one of each is kept (seeded), so the set is 50/50 approve/reject. None of the
twelve SWE-bench repositories appears in the training traces.

    python public_bench.py            # writes public_bench.jsonl (250 reviews, 125 issues)

Then any arm plays it with collect.py --split public.
"""

from __future__ import annotations

import json
import pathlib
import random
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "public_bench.jsonl"
CACHE = HERE / ".cache" / "swebench"
VERIFIED = "princeton-nlp/SWE-bench_Verified"
VERIFIED_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
BUCKET = "s3://swe-bench-submissions/verified"
# Six public submissions, resolve rates 23% to 53%: enough of both outcomes per issue.
SUBMISSIONS = (
    "20240620_sweagent_claude3.5sonnet",
    "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
    "20240728_sweagent_gpt4o",
    "20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022",
    "20241022_tools_claude-3-5-sonnet-updated",
    "20240721_amazon-q-developer-agent-20240719-dev",
)
N_ISSUES = 125
MAX_PATCH_CHARS = 12_000  # the same cap as the training tasks
SEED = 0


def fetch(sub: str) -> None:
    """patch.diff and report.json for every instance of one submission (public, no credentials)."""
    dest = CACHE / sub
    if dest.exists():
        return
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            "--no-sign-request",
            "--only-show-errors",
            f"{BUCKET}/{sub}/logs/",
            str(dest),
            "--exclude",
            "*",
            "--include",
            "*/patch.diff",
            "--include",
            "*/report.json",
        ],
        check=True,
    )


def attempts(sub: str) -> dict[str, tuple[str, bool]]:
    out = {}
    for d in (CACHE / sub).iterdir():
        patch, report = d / "patch.diff", d / "report.json"
        if not (patch.exists() and report.exists()):
            continue
        text = patch.read_text(encoding="utf-8", errors="replace").strip()
        rep = json.loads(report.read_text(encoding="utf-8")).get(d.name, {})
        if text and len(text) < MAX_PATCH_CHARS and "resolved" in rep:
            out[d.name] = (text, bool(rep["resolved"]))
    return out


def build() -> list[dict]:
    from datasets import load_dataset

    for s in SUBMISSIONS:
        fetch(s)
    per = {s: attempts(s) for s in SUBMISSIONS}
    verified = {
        r["instance_id"]: r
        for r in load_dataset(VERIFIED, split="test", revision=VERIFIED_REVISION)
    }
    rng = random.Random(SEED)
    pools = []
    for iid in sorted(verified):
        good = sorted({per[s][iid][0] for s in SUBMISSIONS if iid in per[s] and per[s][iid][1]})
        bad = sorted({per[s][iid][0] for s in SUBMISSIONS if iid in per[s] and not per[s][iid][1]})
        if good and bad:
            pools.append((iid, good, bad))
    chosen = sorted(rng.sample(pools, min(N_ISSUES, len(pools))))
    tasks = []
    for iid, good, bad in chosen:
        v = verified[iid]
        for verdict, pool in (("approve", good), ("reject", bad)):
            tasks.append(
                {
                    "id": f"{iid}:{verdict}",
                    "instance_id": iid,
                    "repo": v["repo"],
                    "base_commit": v["base_commit"],
                    "issue": v["problem_statement"],
                    "patch": rng.choice(pool),
                    "label": verdict,
                    "split": "public",
                }
            )
    OUT.write_text("\n".join(json.dumps(t) for t in tasks) + "\n", encoding="utf-8")
    print(
        f"{len(pools)} Verified issues have both outcomes; kept {len(chosen)} -> {len(tasks)} reviews"
    )
    return tasks


if __name__ == "__main__":
    build()
