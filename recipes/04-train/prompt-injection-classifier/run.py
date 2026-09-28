"""One command through the recipe: data, frozen tests, baseline, the rounds, export, results.

    python run.py --dry-run          # offline: check the three frozen hashes, print results.json
    python run.py --round 3          # the twins round on template carriers (needs the cloned sets, Modal)
    python run.py --round 7          # the model-written-carrier round (needs the simulate rows too)

What the live path does, in the order it ran:

1. ``data.py`` builds the frozen tests (``test.jsonl``, ``test_hard.jsonl``,
   ``test_paste.jsonl``; their sha256 files are written first and checked on
   every later run) and the round-3 and round-5 training rows.
2. ``score.py`` scores the accessible baseline (ProtectAI v2) on the tests.
3. ``sim_llm.py`` (in the scratch directory; see README) makes the six-domain
   traffic with a model as writer, user and world; ``data_llm.py`` turns it
   into round-7 rows.
4. ``train_modal.py`` fine-tunes MiniLM-L6 for three seeds on one L40S.
5. ``round_score.py`` scores each seed on the three tests and the probe pairs,
   ``export_onnx.py`` exports seed 1 to int8 ONNX and times it, ``collect.py``
   writes ``results.json``, ``post_platform.py`` posts the climb.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import HELDOUT_CARRIERS, HELDOUT_FAMILIES, read_jsonl, sha256_rows

PY = sys.executable
TESTS = ("test", "test_hard", "test_paste", "test_llm", "test_external")


def check_frozen() -> None:
    for t in TESTS:
        rows = read_jsonl(HERE / f"{t}.jsonl")
        digest = sha256_rows(rows)
        pinned = (HERE / f"{t}.sha256").read_text().strip()
        if digest != pinned:
            raise SystemExit(
                f"{t}.jsonl does not match {t}.sha256 ({digest[:12]} vs {pinned[:12]}); the test is frozen"
            )
        file_digest = hashlib.sha256((HERE / f"{t}.jsonl").read_bytes()).hexdigest()
        print(
            f"frozen {t}: {len(rows)} rows, rows sha256 {digest[:16]}..., file sha256 {file_digest[:16]}..."
        )
    print(f"held-out families {HELDOUT_FAMILIES}, held-out carriers {HELDOUT_CARRIERS}")


def print_results() -> None:
    p = HERE / "results.json"
    if not p.exists():
        print("no results.json yet")
        return
    r = json.loads(p.read_text())
    print(f"\n{r['recipe']}: {r['verdict']}")
    for name, arm in r["arms"].items():
        h = arm.get("headline_points") or {}
        if not h:
            continue
        cells = "  ".join(f"{k} {v['points']:.0f}±{v['ci95_half']:.0f}" for k, v in h.items())
        print(f"  {name:28s} {cells}")
    for name, lat in (r.get("latency") or {}).items():
        one = lat.get("latency_single_thread", {})
        print(
            f"  {name}: int8 {lat.get('int8_mb')} MB; one thread "
            + ", ".join(f"{k} tok p50 {v['p50_ms']} ms" for k, v in one.items())
        )


def run(*cmd: str) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(list(cmd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="no downloads, no key, no GPU: check the frozen tests and print results",
    )
    ap.add_argument(
        "--round",
        type=int,
        default=7,
        help="3: twins on template carriers; 5: plus channels; 7: model-written carriers",
    )
    ap.add_argument(
        "--ext", default=str(HERE / "ext"), help="cloned InjecAgent, agentdojo, BIPIA, InjecGuard"
    )
    ap.add_argument(
        "--data",
        default=str(HERE / "out"),
        help="deepset.jsonl, gandalf.jsonl, oasst1_prompts.jsonl",
    )
    ap.add_argument(
        "--sim", action="append", default=[], help="wai.simulate rows (jsonl); several for round 7"
    )
    ap.add_argument(
        "--inserts", default=None, help="model-written paraphrases and benign inserts (round 7)"
    )
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()
    check_frozen()
    if a.dry_run:
        print_results()
        return
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    cmd = [
        PY,
        str(HERE / "data.py"),
        "--ext",
        a.ext,
        "--data",
        a.data,
        "--out",
        str(out),
        "--no-spml",
    ]
    if a.sim:
        cmd += ["--sim", a.sim[0]]
        for extra in a.sim[1:]:
            cmd += ["--sim-extra", extra]
    run(*cmd)
    for t in TESTS:
        if (
            sha256_rows(read_jsonl(out / f"{t}.jsonl"))
            != (HERE / f"{t}.sha256").read_text().strip()
        ):
            raise SystemExit(
                f"the rebuilt {t} differs from the frozen one; the public sets or the generator changed"
            )
    run(
        PY,
        str(HERE / "score.py"),
        "--model",
        "protectai/deberta-v3-base-prompt-injection-v2",
        "--threshold",
        "0.5",
        "--out",
        str(out / "scores_protectai.json"),
    )
    for t in ("hard", "paste"):
        run(
            PY,
            str(HERE / "score.py"),
            "--model",
            "protectai/deberta-v3-base-prompt-injection-v2",
            "--threshold",
            "0.5",
            "--test",
            str(HERE / f"test_{t}.jsonl"),
            "--probe",
            "none",
            "--out",
            str(out / f"scores_{t}_protectai.json"),
        )
    if a.round >= 7:
        if not a.inserts or len(a.sim) < 2:
            raise SystemExit(
                "round 7 needs --inserts and the sim_llm_*.jsonl files as --sim (see README)"
            )
        run(
            PY,
            str(HERE / "data_llm.py"),
            *[arg for s in a.sim[1:] for arg in ("--sim", s)],
            "--inserts",
            a.inserts,
            "--ext",
            a.ext,
            "--data",
            a.data,
            "--out",
            str(out),
            "--tag",
            "v7",
        )
        train_file, val_file, tag, version = (
            out / "train_v7.jsonl",
            out / "val_v7.jsonl",
            "v7-llm",
            "v7-llm-carriers",
        )
    else:
        train_file, val_file, tag, version = (
            out / "train.jsonl",
            out / "val.jsonl",
            f"v{a.round}",
            f"v{a.round}",
        )
    run(
        PY,
        "-m",
        "modal",
        "run",
        str(HERE / "train_modal.py"),
        "--train-file",
        str(train_file),
        "--seeds",
        a.seeds,
        "--tag",
        tag,
    )
    note = out / f"note_{tag}.txt"
    if not note.exists():
        note.write_text(
            "Changed:\nMoved:\nWhy:\nLearned:\nReproduce: python run.py --round "
            + str(a.round)
            + "\n"
        )
    n_train = sum(1 for _ in open(train_file))
    run(
        PY,
        str(HERE / "round_score.py"),
        "--tag",
        tag,
        "--version",
        version,
        "--note-file",
        str(note),
        "--n-train",
        str(n_train),
        "--val",
        str(val_file),
        "--seeds",
        a.seeds,
    )
    run(
        PY,
        str(HERE / "export_onnx.py"),
        "--model",
        str(out / f"{tag}-seed1"),
        "--out",
        str(out / f"onnx-{tag}"),
        "--test",
        str(HERE / "test.jsonl"),
    )
    run(PY, str(HERE / "shortcut_probe.py"))
    run(PY, str(HERE / "collect.py"))
    print_results()


if __name__ == "__main__":
    main()
