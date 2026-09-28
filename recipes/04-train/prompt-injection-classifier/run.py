"""One command through the recipe: data, frozen test, baseline, train, export, results.

    python run.py --dry-run          # offline: rebuild the synthetic slices, check the frozen hash, print results.json
    python run.py                    # everything: needs the cloned public sets, a Modal token, about $1

Steps, in the order they ran:

1. ``data.py`` builds the frozen test (six slices) and the training rows from
   the public sets and the planted carriers; ``test.sha256`` is written first.
2. ``score.py`` scores the accessible baseline (ProtectAI v2) on the test.
3. ``train_modal.py`` fine-tunes MiniLM-L6 for three seeds on one L40S.
4. ``score.py`` scores each seed; ``export_onnx.py`` exports seed 1 to int8
   ONNX, times it single-threaded, and scores the test with the int8 graph.
5. ``results.json`` collects every number with its interval.
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


def check_frozen() -> str:
    rows = read_jsonl(HERE / "test.jsonl")
    digest = sha256_rows(rows)
    pinned = (HERE / "test.sha256").read_text().strip()
    if digest != pinned:
        raise SystemExit(
            f"test.jsonl does not match test.sha256 ({digest[:12]} vs {pinned[:12]}); the test is frozen, rebuild nothing"
        )
    file_digest = hashlib.sha256((HERE / "test.jsonl").read_bytes()).hexdigest()
    print(
        f"frozen test: {len(rows)} rows, rows sha256 {digest[:16]}..., file sha256 {file_digest[:16]}..."
    )
    print(f"held-out families {HELDOUT_FAMILIES}, held-out carriers {HELDOUT_CARRIERS}")
    return digest


def print_results() -> None:
    p = HERE / "results.json"
    if not p.exists():
        print("no results.json yet")
        return
    r = json.loads(p.read_text())
    print(f"\n{r['recipe']}: {r['verdict']}")
    for name, arm in r["arms"].items():
        h = arm["indirect_heldout"]
        print(
            f"  {name:24s} indirect held-out AUROC {h['auroc']:.3f} [{h['auroc_ci95'][0]:.3f}, {h['auroc_ci95'][1]:.3f}]"
            f"  recall@1%FPR {h['recall_at_1pct_fpr']:.3f}  NotInject FPR {arm['notinject']['fpr']:.3f}"
        )
    lat = r.get("latency", {}).get("int8_single_thread", {})
    if lat:
        print(
            "  int8 ONNX, one thread:",
            ", ".join(f"{k} tok p50 {v['p50_ms']} ms p99 {v['p99_ms']} ms" for k, v in lat.items()),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="no downloads, no key, no GPU: check the frozen test and print results",
    )
    ap.add_argument(
        "--ext", default=str(HERE / "ext"), help="cloned InjecAgent, agentdojo, BIPIA, InjecGuard"
    )
    ap.add_argument("--data", default=str(HERE / "out"), help="deepset.jsonl and gandalf.jsonl")
    ap.add_argument("--sim", default=None, help="wai.simulate rows to harvest tool results from")
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()
    check_frozen()
    if a.dry_run:
        print_results()
        return
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    cmd = [PY, str(HERE / "data.py"), "--ext", a.ext, "--data", a.data, "--out", str(out)]
    if a.sim:
        cmd += ["--sim", a.sim]
    subprocess.run(cmd, check=True)
    rebuilt = sha256_rows(read_jsonl(out / "test.jsonl"))
    if rebuilt != (HERE / "test.sha256").read_text().strip():
        raise SystemExit(
            "the rebuilt test differs from the frozen one; the public sets or the generator changed"
        )
    subprocess.run(
        [
            PY,
            str(HERE / "score.py"),
            "--model",
            "protectai/deberta-v3-base-prompt-injection-v2",
            "--threshold",
            "0.5",
            "--out",
            str(out / "scores_protectai.json"),
        ],
        check=True,
    )
    subprocess.run(
        [PY, "-m", "modal", "run", str(HERE / "train_modal.py"), "--seeds", a.seeds], check=True
    )
    for s in a.seeds.split(","):
        subprocess.run(
            [
                PY,
                str(HERE / "score.py"),
                "--model",
                str(out / f"minilm-l6-h384-uncased-seed{s}"),
                "--out",
                str(out / f"scores_minilm_seed{s}.json"),
            ],
            check=True,
        )
    subprocess.run(
        [
            PY,
            str(HERE / "export_onnx.py"),
            "--model",
            str(out / "minilm-l6-h384-uncased-seed1"),
            "--out",
            str(out / "onnx-seed1"),
            "--test",
            str(HERE / "test.jsonl"),
        ],
        check=True,
    )
    subprocess.run([PY, str(HERE / "collect.py")], check=True)
    print_results()


if __name__ == "__main__":
    main()
