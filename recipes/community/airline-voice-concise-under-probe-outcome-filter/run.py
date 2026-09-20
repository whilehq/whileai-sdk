"""The whole recipe, from one entry point.

    python run.py --selftest     # the reward, the filter and the maths, offline
    python run.py --prep         # build train/holdout, run the contamination checks
    python run.py --analyse      # paired numbers with intervals from out/

The GPU halves are two `modal run` commands; `--plan` prints them in order so
there is one place that says what this recipe does and in what sequence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PLAN = """\
1. python run.py --prep
2. modal run train_modal.py --arm baseline --steps 40    # flat by the shaped score
3. modal run train_modal.py --arm method   --steps 40    # flat by the binary outcome
4. modal run eval_modal.py                               # base x3 + both arms
5. modal volume get voice-filter-runs eval out --force
6. python run.py --analyse
7. modal deploy serve_modal.py && python fresh_traffic.py --url <url>/v1 --model <winner>
8. modal app stop voice-concise-filter-serve
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="offline: no GPU, no key")
    ap.add_argument("--prep", action="store_true", help="build the prompt sets")
    ap.add_argument("--analyse", action="store_true", help="paired numbers from out/")
    ap.add_argument("--plan", action="store_true", help="print the commands in order")
    args = ap.parse_args()

    if args.plan or not any((args.selftest, args.prep, args.analyse)):
        print(PLAN)
        return

    if args.selftest:
        sys.path.insert(0, str(HERE))
        import reward

        reward.selftest()
        subprocess.run([sys.executable, str(HERE / "analyse.py"), "--selftest"], check=True)
    if args.prep:
        subprocess.run([sys.executable, str(HERE / "prep.py")], check=True)
    if args.analyse:
        subprocess.run([sys.executable, str(HERE / "analyse.py")], check=True)


if __name__ == "__main__":
    main()
