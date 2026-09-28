"""Headline ParseBench scores for runs on your docparse-runs volume (computed on Modal).

    python scores.py wai_baseline_thinking:dev wai_agent_v4:dev wai_agent_v4:test

Each argument is <pipeline>:<split>. The overall score is the mean of the five
dimensions (tables, charts, text content, text formatting, layout), the same
average the ParseBench leaderboard reports. Uses your current Modal profile.
"""

from __future__ import annotations

import argparse
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+", help="<pipeline>:<split>, e.g. wai_agent_v4:test")
    args = ap.parse_args()
    env = {**os.environ, "PYTHONUTF8": "1"}
    out = subprocess.run(
        ["modal", "run", "serve/bench.py::scores", "--runs-", ",".join(args.runs)],
        env=env,
        capture_output=True,
        text=True,
        cwd=HERE,
    )
    keep = ("run ", *[a.split(":")[0] for a in args.runs])
    print("\n".join(line for line in out.stdout.splitlines() if line.startswith(keep)))
    if out.returncode:
        print(out.stderr[-2000:])
    return out.returncode


if __name__ == "__main__":
    raise SystemExit(main())
