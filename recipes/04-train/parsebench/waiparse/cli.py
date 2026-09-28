"""`python -m waiparse.cli ...` = the parse-bench CLI with our providers registered."""

import sys

from parse_bench.cli import main

import waiparse  # noqa: F401  (registers providers + pipelines on import)

if __name__ == "__main__":
    sys.exit(main())
