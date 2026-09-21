"""The reader's side of recipes/01-simulate/bring-your-own-agent.

The page's block writes `my_agent` and hands it to `wai.simulate` with the
recipe's own `TOOLS` and `POLICY`, which run.py defines. The check runs in a
copy of the recipe directory, so they come from there.
"""

from run import POLICY, TOOLS  # noqa: F401
