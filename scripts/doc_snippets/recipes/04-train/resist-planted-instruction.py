"""The reader's side of recipes/04-train/resist-planted-instruction.

The page's last block pushes `selected`, the rows run.py picked with
`wai.select_for_sft` (the best passing completion per prompt) out of a
graded hosted run. There is no key here, so the graded rows are the shared
fixture's offline run, selected by the same rule.
"""

from _common import rows as _rows

import whileai.simulations as _wai

selected, _report = _wai.select_for_sft(
    _rows, min_reward=1.0, target=8, select="top_per_prompt", seed=0
)
