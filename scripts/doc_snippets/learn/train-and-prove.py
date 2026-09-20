"""The reader's side of docs/learn/train-and-prove: the rows the Modal run wrote.

Step 2 of the lesson is ``modal run recipes/04-train/sft/train_modal.py``,
which needs a Modal account and an A10G, so the checker cannot run it.
Step 3 reads ``holdout_rows.jsonl``, the file that run writes next to the
export. This fixture puts the rows from the run the lesson quotes
(``recipes/04-train/sft/runs/lesson7/holdout_rows.jsonl.gz``) in the
page's working directory, so step 3 runs offline on the real rows and the
numbers on the page are checked against them.
"""

import gzip
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_ROWS = _REPO / "recipes" / "04-train" / "sft" / "runs" / "lesson7" / "holdout_rows.jsonl.gz"
Path("holdout_rows.jsonl").write_bytes(gzip.decompress(_ROWS.read_bytes()))
