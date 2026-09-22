"""Every number that steers a sample in this recipe, named and sourced.

CONSTITUTION.md, belief 3: every default is named, sourced and tunable from
the call; a default with no source says "convention, untested". These were
literals inside ``train_modal.py`` until whilehq/whileai-sdk#788 — a
``temperature=0.8`` in the eval sampler, another in ``GRPOConfig``, and a
``seed=17`` that seeded the trainer but not the sampler the eval was read
from. A number that decides what a run measures does not live in the call
that uses it.

Each is an argument on ``train()`` and a flag on ``modal run``, so the call
moves it without editing this file.

    # NAME = value: why (source)
"""

from __future__ import annotations

# SAMPLE_TEMPERATURE = 0.8: the eval and the rollouts are drawn at the same
# temperature, so the pass@1 before and after describe the same policy the
# trainer saw. 0.8 is inside the 0.7-1.0 band the book gives for drawing
# several completions per prompt (Lambert 2025, chapter Rejection Sampling,
# rlhfbook.com/c/09-rejection-sampling.html); TRL's GRPOConfig default is
# 1.0, and this recipe runs a 1.5B instruct model, where the lower end of
# the band keeps the wire format readable.
SAMPLE_TEMPERATURE = 0.8

# SAMPLE_TOP_P = 0.95: nucleus sampling at the value the paper that
# introduced it reports (Holtzman et al., The Curious Case of Neural Text
# Degeneration, arXiv:1904.09751, section 4).
SAMPLE_TOP_P = 0.95

# SAMPLE_SEED = 17: the sampler's own seed, so pass@1 before and pass@1
# after are two draws from the same generator and not two draws from
# whatever torch's global state happened to be (convention, untested: no
# source picks a seed, only the rule that one is recorded -- CONSTITUTION.md
# belief 1, a number is a result only with its seed).
SAMPLE_SEED = 17

# BASE_EVAL_RUNS = 3: the base is evaluated this many times and the spread
# across the re-runs is the eval's own noise, so a delta smaller than it is
# never called a result (Lambert 2025, chapter Evaluation,
# rlhfbook.com/c/16-evaluation.html; the same 3 the paper recipes use and
# the same one recipes/papers/check.py enforces).
BASE_EVAL_RUNS = 3

__all__ = [
    "BASE_EVAL_RUNS",
    "SAMPLE_SEED",
    "SAMPLE_TEMPERATURE",
    "SAMPLE_TOP_P",
]
