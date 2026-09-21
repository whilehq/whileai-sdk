"""Every number the run engine used to carry inline, named, explained and
tunable.

Each constant below has a comment of the form ``# <name> = <value>: <why>
(<source>)``. The source is a measurement, a paper cited as author, year and
arXiv id, a chapter title of Lambert 2025 (arXiv:2504.12501), or the honest
words "convention, untested". A constant with no source is a bug.

Two channels reach the code:

* the module-level constants, for values shared across files (the same
  number must mean the same thing everywhere it appears);
* :class:`RunKnobs`, one field per engine knob, which ``simulate()``
  reads from ``advanced={...}`` under the field's name and validates
  against the bounds in the field's metadata. A knob out of range is a
  ``ValueError`` that names the bound.

Sections are per owner so parallel work reconciles by section, not by
line: ``run/`` (engine, config, rows, simulation, data) and ``generate/``
(agents, generator, diversity, ...) are below. Other stages add their own
section.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields
from types import MappingProxyType
from typing import Any, TypeVar

_T = TypeVar("_T")

# ---------------------------------------------------------------------
# shared across stages
# ---------------------------------------------------------------------

# DEFAULT_BUDGET = 1000: the row cap when simulate() is given none. A round
# number; large enough for a coverage curve to flatten on a three-tool
# agent (explore runs plateau under 300 rows) and small enough to finish on
# a trial key. (convention, untested)
DEFAULT_BUDGET = 1000

# DEFAULT_CONCURRENCY = 32: parallel rollouts, and the cap on parallel
# judge calls. One served vLLM replica's continuous-batching sweet spot
# for short chat turns; no throughput curve was recorded for this
# package. (convention, untested)
DEFAULT_CONCURRENCY = 32

# PASS_THRESHOLD = 0.5: a reward under this is a failure. The outcome
# label is binary, r in {0, 1} (Lambert 2025, chapter Reward Modeling,
# outcome reward models; Lambert et al. 2024, arXiv:2411.15124, verifiable
# rewards gate on all assertions passing), so 0.5 is its midpoint and a
# partial rubric score (or the conduct advisory 0.5 for a truncated reply)
# rounds to the nearer verdict. No source names another cut. The run loop's
# graded-failure gate, score/ audits and the exporters count fails the
# same way; import this rather than writing 0.5 again.
PASS_THRESHOLD = 0.5

# PASS_REWARD = 1.0: the reward a fully passing row carries; anything
# below it counts as failing in the arm-yield tally. (the [0, 1] reward
# scale every judge in the package writes)
PASS_REWARD = 1.0

# SHORT_HASH_CHARS = 16: hex chars kept from a sha256 when a row names a
# system prompt, a fingerprint or a policy hash. 64 bits: collisions
# among the prompts one account will ever run are not a concern.
# (convention, untested)
SHORT_HASH_CHARS = 16

# MAX_COMPLETIONS_PER_REQUEST = 8: writer completions one request may ask
# for (the ``n`` of a chat call). vLLM prefills once for all of them and
# above eight a busy endpoint dropped the request (hosted pool). The
# generate/ section reads it as MAX_SAMPLES_PER_CALL (one value, one home).
# (convention, untested)
MAX_COMPLETIONS_PER_REQUEST = 8

# FAULT_STATUSES = {error, timeout, not_found, denied, malformed}: a tool
# result ``status`` that means the call failed
# (the run's own vocabulary; a status outside this set is the tool's own
# word, not a fault, #261). ``score.grading._KNOWN_FAULTS`` is the wider
# alias table; this is the subset the engine re-rolls and mutates on.
FAULT_STATUSES = frozenset({"error", "timeout", "not_found", "denied", "malformed"})

# OK_STATUSES = {ok, success}: a tool result ``status`` that means the call
# worked (the run's own vocabulary, #261).
OK_STATUSES = frozenset({"ok", "success"})

# MESSAGE_EXAMPLES = 5: items a warning or an error names before ", ..."
# (unknown ids, bad rows, tools with no result shape); enough to spot the
# pattern without the message becoming the list. (convention, untested)
MESSAGE_EXAMPLES = 5

# LEAK_MIN_QUOTE_CHARS = 12: the shortest run of characters that counts as
# quoting the privileged block. Shorter matches are common words.
# Mirrors ``score.privileged.leak_report(min_len=)``. (convention, untested)
LEAK_MIN_QUOTE_CHARS = 12

# ---------------------------------------------------------------------
# run/: engine, config, rows, simulation, data
# ---------------------------------------------------------------------

# RL_ROLLOUTS_PER_PROMPT = 8: k under mode="rl". A grouped update (GRPO,
# Shao et al. 2024, arXiv:2402.03300) needs enough samples per
# prompt for the group mean to be a usable baseline. 8 is what Dr. GRPO
# trains with (arXiv 2503.20783, 8 responses per question) and the
# agentic-RL recipe in arXiv 2603.21972 (G=8, section 4.1); DAPO (arXiv
# 2503.14476), ProRL (arXiv 2505.24864) and Skywork-OR1 (arXiv
# 2505.22312) use 16. The package differs from the 16 camp on purpose.
# The only measured curve is for eval resampling, not group size: Miller
# (arXiv 2411.00640, section 3.1) shows K=2 samples per question cut the
# variance of an eval score by a third and K=4 by a half in the
# uniform-difficulty example, with diminishing returns after; that the
# next doubling buys under a sixth more is derived from the same 1/K
# shape, not stated there. So 8 is a first look at half the 16 cost;
# raise repeats= to 16 for a training set on hard prompts, where 8
# leaves more zero-variance groups.
RL_ROLLOUTS_PER_PROMPT = 8

# SFT_PHRASINGS_PER_SITUATION = 3: n under mode="sft". Three wordings of
# one situation give the SFT set paraphrase variety without tripling the
# situation count. (convention, untested)
SFT_PHRASINGS_PER_SITUATION = 3

# SFT_COMPLETIONS_PER_PROMPT = 4: k under mode="sft", the completions of
# one phrasing that ``select_for_sft(select="top_per_prompt")`` chooses
# among. It was 1, and at k=1 there is nothing to choose: the pick is a
# pass/fail filter, and ``random_per_prompt``, the control
# Lambert 2025, chapter Rejection Sampling, asks for, returns the same rows,
# so a claimed gain from selection could not be checked against chance.
# With a binary judge best-of-k is a pass@k yield: a prompt the policy
# passes 30% of the time ships a demonstration 30% of the time at k=1 and
# 76% at k=4 (1 - 0.7^4), so the set keeps the prompts inside the 20-80
# band instead of the easy ones (Lambert 2025, chapter Reasoning,
# difficulty filtering). Four is under the 10 to 30 that Lambert 2025
# (chapter Rejection Sampling) and Llama 3
# (arXiv:2407.21783, section 4.2.2) sample per prompt, on purpose: it is
# the smallest k the package already reports a k-way number on
# (ROLLOUTS_PER_TASK, tau2-bench's pass^4) at four times the rollouts, not
# ten. ``repeats=`` moves it; the SFT report says ``pass_filter`` when the
# rows it was given still carry one completion per prompt. (convention,
# untested against 1 or 10 on an SFT delta)
SFT_COMPLETIONS_PER_PROMPT = 4

# DEFAULT_PROBE = 2: rollouts a prompt gets before the run decides whether
# its group splits; two is the least that can disagree. A unanimous group
# carries no gradient, so DAPO (arXiv 2503.14476, eq. 11) and ProRL (arXiv
# 2505.24864) drop prompts whose accuracy is 0 or 1 after the fact, and
# GRESO (arXiv 2506.02177) predicts zero-variance prompts before spending
# rollouts on them. Probing first is the engine's version of that; no
# paper states a probe size, so 2 is the structural minimum, not a
# measured optimum.
DEFAULT_PROBE = 2

# DEFAULT_FAULT_RATE = 0.5: the share of fault-tagged grid cells that keep
# their sandbox fault plan outside mode="rl". This is not a per-call
# failure rate: tagged cells are a small slice of the grid (SUCCESS_SHARE
# in generate/scenarios.py flips nine in ten to success first), so rows
# with a fault stay under about 10% of a run at 0.5, inside the band
# training-time injection is stable in (arXiv 2603.21972, section 4.6: a
# 3B agent held its test score with tool failures injected at up to 5%
# per call and degraded at 10%; AgentCE-Bench, arXiv 2604.06111,
# evaluates p in {0, 0.1, 0.3}). Independent of failure mutation (that is
# a purposeful-fail arm, this is tool faults). ``fault_rate=`` or
# ``risk=`` moves it; 0 disables injection. (convention, untested against
# other shares inside the band)
DEFAULT_FAULT_RATE = 0.5

# RL_FAULT_RATE = 0.8: the same share under mode="rl". RL raises it because
# the faulted cells are where a base fails (Lambert 2025, chapter Reasoning,
# difficulty filtering) and PALADIN trains on an 80/20 composition of
# recovery-bearing to clean traces (arXiv 2509.25238, appendix I.4: a
# dataset mix, not a keep rate), which is the shape 0.8 gives the tagged
# slice; the per-row rate stays inside the 2603.21972 / 2604.06111 band
# for the same reason as above. Untested against 0.5 on a training run.
RL_FAULT_RATE = 0.8

# DEFAULT_AVG_TURNS = 12: target thread length (user and agent turns
# together) for simulate(), local_model() and the turn sampler alike. The
# person speaks at most avg_turns // 2 times, so 12 leaves six user turns:
# room to verify, look up, confirm and write. Measured on this package
# (#299, 4000 rows): mean user depth 2.13 / 4.02 / 5.28 and 45% / 70% /
# 78% of threads reaching a third user turn at avg_turns 6 / 12 / 16.
# Confirm-before-acting needs that third turn (ask, name the action and
# ask, yes, act), so at 6 it is missing from over half the rows and no
# selection downstream can recover it (DAPO, arXiv 2503.14476: a group
# whose rollouts all fail has zero advantage and no gradient). The cost is about three agent calls
# per row instead of one (#299). The chat-assistant means in the
# literature are lower (SimulatorArena, arXiv 2510.05444: 7.8 and 6.9
# turns per human thread) but carry no confirm step; the caps sit well
# above (tau-bench stops at 30 agent actions, arXiv 2406.12045; APIGen-MT
# trajectories top out at 29 turns, arXiv 2504.03601). local_model()
# used to default to 6 on its own; it now reads this constant. Fit it from
# traces= when you have them.
DEFAULT_AVG_TURNS = 12.0

# DEFAULT_MIN_USER_TURNS = 1: the person always speaks at least once.
DEFAULT_MIN_USER_TURNS = 1

# AGENT_MAX_TOKENS_FLOOR = 64: the smallest reply budget simulate() accepts;
# below it every reply is cut mid-sentence and scores 0. (convention, untested)
AGENT_MAX_TOKENS_FLOOR = 64

# DEFAULT_SEED = 0: the seed every draw starts from when simulate() is
# given none; zero so two runs with no seed argument agree. (convention, untested)
DEFAULT_SEED = 0

# DEFAULT_POOL_SIZE = 80: prompts the situation writer keeps per round
# (advanced per_round). (convention, untested)
DEFAULT_POOL_SIZE = 80

# DEFAULT_WRITER_FLIGHT = 4: writer waves in flight. Too many starves
# rollouts of the GPU; how many is too many was never measured.
# (convention, untested)
DEFAULT_WRITER_FLIGHT = 4

# DEFAULT_CARDS_PER_WAVE = 8: situation cards one writer wave asks for.
# (convention, untested)
DEFAULT_CARDS_PER_WAVE = 8

# DEFAULT_COMPLETIONS_PER_REQUEST = 1: completions one writer request asks
# for unless advanced completions_per_request raises it; one keeps a
# request cheap to retry. (convention, untested)
DEFAULT_COMPLETIONS_PER_REQUEST = 1

# DEFAULT_EXTRA_CARDS = 1: spare cards per wave so a rejected card does not
# leave the wave short. (convention, untested)
DEFAULT_EXTRA_CARDS = 1

# SATURATION_CAP = 50_000: rows a saturation-bounded run may produce before
# the loop gives up when budget=None. (convention, untested)
SATURATION_CAP = 50_000

# HUNG_SLOT_S = 45.0: kill a hung slot after this wait; the row is
# omitted. Ping-pong is several HTTP calls; 24 s dropped healthy
# two-person traces and the replacement oversubscribed the GPU.
# (observed on the hosted endpoint; 45 is a convention above that)
HUNG_SLOT_S = 45.0

# STOP_GRACE_S = 5.0: after a stop, wait this long for rollouts already
# running; queued ones are cancelled at once. (convention, untested)
STOP_GRACE_S = 5.0

# DEAD_AGENT_MIN_ERRORS = 16: an agent that raises on every call is called
# off once this many rollouts were lost with no row landed, or
# DEAD_AGENT_BUDGET_MULTIPLE x budget, whichever is larger (#88).
# (convention, untested)
DEAD_AGENT_MIN_ERRORS = 16
# DEAD_AGENT_BUDGET_MULTIPLE = 2: the budget multiple in that rule; twice
# the rows asked for is more failures than any live agent produces.
# (convention, untested)
DEAD_AGENT_BUDGET_MULTIPLE = 2

# ALLOC_GAIN = 4.0: how hard a hot trace region pulls cell weight toward
# itself; a region with budget share s and a full recipe match multiplies
# the cell weight by 1 + 4 s. (convention, untested)
ALLOC_GAIN = 4.0

# TIER_MIX_MIN_ROWS = 20: rows before the realized difficulty mix is
# compared with the ask. At 20 rows a share of 0.4 has binomial sd
# sqrt(0.4 x 0.6 / 20) = 0.11, about the tolerance below, so a smaller
# run cannot tell a shortfall from noise. (binomial sd; the 20 is the
# smallest n that gets the sd to the tolerance)
TIER_MIX_MIN_ROWS = 20
# TIER_MIX_TOLERANCE = 0.10: how far (in share) the drawn rows may land
# below the ask before the run says so. (convention, untested)
TIER_MIX_TOLERANCE = 0.10

# PROGRESS_MIN_ROWS_FOR_ESTIMATE = 5: rollouts before the rate is worth
# extrapolating; before that the estimate is the first rollout's latency
# dressed up as a forecast. (convention, untested)
PROGRESS_MIN_ROWS_FOR_ESTIMATE = 5
# PROGRESS_MIN_BUDGET = 10: runs smaller than this say nothing; they are
# over before a line helps. (convention, untested)
PROGRESS_MIN_BUDGET = 10

# TIMEOUT_TOKENS_PER_SECOND = 4: the per-request decode rate the default
# call timeout is sized for. simulate(timeout=) defaults to
# max(LOCAL_MODEL_TIMEOUT, reply budget / this), so a 4,096-token reply
# budget gets 1,024 s instead of the flat 300 s that re-rolled every long
# reply on a 9B model at 16-32 requests in flight on one L40S and turned a
# two-hour run into six and a half (whilehq/whileai-sdk#470). 4 tokens a
# second is the slow end of what that server did under load; the flat
# floor still holds for short replies. (convention, untested on other
# servers)
TIMEOUT_TOKENS_PER_SECOND = 4

# SYSTEM_PROMPT_HEAD_CHARS = 120: opening chars of the system prompt kept
# on every row, enough to tell a numbered policy from a bare prompt at a
# glance (#296). (convention, untested)
SYSTEM_PROMPT_HEAD_CHARS = 120

# PARENT_HEAD_CHARS = 80: chars of a failing prompt that name it as a
# mutation parent; the mutated row's ``parent`` and the aim table key
# on the same head, so the two must agree. (convention, untested)
PARENT_HEAD_CHARS = 80

# SCENARIO_ID_CHARS = 6: hex chars of the prompt hash that name a probe
# row's scenario_id (24 bits; ids are per run). (convention, untested)
SCENARIO_ID_CHARS = 6

# REPORT_LIST_ITEMS = 20: how many missing pinned prompts a report lists
# before it stops. (convention, untested)
REPORT_LIST_ITEMS = 20

# FINGERPRINT_STEM_MIN_LEN = 4: words longer than this lose a trailing
# "s" before fingerprinting, so "refunds" and "refund" match and "is"
# stays "is". (convention, untested)
FINGERPRINT_STEM_MIN_LEN = 4

# TOOL_SCHEMA_SPAN_CHARS = 500: how far apart "name", "description" and
# "parameters" may sit in visible text and still read as a leaked tool
# schema. (convention, untested)
TOOL_SCHEMA_SPAN_CHARS = 500

# DEFAULT_LLM_JUDGE_CONCURRENCY = 16: parallel calls to the hosted LLM
# judge; half the rollout concurrency because the judge is one model
# serving every account. (convention, untested)
DEFAULT_LLM_JUDGE_CONCURRENCY = 16

# JUDGE_COMPARE_CONCURRENCY = 8: parallel calls per judge while
# ``compare_judges`` runs; judges run one after another so a rate limit
# on one provider never slows the others (convention, untested).
JUDGE_COMPARE_CONCURRENCY = 8
# JUDGE_CONCURRENCY_CAP = 32: the most parallel judge calls ``grade()``
# lets a caller ask for, whatever ``concurrency=`` says. (convention,
# untested)
JUDGE_CONCURRENCY_CAP = 32

# DEFAULT_SELECT_TARGET = 1000: rows ``select()`` / ``export_training()``
# aim for when the caller names no target. (convention, untested)
DEFAULT_SELECT_TARGET = 1000

# HOLDOUT_BUCKET_HEX_CHARS = 8: hex chars of the task hash that place a
# task on the train or holdout side; 32 bits is fine resolution for a
# fraction. (convention, untested)
HOLDOUT_BUCKET_HEX_CHARS = 8

# ---------------------------------------------------------------------
# generate/: HTTP transport (agents.py, anthropic_backend.py)
# ---------------------------------------------------------------------

# TRANSIENT_TRIES = 3: a 5xx, a 429 (Anthropic) or a dropped Modal request
# is retried this many times before the rollout is lost (convention,
# untested; both backends read it so neither is silently flakier).
TRANSIENT_TRIES = 3
# TRANSIENT_BACKOFF_S = 0.4: the first sleep before a transient retry,
# doubled each try (0.4, 0.8, 1.6 s) (convention, untested).
TRANSIENT_BACKOFF_S = 0.4
# MAX_SAMPLES_PER_CALL = 8: the most completions one request asks for
# with ``n``; the same cap run/ calls MAX_COMPLETIONS_PER_REQUEST (above),
# so completions_per_request and the writer's ``n`` cannot drift apart.
MAX_SAMPLES_PER_CALL = MAX_COMPLETIONS_PER_REQUEST
# MIN_REPLY_TOKENS = 256: no request asks for fewer reply tokens than
# this; the input is shrunk instead, because a reply cut under 256 tokens
# is a fragment the junk gate drops anyway. Both HTTP backends read it
# (convention, untested).
MIN_REPLY_TOKENS = 256
# DECISION_TIMEOUT_S = 30: seconds one TypeSafe decision request may
# take (``typesafe:`` judge, typesafe_backend.py). Jev's stated end-to-end
# latency is 70 to 500 ms and its own SDK's default timeout is 10 s;
# 30 s leaves room for a queue behind its 1,200-requests-a-minute limit
# without a stuck request holding a judge worker for the chat judge's
# 120 s (convention, untested; the vendor numbers are from
# typesafe.ai/blog/introducing-system-one-models-and-jev).
DECISION_TIMEOUT_S = 30.0

# SAMPLING_TEMPERATURE_MAX = 2.0: the ceiling every temperature knob is
# validated against (user_temperature, the hosted trainer's rollout
# temperature); the OpenAI-compatible chat API's own range is 0 to 2, and
# above about 1.2 sampling is noise on every model measured (ProRL trains
# at 1.2, arXiv 2505.24864). (the API's range)
SAMPLING_TEMPERATURE_MAX = 2.0

# LOCAL_MODEL_TEMPERATURE = 0.8: sampling temperature of a model-backed
# rollout unless simulate(advanced={"temperature": ...}) says otherwise,
# recorded on every row under ``sampling``. Inside the 0.7 to 1.0 band
# rejection sampling is run at (Lambert 2025, chapter Rejection Sampling,
# "Implementation Details") and under the 1.0 RL rollouts use (DAPO
# 2503.14476, group size 16); the agent benchmarks that want reproducible
# scores run at 0 (tau-bench 2406.12045, tau2-bench 2506.07982), which is
# what ``reproducible=`` and an explicit temperature are for. 0.8 within
# the band is a convention. The monitor samples its holdout at the same
# value (MONITOR_SAMPLE_TEMPERATURE).
LOCAL_MODEL_TEMPERATURE = 0.8

# ---------------------------------------------------------------------
# generate/: the rule axis (scenarios.py; read by score/preflight.py)
# ---------------------------------------------------------------------

# RULE_AXIS_CAP_GRID = 16: policy clauses that become cells of the
# generation grid's rule axis. The pairwise covering array grows with its
# largest axis, so the grid stays bounded; the environment variable
# ZP_RULE_CAP overrides it for one process. Clauses past the cap are
# dropped in document order and the run says how many (#391).
# (convention, untested)
RULE_AXIS_CAP_GRID = 16
# RULE_AXIS_CAP_REPORT = None: ``coverage_gap`` and ``preflight`` report
# over a suite that already exists, so there is no grid to bound and every
# clause is on the axis. A 68 KB production prompt had about 160
# imperative clauses; the first 16 were banner text ("Read it.") and the
# report said "14 of 16 covered" (#391). ``rule_cap=`` on either call
# sets a number.
RULE_AXIS_CAP_REPORT = None

# ---------------------------------------------------------------------
# generate/: context budgets (agents.py, generator.py)
# ---------------------------------------------------------------------

# CHARS_PER_TOKEN = 3: the token estimate every context budget is sized
# with, for the agent's history and the writer's prompt alike. English
# tokenizers sit at 3.5 to 4.5 chars per token, so this over-counts and
# the budget errs on the safe side (convention, untested on this data).
CHARS_PER_TOKEN = 3

# ---------------------------------------------------------------------
# score: statistics
# ---------------------------------------------------------------------
#
# The statistics every verdict in ``score/`` rests on. ``holdout_size``,
# ``detectable_effect``, ``noise_band``, ``bootstrap_ci``, ``compare_runs``
# and ``delta_report`` take these as keywords (``alpha=``, ``power=``,
# ``level=``, ``n_boot=``); the constants are only their defaults.

# ALPHA = 0.05: two-sided false-positive rate behind every interval and
# verdict. The convention the evaluation literature runs on (Miller 2024,
# arXiv:2411.00640, section 5, plugs alpha=0.05 into its power example).
# Lambert 2025, chapter Evaluation, names no level. Untested against any
# other value.
ALPHA = 0.05
# CI_LEVEL = 0.95: the interval every ``ci95`` key carries. ``1 - ALPHA`` so
# the interval and the verdict agree: a delta whose interval excludes zero
# at CI_LEVEL is the one a test at ALPHA rejects.
CI_LEVEL = 1.0 - ALPHA
# Z_95 = 1.96: the normal quantile at CI_LEVEL, rounded to the two decimals
# every table prints (Miller 2024 writes the interval as 1.96 x SE). Kept
# at 1.96 rather than 1.959964 so a printed band can be checked by hand;
# the difference moves a band in the fourth decimal.
Z_95 = 1.96
# POWER = 0.8: the chance a holdout of the size ``holdout_size`` names
# detects a real gain. beta = 0.20 is the example Miller 2024 (section 5)
# and the classical power literature use; Lambert 2025, chapter Evaluation,
# says the point of a better eval is statistical power when comparing
# training runs, without naming a number.
POWER = 0.8
# BOOTSTRAP_DRAWS = 2000: resamples behind a percentile interval. Efron and
# Tibshirani put the floor for percentile intervals at 1000; 2000 halves
# the Monte Carlo error on the endpoints and still runs in well under a
# second on a few hundred tasks. Convention above the floor, untested
# against 1000.
BOOTSTRAP_DRAWS = 2000
# MIN_CI_TASKS = 3: tasks a bootstrap interval needs. Below three the
# resampled statistic is one of a handful of arrangements of the data
# itself, so the interval says nothing (convention, untested).
MIN_CI_TASKS = 3
# MIN_RERUNS = 3: re-runs of one eval before ``run_std`` is read as a
# spread. Two runs give one difference, not a distribution; three is the
# fewest that give a sample sd with two degrees of freedom
# (Lambert 2025, chapter Evaluation: a held-constant eval moves 0.25 to
# 1.5 points between runs; convention on the count).
MIN_RERUNS = 3
# MIN_TRAIN_SEEDS = 2: independently trained seeds per arm before
# ``delta_report(train_runs=)`` may call a delta between two trained
# models "moved"; one seed per arm reads ``unresolved``. The claim is a
# delta between two separately trained models, and the eval re-run floor
# (``run_std``) measures only the eval, so one seed cannot separate the
# change from run-to-run training variance (#356: the same recipe
# flipped sign, -0.065 to +0.050, between two runs at one seed). Two is
# the fewest that give a between-seed spread at all; the between-seed
# term is the seed-to-seed variance of each arm's mean (Lambert 2025,
# chapter Evaluation; Miller 2024, arXiv:2411.00640, on adding the
# variance components a claim rests on). Convention on the count.
MIN_TRAIN_SEEDS = 2
# BASE_PASS_RATE = 0.6: the before-side pass rate ``holdout_size`` assumes
# when no rows are given. The centre of the 20-80 difficulty band, where a
# binary task carries the most variance and the sizing is most
# conservative. Measured lanes sat between 0.5 and 0.7 (#288).
BASE_PASS_RATE = 0.6
# CEILING_PASS_RATE = 0.9: a before side passing this share of its tasks
# has at most 10 points of room, under the noise band of most agent evals
# (run_std 0.02-0.04 measured across our lanes gives a band of 0.06-0.11),
# so ``delta_report`` flags ``ceiling`` and ``holdout_size(before=rows)``
# refuses to size on the collapsed variance (a saturated suite read as
# "2 tasks are enough", #392). Above DIFFICULTY_BAND's top (0.8) on
# purpose: the band prunes training prompts, the ceiling flags an eval.
# Convention on the exact share.
CEILING_PASS_RATE = 0.9
# ROLLOUTS_PER_TASK = 4: the per-task rollout count the sizing assumes and
# the smallest k ``pass_at`` reports pass^k at. tau-bench (arXiv:2406.12045)
# plots pass^k to k=8 from at least 3 trials; tau2-bench (arXiv:2506.07982)
# runs every task 4 times and reports pass^1 and pass^4. Four is the
# smallest count those benchmarks report a k-way number on.
ROLLOUTS_PER_TASK = 4
# PROVE_EFFECT = 0.05: the gain in pass rate the package asks a held-out
# set to prove: ``eval_power(rows)`` reads it as the effect to size for,
# and a pushed holdout is sized to it (PLATFORM_HOLDOUT_PROVE_EFFECT is
# this value under the platform name). Five points is the package's proof
# bar (a 5-point move on 50+ judged tasks); Lambert 2025, chapter Evaluation,
# puts held-constant eval noise at 0.25 to 1.5 points, so five is several
# noise floors. (convention, untested; sits above the measured noise)
PROVE_EFFECT = 0.05

# ---------------------------------------------------------------------
# score: difficulty band
# ---------------------------------------------------------------------
#
# DIFFICULTY_BAND = (0.2, 0.8): keep tasks the current policy passes between
# 20% and 80% of the time. Lambert 2025, chapter Reasoning ("Common
# Practices in Training Reasoning Models"): difficulty filtering restricts
# RL prompts to those
# the starting model solves 20-80% of the time, measured from N=16
# samples. DAPO (arXiv:2503.14476) is the online form: groups with
# accuracy 0 or 1 are dropped from the batch (G=16). The band edges are a
# reported practice, not an ablation, so every selector takes ``band=``.
DIFFICULTY_BAND: tuple[float, float] = (0.2, 0.8)
# DIFFICULTY_BAND_ROLLOUTS = 16: rollouts per task the band is measured
# from in the sources above (Lambert 2025, chapter Reasoning, N=16; DAPO
# G=16). Below it
# a task's band assignment carries a Wilson half-width near 0.3 at k=8.
DIFFICULTY_BAND_ROLLOUTS = 16
# REJECTION_SAMPLING_MIN_K = 10: completions per prompt a best-of-N pick
# wants. Llama 3 (arXiv:2407.21783, section 4.2.2) samples K between 10 and
# 30 per prompt; Lambert 2025, chapter Rejection Sampling ("Implementation
# Details": 10 to 30 or more completions per prompt) repeats the range.
# Fewer makes the pick a filter, not a choice.
REJECTION_SAMPLING_MIN_K = 10
# RL_ROLLOUTS_PER_ASK = 8: the rollouts per ask ``recommend`` sizes an RL
# run for; the same quantity as RL_ROLLOUTS_PER_PROMPT (mode="rl" k) under
# the score/ name, so the two cannot drift. One value, one home.
RL_ROLLOUTS_PER_ASK = RL_ROLLOUTS_PER_PROMPT

# ---------------------------------------------------------------------
# score: decontamination
# ---------------------------------------------------------------------
#
# DECONTAM_NGRAM = 8: word n-gram behind the near-copy rule. Tulu 3
# (arXiv:2411.15124) decontaminates on 8-gram overlap between training
# prompts and evaluation prompts; Lambert 2025, chapter Evaluation, found
# its own contaminations with the same 8-gram test.
DECONTAM_NGRAM = 8
# DECONTAM_OVERLAP = 0.8: share of a row's words one eval text has to
# cover with shared 8-grams. The Llama 2 rule (80% of tokens). Tulu 3 uses
# 50%; this package keeps 80% on purpose: template-written situations
# share whole sentences that say nothing about which question was asked,
# and at 50% the near-copy rule flags rows that never saw the eval
# question (#286). ``overlap=`` sets it per call.
DECONTAM_OVERLAP = 0.8
# SEMANTIC_SIMILARITY = 0.85: cosine at or above which two prompts read as
# one task to an embedder. Read off BGE-small (unrelated prompts score
# about 0.55 there, paraphrases above 0.85, #286). SemDeDup
# (arXiv:2303.09540) tunes its eps per dataset to hit a target size: the
# eps 0.03 to 0.07 settings (cosine 0.93 or higher) are its LAION runs on
# CLIP embeddings, and C4 on OPT embeddings needs different eps for the
# same fraction kept (its figure A17). Either way that is a duplicate
# threshold for pretraining text, not a paraphrase threshold for task
# prompts, so this package sits lower and calibrates against the eval
# set's own distinct-task similarity at run time.
SEMANTIC_SIMILARITY = 0.85

# ---------------------------------------------------------------------
# score: judge floors
# ---------------------------------------------------------------------
#
# MIN_GOLD = 50: human labels before a judge's accuracy number means
# anything. Lambert 2025, chapter Reward Modeling ("Suggested
# Experiments"): a 50- to 200-example held-out set is the size it asks
# for to evaluate a reward model; below 50 the Wilson interval on
# agreement is about +/-0.1 wide.
MIN_GOLD = 50
# MAX_GOLD_ASK = 200: the most labels the trust report will ask a person
# for before it says the judge itself is the problem (the top of the same
# 50-200 range).
MAX_GOLD_ASK = 200
# MIN_AGREEMENT = 0.8: Wilson lower bound of judge-human agreement a judge
# has to reach. Zheng et al. (arXiv:2306.05685, MT-Bench): human-human
# agreement is 81% and GPT-4 reaches 85% against humans, so a judge under
# 80% agrees with people less than people do with each other.
MIN_AGREEMENT = 0.8
# MIN_KAPPA = 0.6: chance-corrected agreement floor. Landis and Koch (1977)
# call 0.61-0.80 "substantial"; 0.6 is the bottom of that band. A 2026
# sweep of 21 judges (arXiv:2606.19544, run March to April 2026) measured
# Cohen's kappa 0.376 to 0.511 against human preference labels on
# MT-Bench, so this floor is demanding on purpose: it asks for agreement
# beyond what raw accuracy hides.
MIN_KAPPA = 0.6
# JUDGE_CHECK_SAMPLE = 40: rows a judge check (perturbation, probes, the
# verifier audit) re-judges by default. Enough that a 10% effect shows as
# four flips; kept under MIN_GOLD because these checks cost a judge call
# per row (convention, untested).
JUDGE_CHECK_SAMPLE = 40
# LENGTH_GAP_FLAG = 0.15: judge pass rate gap between short and long
# replies with the same human label that reads as length bias. Zheng et
# al. (arXiv:2306.05685) found a 91% failure rate on a verbosity attack
# for GPT-3.5 and 9% for GPT-4; the 2026 sweep (arXiv:2606.19544) reports
# verbosity bias under 0.011 on its own pairwise measure. Those two are
# not on one scale (an attack failure rate against a bias coefficient),
# so 0.15 is not read off either: it is a convention, a gap wide enough
# to show as six flips on a JUDGE_CHECK_SAMPLE of 40.
LENGTH_GAP_FLAG = 0.15
# FLIP_FLAG = 0.10: share of verdicts that change on an identical re-judge,
# on appended filler, or under a probe, before the judge is called
# exploitable. Test-retest consistency of current judges is 0.889 to
# 0.992 on MT-Bench (arXiv:2606.19544), so a tenth of verdicts moving is
# far outside the measured range. Convention on the exact number.
FLIP_FLAG = 0.10
# PROBE_MIN_N = 20: rows a judge probe needs in its denominator (originally
# failing replies for an additive probe, re-judged replies for a
# replacement one) before ``flagged`` may be true. Under it one flipped row
# is already FLIP_FLAG: 1 of 10 is 0.10 exactly, and its 95% Wilson
# interval runs 0.02 to 0.40, so the flag would rest on a single verdict
# (#347). At 20 one row is 0.05, half the flag, and the interval on 2 of
# 20 is 0.03 to 0.30. Below the floor the probe says "low power" with the
# rate it could resolve at POWER instead of flagging. Convention on the
# exact number (convention, untested).
PROBE_MIN_N = 20
# MAX_SKIPPED_SHARE = 0.10: share of a judge_trust gold sample the judge may
# leave out of the agreement count (a reward that is not exactly 0 or 1,
# so ``judge_agreement`` skips the row) before ``ok`` is false. The rows
# that survive a skip are not a random half: a Rubric of principles
# scores the mean of its criteria, so the skipped rows are the ones the
# judge was unsure about and the kept rows are the ones most likely to
# agree with anyone, which biases agreement upward by construction (#345:
# 40 of 80 labeled rows skipped, PASS at 100%). Held-out judge accuracy
# is measured over the whole labeled set or not at all
# (Lambert 2025, chapter Reward Modeling, "Suggested Experiments"); one in
# ten is the same tolerance FLIP_FLAG gives a re-judge. Convention on the
# exact number; ``judge_trust(max_skipped_share=)`` moves it.
MAX_SKIPPED_SHARE = 0.10
# POSITION_FLIP_FLAG = 0.2: share of pairs a pairwise judge decides
# differently when A and B are swapped before its position bias is a
# warning. Zheng et al. (arXiv:2306.05685, Table 2) measured 65%
# consistency for GPT-4 (35% flipped) and 46% for GPT-3.5; the swap-both-
# ways rule already turns a flip into a tie, so the flag marks a judge
# whose prompt needs work, not a broken report. Convention on the number.
POSITION_FLIP_FLAG = 0.2

# ---------------------------------------------------------------------
# score: judge payload
# ---------------------------------------------------------------------
#
# What the LLM judge is shown. Every cap is a character budget on the JSON
# the judge reads; ``grade_one``, ``apply_grade_llm`` and ``audit_grades``
# take ``payload_chars=`` to move the total.

# JUDGE_PAYLOAD_CHARS = 8000: the whole judge payload. About 2000 tokens,
# which leaves a 4B hosted judge with an 8k window room for its system
# prompt and reply. Measured: at this cap 37 of 120 rows on one paired
# eval were being cut mid-JSON before #290 taught the payload to shrink
# structure instead; the cap itself is the window, not a finding.
JUDGE_PAYLOAD_CHARS = 8000
# JUDGE_SITUATION_CHARS = 4000: the user request as the judge sees it.
# Half the payload: a situation longer than this is a document, and the
# verdict needs the reply and the steps more (convention, untested).
JUDGE_SITUATION_CHARS = 4000
# JUDGE_FINAL_TEXT_CHARS = 2000: the agent's final reply. A quarter of the
# payload; a reply past it is judged on its first 2000 characters and the
# cut is announced in the text (convention, untested).
JUDGE_FINAL_TEXT_CHARS = 2000
# JUDGE_POLICY_CHARS = 2000: the agent's policy or harness text, and each
# judge-only field (principle, reference). Same budget as the reply
# (convention, untested).
JUDGE_POLICY_CHARS = 2000
# JUDGE_MAX_TOKENS = 120: the judge's reply budget. Reason first, then a
# score, in one sentence: 4B judges needed room for the sentence and
# 120 tokens held every reply measured; a pairwise verdict is the same
# shape. Raise it for a judge asked to explain at length.
JUDGE_MAX_TOKENS = 120
# JUDGE_TEMPERATURE = 0.0: a judge is read at zero for stable ratings
# (Lambert 2025, chapter Reward Modeling, LLM-as-a-judge: "a common trick
# to improve the robustness of LLM-as-a-judge workflows is to use a
# sampling temperature of 0").
JUDGE_TEMPERATURE = 0.0
# DECISION_UNSURE_BAND = 0.1: a decision judge's (``typesafe:``) verdict
# probability within this of PASS_THRESHOLD, so 0.4 to 0.6, marks the row
# ``unsure`` in its judge_meta and the grade report counts them. Jev's
# stated property is calibration, higher confidence means higher accuracy
# (typesafe.ai/blog/introducing-system-one-models-and-jev), so a
# near-even probability is a row for a person to read, not a label to
# train on. The width is a convention, untested against gold;
# ``judge_trust`` on labeled rows is how to check it.
DECISION_UNSURE_BAND = 0.1

# ---------------------------------------------------------------------
# score: reply truncation
# ---------------------------------------------------------------------
#
# TRUNCATED_REPLY_CHARS = 600: a final reply longer than this that does not
# end on terminal punctuation or a sign-off is read as cut by the token
# cap (conduct advisory 0.5; junk for SFT). Short replies get the benefit
# of the doubt: a one-line answer often ends on a number or a name.
# Convention, untested; ``hygiene.is_truncated`` uses a looser 200 for its
# report-only count.
TRUNCATED_REPLY_CHARS = 600

# ---------------------------------------------------------------------
# score: reward hacks
# ---------------------------------------------------------------------
#
# HACK_THRESHOLD = 0.3: |corr(reward, feature)| at or above this is flagged
# as a shortcut the policy will learn. From the RLVR signal sweeps the
# scan descends from: the endorsed feature cleared 0.5 and delimiter
# hacks sat near 0.9, so 0.3 catches a hack before it dominates. Gao et
# al. (arXiv:2210.10760) is the mechanism: optimising a proxy the gold
# reward does not credit. Measured on our own lanes, not published.
HACK_THRESHOLD = 0.3

# ---------------------------------------------------------------------
# world (sandbox)
# ---------------------------------------------------------------------
#
# Reachable through ``MockEnvironment(options=WorldOptions(...))``,
# ``export_environment(world=...)`` / ``load_environment(world=...)`` and
# ``simulate(advanced={"world": {...}})``. The sandbox is seeded: changing a
# value here changes the golden rows (scripts/golden.py), so a change is a
# stated diff, never a drive-by.

# WORLD_DEFAULT_FAULT_MODE = "timeout": the fault a plan gets when it names
# none. Timeout is one of the six transition perturbations (runtime tool
# errors: timeout, rate limit, auth error, 5xx, malformed, schema drift)
# in arXiv:2605.11928 and among the orchestration failures arXiv:2606.01416
# lists; neither ranks them, so the pick is a convention: the one failure
# every tool can produce. Which mode a plan carries is decided upstream by
# the tool_condition axis through WORLD_CONDITION_MODES (read by
# generate/scenarios.py), not here.
WORLD_DEFAULT_FAULT_MODE = "timeout"
# WORLD_CONDITION_MODES = {timeout: timeout, ...}: the coverage grid's
# tool_condition value -> the fault mode the plan carries. The four
# shipped conditions map to the four shipped modes; a condition that is
# itself a ``fault_modes`` key (one a caller added through
# ``WorldOptions(fault_modes=)``) maps to that mode without an entry
# here. (the run's own vocabulary)
WORLD_CONDITION_MODES: Mapping[str, str] = MappingProxyType(
    {
        "timeout": "timeout",
        "malformed_result": "malformed",
        "stale_result": "stale",
        "permission_denied": "permission_denied",
    }
)
# WORLD_DEFAULT_FAULT_RATE = 1.0: a fault plan without a rate fires on every
# call. The injection benchmarks fire at 100% per sample so every model is
# tested at the same trajectory step (arXiv:2605.11928 §4.3); how often a
# situation carries a plan at all is ``simulate(fault_rate=)`` (default in
# generate/scenarios.py, DEFAULT_FAULT_RATE), a separate dial.
WORLD_DEFAULT_FAULT_RATE = 1.0
# WORLD_STALE_AS_OF = "3 days ago": the age stamped on a stale read. A
# stale answer is a real record with an age, so the agent can notice it
# is old (measured: a hash taught agents to invent a shipment around it and
# a rubric judge passed them). The age itself is convention, untested.
WORLD_STALE_AS_OF = "3 days ago"
# WORLD_MALFORMED_PAYLOAD = "<<garbled resp0nse": what a malformed fault
# returns. Unparseable text
# is the "Malform" perturbation of arXiv:2605.11928 (malformed JSON);
# convention for the exact bytes.
WORLD_MALFORMED_PAYLOAD = "<<garbled resp0nse"
# WORLD_EXISTS_SHARE = 0.7: share of user-named references that exist in the
# world when no state says otherwise. Under 1.0 so "not found" is a path
# the agent meets without a fault plan; the number is convention, untested.
WORLD_EXISTS_SHARE = 0.7
# WORLD_SEARCH_HITS = (1, 6): a search or listing returns this many records,
# inclusive. Measured: a fixed 2 to 3 taught models that searches always
# return two or three results.
WORLD_SEARCH_HITS = (1, 6)
# WORLD_TEMPLATE_HITS = (1, 5): records a model-written list template
# expands to, inclusive. Convention, untested; same reasoning as search hits.
WORLD_TEMPLATE_HITS = (1, 5)
# WORLD_ID_RANGE = (1000, 90000): generated record ids, lower inclusive,
# upper exclusive. Four to five digits reads as an id and never collides
# with the ``<stem>_<hex>`` ids the world issues on create. Convention.
WORLD_ID_RANGE = (1000, 90000)
# WORLD_DATE_YEARS = (2022, 5): generated dates fall in these calendar
# years (first year, span). Convention: recent enough to read as live data,
# fixed so seeded rows do not drift with the clock.
WORLD_DATE_YEARS = (2022, 5)
# WORLD_JITTER_DIVISOR = 3: a number in a model-written result template
# moves by up to one part in this many of itself (900 lands in 600 to
# 1200). Wide enough that a policy threshold near the template value is
# crossed both ways; convention, untested. Documented on local_model().
WORLD_JITTER_DIVISOR = 3
# WORLD_CREATED_ID_MODULUS = 100000: issued ids on create are
# ``<stem>_<n mod this>``. Five digits; convention.
WORLD_CREATED_ID_MODULUS = 100_000
# WORLD_ISSUED_ID_HEX = 12: hex characters in a world-issued id and in the
# per-call digest. 48 bits: no collision in any run this package makes.
WORLD_ISSUED_ID_HEX = 12
# WORLD_REF_CHARS = 8: characters of the call digest echoed as ``ref``,
# enough to tell two calls apart in a row and short enough to read.
# (convention, untested)
WORLD_REF_CHARS = 8
# WORLD_EXPRESSION_CHARS = 200: a calculator expression is echoed back cut
# to this many characters, so a runaway argument cannot bloat a row.
WORLD_EXPRESSION_CHARS = 200
# WORLD_HINT_CHARS = 40: characters of a caller's query kept as the name
# stem of generated records. Long enough for a product name; convention.
WORLD_HINT_CHARS = 40
# WORLD_SHELL_FLAVORS = 11: one shell call in this many is a permission
# error, one a failing test run, one a merge conflict, one an ``ls``; the
# rest pass. About a third of shell calls therefore fail. No benchmark
# reports a per-call shell failure rate; tau-bench's task-level numbers
# (gpt-4o pass^1 61.2 on retail and 35.2 on airline, arXiv:2406.12045
# Table 2) say only that failure is common. Convention.
WORLD_SHELL_FLAVORS = 11
# WORLD_CI_FAIL_ONE_IN = 7: one CI listing in this many carries a failed
# check. Convention, untested.
WORLD_CI_FAIL_ONE_IN = 7

# ---------------------------------------------------------------------
# ingest (traces)
# ---------------------------------------------------------------------
#
# Reachable through keywords on the ingest helpers (``mine_result_exemplars``,
# ``leakage_report``, ``split_pseudo_production``, ``behavior_state``,
# ``dimensions_from_traces``).

# TRACE_EXEMPLARS_PER_TOOL = 3: real result payloads kept per tool as shape
# templates. Three shows the shape family (record, list, text) without
# repeating it; convention, untested.
TRACE_EXEMPLARS_PER_TOOL = 3
# TRACE_EXEMPLAR_MAX_CHARS = 500: serialized size cap on one exemplar, about
# 125 tokens at 4 characters a token, so three per tool stay under a few
# hundred tokens of the writer prompt. Token-budget reasoning; the exact
# number is convention.
TRACE_EXEMPLAR_MAX_CHARS = 500
# TRACE_EXEMPLAR_STRING_CHARS = 160: a string inside an exemplar is cut here
# with an ellipsis. TRACE_EXEMPLAR_LIST_ITEMS = 2 / TRACE_EXEMPLAR_DICT_KEYS
# = 12: bound the nesting so the 500-char cap is reachable by trimming,
# not by dropping the whole payload. (convention, untested)
TRACE_EXEMPLAR_STRING_CHARS = 160
TRACE_EXEMPLAR_LIST_ITEMS = 2
TRACE_EXEMPLAR_DICT_KEYS = 12
# TRACE_LEAK_THRESHOLD = 0.9: cosine similarity at or above which a
# generated prompt is a near copy of a source trace. Exact matches always
# flag. Lambert 2025, chapter Evaluation ("Contamination") uses 8-gram
# overlap for exact leakage; 0.9 on hashed n-gram vectors is the same test
# with a small paraphrase allowance. Convention for the exact number.
TRACE_LEAK_THRESHOLD = 0.9
# TRACE_LEAK_EXAMPLES = 20: offenders listed in a leakage report; the count
# is always complete. Report size, convention.
TRACE_LEAK_EXAMPLES = 20
# TRACE_PSEUDO_PRODUCTION_FRACTION = 0.2: rows set aside as pseudo
# production. The usual 80/20 train/holdout split; convention.
TRACE_PSEUDO_PRODUCTION_FRACTION = 0.2
# TRACE_MIN_SUPPORT = 3: graded rows a behavior region needs before its
# allocation is more than a hint. Statistical reason: the Clopper-Pearson
# 95% upper bound on a 0-of-n fail rate is 0.95 at n=1, 0.78 at n=2 and
# 0.63 at n=3, so three is the smallest count at which "never failed" rules
# out a majority fail rate.
TRACE_MIN_SUPPORT = 3
# TRACE_EXPLORATION_FLOOR = 0.2 and TRACE_EXPLORATION_MAX = 0.6: share of
# the budget always kept for broad coverage (floor) and the most a caller
# can reserve (max). An allocation that follows observed failures alone
# never finds a new one; 20% is an epsilon-greedy convention, untested.
TRACE_EXPLORATION_FLOOR = 0.2
TRACE_EXPLORATION_MAX = 0.6
# TRACE_STATE_PRIORITY = {new: 1.0, ...}: status weight in the region
# priority. New and persistent failures draw first; a solved region keeps a small weight so
# it is re-checked. Convention, untested.
TRACE_STATE_PRIORITY: Mapping[str, float] = MappingProxyType(
    {
        "new": 1.0,
        "persistent": 0.9,
        "uncertain": 0.35,
        "improving": 0.2,
        "solved": 0.05,
        "passing": 0.05,
    }
)
# TRACE_SUPPORT_SATURATION = 6: failures at which the support factor
# reaches 1.0 (it starts at 0.5 with none). Convention, untested.
TRACE_SUPPORT_SATURATION = 6
# TRACE_RATE_FLOOR = 0.25: the rate factor is floor + (1 - floor) * fail
# rate, so a region that never fails still draws a quarter of the weight a
# region that always fails does. Convention, untested.
TRACE_RATE_FLOOR = 0.25
# TRACE_REPORT_LIST_CAP = 5: names shown per axis in the text report,
# enough to see the leaders without the report becoming the list.
# (convention, untested)
TRACE_REPORT_LIST_CAP = 5
# TRACE_TASK_HASH_CHARS = 200: prompt characters in the split hash; two
# prompts that agree on their first 200 characters are one task for the
# train/pseudo-production split. (convention, untested)
TRACE_TASK_HASH_CHARS = 200

# ---------------------------------------------------------------------
# connect (platform)
# ---------------------------------------------------------------------
#
# Reachable through ``timeout=`` / ``poll=`` keywords on the platform calls.

# PLATFORM_REQUEST_TIMEOUT_S = 120: one gate request. A finalize on a large
# set takes tens of seconds; two minutes is convention, untested.
PLATFORM_REQUEST_TIMEOUT_S = 120
# PLATFORM_UPLOAD_TIMEOUT_S = 300: the studio import, which grades every row
# server side before answering. Convention, sized to a 20k-row push.
PLATFORM_UPLOAD_TIMEOUT_S = 300
# PLATFORM_PUT_TIMEOUT_S = 120 / PLATFORM_PUT_S_PER_MB = 4: the presigned S3
# PUT of a pushed JSONL set may take two minutes plus four seconds per
# megabyte (a 117 MB eval set of 4k-token rollouts gets about ten minutes;
# it timed out at the flat cap, #386). Four seconds a megabyte is a 2 Mbit/s
# floor, the slow end of a home uplink; ``timeout=`` on ``push_rows``
# overrides. Convention.
PLATFORM_PUT_TIMEOUT_S = 120
PLATFORM_PUT_S_PER_MB = 4
# PLATFORM_CREDENTIAL_TTL_S = 3600: default life of a delegated credential.
# One hour matches the Clerk session token that mints it.
PLATFORM_CREDENTIAL_TTL_S = 3600
# PLATFORM_ERROR_DETAIL_CHARS = 400: gate error body echoed in the
# exception. Enough for the gate's one-line reason; convention.
PLATFORM_ERROR_DETAIL_CHARS = 400
# PLATFORM_STUDIO_MAX_ROWS = 20000: the studio import endpoint's cap; a
# larger push is refused client side with the number. (the gate's limit)
PLATFORM_STUDIO_MAX_ROWS = 20_000
# PLATFORM_HF_DATASET_TIMEOUT_S = 600 / PLATFORM_HF_MODEL_TIMEOUT_S = 900:
# how long ``hf_publish`` / ``hf_publish_run`` wait for the Hugging Face
# push; an adapter upload is bigger than a JSONL set. Convention.
PLATFORM_HF_DATASET_TIMEOUT_S = 600.0
PLATFORM_HF_MODEL_TIMEOUT_S = 900.0
# PLATFORM_HF_POLL_S = 2.5 / PLATFORM_HF_MODEL_POLL_S = 3.0 /
# PLATFORM_IMPORT_POLL_S = 3.0: seconds between status reads while waiting.
PLATFORM_HF_POLL_S = 2.5
PLATFORM_HF_MODEL_POLL_S = 3.0
PLATFORM_IMPORT_POLL_S = 3.0
# PLATFORM_IMPORT_TIMEOUT_S = 900 / PLATFORM_IMPORT_MAX_ROWS = 100000: an
# import of a Hugging Face split. The row cap is the gate's.
PLATFORM_IMPORT_TIMEOUT_S = 900.0
PLATFORM_IMPORT_MAX_ROWS = 100_000
# PLATFORM_TRACE_PAGE_SIZE = 200: traces read per page when listing an
# agent's traces (the ``limit=`` the traces route takes). Convention.
PLATFORM_TRACE_PAGE_SIZE = 200
# PLATFORM_HOLDOUT_PROVE_EFFECT = PROVE_EFFECT: the gain a pushed holdout
# is sized to prove at 80% power (holdout_size); the same quantity
# ``eval_power`` sizes for, under the platform name, so the two cannot
# drift. One value, one home (PROVE_EFFECT, score: statistics).
PLATFORM_HOLDOUT_PROVE_EFFECT = PROVE_EFFECT
# PLATFORM_REWARD_MODEL_BATCH = 256: rows per scoring request to a hosted
# reward model; the gate's request cap, so a batch never 413s. (the gate's
# limit)
PLATFORM_REWARD_MODEL_BATCH = 256
# PLATFORM_UNKNOWN_IDS_SHOWN = 3: unknown trace ids named in an error
# before ", ..."; enough to spot a typo pattern. (convention, untested)
PLATFORM_UNKNOWN_IDS_SHOWN = 3

# ---------------------------------------------------------------------
# training
# ---------------------------------------------------------------------
#
# Reachable through ``training_run(flush_every=, flush_seconds=)``,
# ``TrainingRun(max_batch=)``, ``train(...)`` keywords and ``run.wait(poll=)``.
# The trainer's own defaults live on the platform; what the SDK holds is
# the range each knob is accepted in and the value the literature reaches
# for, so the error message can name a fix.

# TRAINING_FLUSH_EVERY = 25 points / TRAINING_FLUSH_SECONDS = 15: when the
# log buffer is sent. One request per 25 steps keeps a 1k-step run under
# 50 calls; 15 s keeps a slow run's curve live. Convention.
TRAINING_FLUSH_EVERY = 25
TRAINING_FLUSH_SECONDS = 15.0
# TRAINING_MAX_BATCH = 500: points per log request. Keeps one request
# well under the gate's body limit; convention.
TRAINING_MAX_BATCH = 500
# TRAINING_POLL_S = 15 / TRAINING_POLL_MIN_S = 1: seconds between reads of
# a hosted run's state, and the floor so a caller cannot hammer the gate.
TRAINING_POLL_S = 15.0
TRAINING_POLL_MIN_S = 1.0
# TRAINING_ERROR_CHARS = 2000: a finish error is cut here; the run page
# shows one screen of it.
TRAINING_ERROR_CHARS = 2000
# GPU_USD_PER_HOUR = {A10G 1.10, L40S 1.95, H100 3.95, A100 2.50, T4 0.59}:
# what one GPU-hour of a hosted training run costs, as an estimate. The
# platform's trainer runs on Modal, so Modal's on-demand list price is the
# honest rate. Modal quotes per second (A10 $0.000306, L40S $0.000542,
# H100 SXM5 $0.001097, A100 80 GB $0.000694, T4 $0.000164); the table is
# those rates times 3600, to the cent. A run's ``cost_usd`` is
# ``seconds / SECONDS_PER_HOUR * rate``. Rollouts and judge calls on the
# shared serving endpoint are not priced. (https://modal.com/pricing, read
# 2026-09-20)
GPU_USD_PER_HOUR: Mapping[str, float] = MappingProxyType(
    {"A10G": 1.10, "L40S": 1.95, "H100": 3.95, "A100": 2.50, "T4": 0.59}
)
# GPU_PRICE_SOURCE = "modal.com/pricing 2026-09-20": the page and the day
# GPU_USD_PER_HOUR was read from, quoted in every ``cost_basis`` so the
# reader can check the rate behind the number. (https://modal.com/pricing,
# read 2026-09-20)
GPU_PRICE_SOURCE = "modal.com/pricing 2026-09-20"
# SECONDS_PER_HOUR = 3600: the unit conversion between a run's ``seconds``
# and the per-hour rate. (definition)
SECONDS_PER_HOUR = 3600

# TRAINING_KNOBS = {knob: {lo, hi, ref, why}}: accepted range and reference
# value per knob, by method. ``lo``/``hi`` are what ``train`` accepts; ``ref``
# is what the cited source used, so the message that rejects a value can
# say what to reach for.
#
# generations: GRPO group size G. DAPO trains at G=16 (arXiv:2503.14476,
#   Table of hyperparameters), Dr. GRPO at 8 (arXiv:2503.20783, Table 6),
#   ProRL at 16 (arXiv:2505.24864). 2 is the structural minimum (one
#   advantage needs a pair); 32 is the hosted trainer's memory cap.
# learning_rate: RL runs at 1e-6 (DAPO, Dr. GRPO) to 2e-6 (ProRL); SFT one
#   to two orders below pretraining: 1e-5 (OLMo 2) to 5e-5 to 8e-5 (OLMo 3)
#   full fine-tune (Lambert 2025, chapter Instruction Tuning,
#   "Implementation Details"; the range stitches two models' settings, it
#   is not one recipe), 2e-4 for a LoRA adapter (LoRA arXiv:2106.09685
#   tunes at a higher rate than full fine-tuning). DPO wants "surprisingly low learning rates"
#   (Lambert 2025, chapter Direct Alignment); the DPO paper used 1e-6
#   (arXiv:2305.18290, App. B).
# beta: the KL coefficient. DAPO, Dr. GRPO and CISPO drop it (0.0;
#   arXiv:2503.14476, 2503.20783, 2506.13585); ProRL keeps it with
#   reference resets (arXiv:2505.24864). DPO's beta is the preference
#   temperature, 0.1 in the paper (arXiv:2305.18290).
# max_completion_length: DAPO's soft overlong window is 4096 tokens under a
#   16384 cap (arXiv:2503.14476); the hosted trainer serves up to 4096.
# temperature: ProRL samples at 1.2 (arXiv:2505.24864); rejection sampling
#   runs 0.7 to 1.0 (Lambert 2025, chapter Rejection Sampling).
# clip (host key epsilonHigh): PPO/GRPO clip 0.2
#   (Lambert 2025, chapter Reinforcement Learning, shows ``eps = 0.2`` only as
#   an example in its code listing, not as a recommendation); DAPO
#   clip-higher 0.28, ProRL 0.4.
# Read-only: the table and every row are MappingProxyType, so a caller
# cannot widen a range by mutating it.
_TRAINING_KNOB_TABLE: dict[str, dict[str, Any]] = {
    # lo/hi: accepted range; open_lo/open_hi: the endpoint itself is
    # rejected; range: the words the error uses; ref: the cited value.
    "generations": {
        "lo": 2,
        "hi": 32,
        "open_lo": False,
        "open_hi": False,
        "range": "2 to 32 rollouts per prompt",
        "ref": 8,
        "methods": ("grpo",),
        "why": "the GRPO group size; Dr. GRPO 8, DAPO and ProRL 16 "
        "(arXiv:2503.20783, 2503.14476, 2505.24864)",
    },
    "learning_rate": {
        "lo": 0.0,
        "hi": 1.0,
        "open_lo": True,
        "open_hi": True,
        "range": "a positive step below 1",
        "ref": {"sft": 2e-4, "grpo": 1e-6, "dpo": 1e-6, "rm": 1e-5},
        "methods": ("sft", "grpo", "dpo", "rm"),
        "why": "the optimizer step; 1e-6 for RL (DAPO, Dr. GRPO), 1e-6 for DPO "
        "(arXiv:2305.18290), 2e-4 for a LoRA SFT adapter",
    },
    "beta": {
        "lo": 0.0,
        "hi": None,
        "open_lo": False,
        "open_hi": False,
        "range": "0 or more",
        "ref": {"grpo": 0.0, "dpo": 0.1},
        "methods": ("grpo", "dpo"),
        "why": "the KL coefficient; 0 in DAPO, Dr. GRPO and CISPO, 0.1 as the DPO "
        "preference temperature (arXiv:2305.18290)",
    },
    "max_completion_length": {
        "lo": 16,
        "hi": 4096,
        "open_lo": False,
        "open_hi": False,
        "range": "16 to 4096 tokens",
        "ref": 4096,
        "methods": ("grpo", "dpo"),
        "why": "the token cap on a sampled reply; DAPO's soft overlong window is 4096 "
        "(arXiv:2503.14476)",
    },
    "temperature": {
        "lo": 0.0,
        "hi": 2.0,
        "open_lo": True,
        "open_hi": False,
        "range": "above 0 and at most 2",
        "ref": 1.0,
        "methods": ("grpo",),
        "why": "the GRPO rollout temperature; ProRL 1.2, rejection sampling 0.7 to 1.0",
    },
    "clip": {
        "lo": 0.0,
        "hi": 1.0,
        "open_lo": False,
        "open_hi": False,
        "range": "0 to 1",
        "ref": 0.28,
        "methods": ("grpo",),
        "why": "the upper clip (host key epsilonHigh); 0.2 PPO, 0.28 DAPO clip-higher, 0.4 ProRL",
    },
}
TRAINING_KNOBS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {name: MappingProxyType(row) for name, row in _TRAINING_KNOB_TABLE.items()}
)
# TRAINING_LORA_RANK = 16 / TRAINING_LORA_ALPHA = 32: reference adapter
# shape. LoRA sets alpha to the first r tried and does not tune it
# (arXiv:2106.09685 §4.1) and finds r=4 to 8 on the attention projections
# enough (§7.1); alpha=2r is the PEFT convention. Advisory: the hosted
# trainer owns its own values.
TRAINING_LORA_RANK = 16
TRAINING_LORA_ALPHA = 32
# TRAINING_BATCH_PROMPTS = 256: reference prompt batch. OLMo 2 post-trains
# at 256 prompts (Lambert 2025, chapter Instruction Tuning); ProRL at 256
# (arXiv:2505.24864); DAPO at 512. Advisory.
TRAINING_BATCH_PROMPTS = 256
# TRAINING_SFT_EPOCHS = 2: reference SFT epochs; Lambert 2025 gives no number
# and the hosted trainer owns its own. Convention, untested.
TRAINING_SFT_EPOCHS = 2
# TRAIN_MIN_MIXED_TASKS = 32: tasks with both a pass and a fail a grouped
# or paired method (grpo, dpo, rm) should have before ``train`` starts a
# hosted run without a warning. A unanimous group carries no advantage
# (GRPO's baseline is the group mean, Shao et al. 2024, arXiv:2402.03300),
# so DAPO (arXiv 2503.14476, eq. 11) and ProRL (arXiv 2505.24864) keep
# only mixed prompts, and they draw them from tens of thousands; the
# hosted trainer steps one prompt group at a time, so under this count a
# default run is several passes over a handful of groups (#397: 6 mixed
# tasks, 20 steps, 3.3 epochs, grad_norm 0). 32 is the same count as
# MONITOR_N_PROMPTS, the fewest prompts the package reads a curve on;
# ``train(min_mixed_tasks=)`` moves it. (convention, untested)
TRAIN_MIN_MIXED_TASKS = 32

# ---------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------
#
# Reachable through ``HackMonitor(...)`` keywords.

# MONITOR_N_PROMPTS = 32 / MONITOR_K = ROLLOUTS_PER_TASK: holdout asks
# sampled per eval and completions per ask (4, the same k the sizing and
# pass^k rest on, so 128 rollouts). At p=0.5 a 32-task bootstrap band is
# about +-0.17, wide enough that only large proxy/gold divergence reads;
# the size is a cost choice, and ``n_prompts=`` raises it. Convention.
MONITOR_N_PROMPTS = 32
MONITOR_K = ROLLOUTS_PER_TASK
# MONITOR_EVERY = 10: trainer steps between evals. Convention.
MONITOR_EVERY = 10
# MONITOR_WINDOW = 3: evals the alarms look back over; three is the fewest
# that separate a trend from one noisy eval. (convention, untested)
MONITOR_WINDOW = 3
# MONITOR_DELTA = 0.1: proxy gain over the window that counts as climbing.
# Ten points is well outside the 0.25 to 1.5 point eval noise
# Lambert 2025, chapter Evaluation, reports; convention for the number.
MONITOR_DELTA = 0.1
# MONITOR_LENGTH_PCT = 0.25: completion-length growth that counts.
# Lambert 2025, chapter Over-optimization, lists the qualitative
# signatures (stock phrases, hedging and repetition, sycophancy,
# over-refusal) and does not name length; length growth is the bias Dr.
# GRPO removes from the GRPO objective (arXiv:2503.20783), which is why
# the monitor reads it. A quarter is convention, untested.
MONITOR_LENGTH_PCT = 0.25
# MONITOR_BUFFER = 512: completions the reward wrapper keeps for the
# feature scan. MONITOR_SCAN_MIN = 8: the fewest it scans, two groups of
# MONITOR_K, so a correlation has something to correlate. (convention, untested)
MONITOR_BUFFER = 512
MONITOR_SCAN_MIN = 8
# MONITOR_SCAN_PERMUTATIONS = 50 / MONITOR_WINDOW_BOOTSTRAPS = 500: the
# permutation count for hack_scan's feature test and the bootstrap count
# for the gold-vs-window interval. 500 resamples give a 95% band to about
# two decimals; 50 permutations a p-value to 0.02. Convention.
MONITOR_SCAN_PERMUTATIONS = 50
MONITOR_WINDOW_BOOTSTRAPS = 500
# MONITOR_MAX_NEW_TOKENS = 256: completion length sampled on the holdout;
# the same floor as MIN_REPLY_TOKENS, so a holdout reply is never a
# fragment. (convention, untested)
MONITOR_MAX_NEW_TOKENS = 256
# MONITOR_CONCURRENCY = 8: gold judge calls in flight; a quarter of the
# rollout concurrency because the eval runs beside training. (convention,
# untested)
MONITOR_CONCURRENCY = 8
# MONITOR_SAMPLE_TEMPERATURE = LOCAL_MODEL_TEMPERATURE: the default sampler
# draws at the temperature the rollout engine uses (one value, one home) so
# the holdout is sampled the way the data was.
# MONITOR_SAMPLE_TOP_P = 0.95 / MONITOR_SAMPLE_BATCH = 16: its nucleus cut
# and prompts per generate call (convention, untested).
MONITOR_SAMPLE_TEMPERATURE = LOCAL_MODEL_TEMPERATURE
MONITOR_SAMPLE_TOP_P = 0.95
MONITOR_SAMPLE_BATCH = 16

# ---------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------
#
# Reachable through ``build_tasks`` / ``export_environment`` keywords.

# The solve-rate band (0.2, 0.8) an environment keeps prompts in is
# DIFFICULTY_BAND above (one home); score/optimize.py and environment.py
# both alias it as DEFAULT_BAND. DAPO's dynamic sampling drops prompts at
# accuracy 0 and 1 (arXiv:2503.14476); the 20 to 80 band is the same rule
# with a margin for k=8 noise.
# ENV_HOLDOUT_FRACTION = TRACE_PSEUDO_PRODUCTION_FRACTION: tasks held out,
# the same 80/20 split the trace lane uses (one value, one home).
ENV_HOLDOUT_FRACTION = TRACE_PSEUDO_PRODUCTION_FRACTION
# ENV_DECONTAMINATION_NGRAM = DECONTAM_NGRAM: n-gram size for
# train-vs-holdout overlap, the same 8-gram test as the decontamination
# in score/ (one value, one home).
ENV_DECONTAMINATION_NGRAM = DECONTAM_NGRAM
# ENV_DECONTAMINATION_EXAMPLES = 3: overlaps shown in the report before
# ", ..."; enough to see what kind of text leaked. (convention, untested)
ENV_DECONTAMINATION_EXAMPLES = 3
# ENV_MAX_TURNS_FALLBACK = 10: turn cap when a spec carries none; under
# DEFAULT_AVG_TURNS because an exported task carries no simulated user
# and the cap only stops a runaway loop. (convention, untested)
ENV_MAX_TURNS_FALLBACK = 10
# ENV_EVAL_EXAMPLES = 5 / ENV_EVAL_ROLLOUTS = 3: the verifiers smoke eval
# written into the exported pyproject: five tasks, three rollouts each,
# enough to prove the package installs and grades, not to measure it.
# (convention, untested)
ENV_EVAL_EXAMPLES = 5
ENV_EVAL_ROLLOUTS = 3
# ENV_HARNESS_MIX = "uniform": how load_environment spreads a spec's
# harnesses over its tasks; each task draws one with equal weight, or with
# the weights an explicit list gives. Kim et al. 2026 (arXiv:2606.25447)
# train across harnesses and report out-of-distribution gains, and do not
# compare mixing weights, so equal weight is (convention, untested).
ENV_HARNESS_MIX = "uniform"
# ENV_HARNESS_SEED = DEFAULT_SEED: the seed hashed with the task id to pick
# its harness, so a re-run of the environment draws the same one per task
# (one value, one home).
ENV_HARNESS_SEED = DEFAULT_SEED

# ---------------------------------------------------------------------
# text heuristics
# ---------------------------------------------------------------------
#
# TEXT_HEURISTICS = TextHeuristics(): the character and word counts the
# English lexicon gates in generate/ and score/ turn on ("a clause is
# eight characters or more", "a sign-off is six words or fewer"). They
# describe the hosted writer's habits and the package's own English
# scrubbers, not the customer's domain; every one is a convention,
# untested, and named here so none is a bare number at its call site.
# A run in another language sets ``user_model=`` to a writer that does
# not produce the habits these gates catch.


@dataclass(frozen=True)
class TextHeuristics:
    """The text-gate thresholds, one field each, with what each decides.

    Frozen: a caller who wants another value builds a new instance and
    passes it to the function that takes ``heuristics=``; nothing reads
    these through ambient state.
    """

    # --- generate/agents.py: the simulated user's lines ------------------
    #: a tool parameter label longer than this is a sentence, not a hint
    detail_label_max_chars: int = 24
    #: a polite user line longer than this is filler the persona drops
    polite_filler_min_chars: int = 40
    #: this many double quotes in a user line reads as pasted code or JSON
    code_quote_marks: int = 4
    #: a bare yes/no answer to a confirm question is at most this long
    confirm_reply_max_chars: int = 60
    #: a reply that is only an identifier still needs this many characters
    id_reply_min_chars: int = 3
    #: a follow-up shorter than this is a fragment, not a turn
    followup_min_chars: int = 8
    #: a user line with fewer distinct words than this cannot repeat one
    history_min_words: int = 5
    #: a user line with fewer letter words than this cannot echo the agent
    echo_min_words: int = 4
    # --- generate/explore.py: mutations of an ask --------------------------
    #: an ask needs this many words before one can be deleted
    mutation_min_words: int = 4
    #: an ask needs this many words before two can be swapped
    swap_min_words: int = 3
    #: only words with this many letters are swapped (not "a", "the")
    swap_min_word_chars: int = 5
    # --- generate/generator.py: the situation writer's output -------------
    #: a policy or brief sentence shorter than this is a heading, not a rule
    sentence_min_chars: int = 8
    #: a brief chunk this long that recurs in a card is a copied brief
    repeated_chunk_chars: int = 28
    #: a word needs this many letters before a typo is planted in it
    word_min_letters: int = 5
    #: what "short" means for a user opener, in words, inclusive
    short_words: tuple[int, int] = (3, 16)
    #: what "long" means for a user opener: at least this many words
    long_min_words: int = 40
    #: a "curt" opener over this many words is not curt
    curt_max_words: int = 22
    #: a policy clause shorter than this is a label, not a rule
    policy_clause_min_chars: int = 24
    #: a usable situation card is this many characters, inclusive
    card_chars: tuple[int, int] = (8, 300)
    # --- generate/scenarios.py, generate/diversity.py ---------------------
    #: a policy clause shorter than this is dropped when a rule is split
    clause_min_chars: int = 8
    #: a stem marker shorter than this matches too many words by prefix
    stem_marker_min_chars: int = 4
    # --- score/: grading, checklist, preflight --------------------------
    #: a last line of at most this many words can be a sign-off
    sign_off_max_words: int = 6
    #: a short capitalized last line under this many words is a sign-off
    sign_off_short_words: int = 5
    #: an identifier in a reply needs this many digits to be checked
    id_min_digits: int = 3
    #: a reference value shorter than this is a word, not an identifier
    reference_min_chars: int = 3
    #: an agent utterance this long counts toward the answer's size
    sizable_utterance_chars: int = 24
    #: a word this long that ends in "s" is stemmed to its singular
    plural_min_chars: int = 5
    #: a coverage-gap word shorter than this is a stop word
    gap_word_min_chars: int = 3
    #: a tool-name word shorter than this ("to", "by") is not matched to an ask
    tool_word_min_chars: int = 3
    # --- verify/, run/spec.py, world/sandbox.py ---------------------------
    #: a gold line shorter than this is not matched against a traceback
    gold_line_min_chars: int = 8
    #: a spec argument without spaces and under this long is a slug
    slug_max_chars: int = 64
    #: a domain noun this long that ends in "s" (not "ss") is singularized
    plural_noun_min_chars: int = 4


TEXT_HEURISTICS = TextHeuristics()


def knob(default: _T, *, lo: float | None = None, hi: float | None = None) -> _T:
    """A :class:`RunKnobs` field with its bounds. ``lo`` and ``hi`` are
    inclusive; ``None`` is open. Typed as the default's own type so the
    dataclass field reads as ``int`` or ``float`` to a checker."""
    return field(default=default, metadata={"lo": lo, "hi": hi})


@dataclass(frozen=True)
class RunKnobs:
    """The engine's tunables, one field per ``advanced={...}`` key.

    Every field's default is the value the engine carried inline before
    it was named; none changed. The comment above each says why that
    value, or says honestly that nobody has tested another.
    """

    # --- stopping and restarting ------------------------------------

    # empty_rounds_to_stop = 8: consecutive scheduler rounds with nothing
    # to run and nothing in flight before a run with a model writer and
    # an empty pool gives up as a writer failure. (convention, untested)
    empty_rounds_to_stop: int = knob(8, lo=1)
    # writer_idle_rounds_to_restart = 4: rounds the pool has stayed empty
    # with writers returning only duplicates before the writer is
    # restarted with a rotated seed and an avoid window. (convention,
    # untested)
    writer_idle_rounds_to_restart: int = knob(4, lo=1)
    # writer_idle_rounds_to_rest = 2: idle rounds after which a plain
    # (not unique, no clock) run stops launching new waves and lets the
    # restart rule decide. (convention, untested)
    writer_idle_rounds_to_rest: int = knob(2, lo=1)
    # restart_avoid_window = 8: used asks handed to the restarted writer
    # as avoid pressure per restart; reseeding alone reconverged to the
    # same asks (measured: 26 distinct asks vs the old ceiling of 28).
    # The window size itself is a convention, untested.
    restart_avoid_window: int = knob(8, lo=1)
    # rows_per_extra_restart = 100: the writer restart allowance grows by
    # one per this many budgeted rows, above MAX_NOVELTY_RESTARTS; a
    # 10k-row budget cannot live on a smoke run's retries. (convention,
    # untested)
    rows_per_extra_restart: int = knob(100, lo=1)
    # dead_agent_errors = 16: see DEAD_AGENT_MIN_ERRORS.
    dead_agent_errors: int = knob(DEAD_AGENT_MIN_ERRORS, lo=1)
    # dead_agent_budget_multiple = 2: see DEAD_AGENT_BUDGET_MULTIPLE.
    dead_agent_budget_multiple: int = knob(DEAD_AGENT_BUDGET_MULTIPLE, lo=0)
    # closing_margin = 2.0: under a clock, stop opening new groups when
    # the time left is under this many median rollout durations, so a
    # group is never cut mid-group. Two: one for the rollouts in flight,
    # one for the group's own. (convention, untested)
    closing_margin: float = knob(2.0, lo=0.0)
    # closing_window_rollouts = 20: recent rollout durations the median
    # is read from. (convention, untested)
    closing_window_rollouts: int = knob(20, lo=1)

    # --- scheduler timing -------------------------------------------

    # collect_wait_s = 0.35: how long a round waits for a rollout or a
    # verdict to land before re-planning; the loop's tick. Shorter spins
    # the CPU, longer delays refills. (convention, untested)
    collect_wait_s: float = knob(0.35, lo=0.0)
    # collect_wait_floor_s = 0.1: the shortest tick when the clock is
    # nearly out. (convention, untested)
    collect_wait_floor_s: float = knob(0.1, lo=0.0)
    # writer_wait_s = 0.5: how long the loop waits on a writer wave when
    # the pool is empty and nothing is in flight. (convention, untested)
    writer_wait_s: float = knob(0.5, lo=0.0)
    # writer_wait_floor_s = 0.2: the shortest writer or scene wait when
    # the clock is nearly out. (convention, untested)
    writer_wait_floor_s: float = knob(0.2, lo=0.0)
    # hosted_touch_s = 5.0: timeout on the warm-up ping sent to the hosted
    # writer before the first wave. (convention, untested)
    hosted_touch_s: float = knob(5.0, lo=0.0)
    # scene_join_s = 8.0: how long shutdown waits for the scene-brief
    # thread with no clock; scene_join_clocked_s = 1.0 under a clock.
    # (convention, untested)
    scene_join_s: float = knob(8.0, lo=0.0)
    scene_join_clocked_s: float = knob(1.0, lo=0.0)

    # --- writer pipeline --------------------------------------------

    # writer_buffer_waves = 2: keep about this many waves of prompts in
    # the pipe ahead of the rollouts. (convention, untested)
    writer_buffer_waves: int = knob(2, lo=1)
    # writer_buffer_cap = 96: the most prompts the buffer plans for,
    # whatever the concurrency. (convention, untested)
    writer_buffer_cap: int = knob(96, lo=1)
    # writer_typical_completions = 3: completions per card the buffer
    # math assumes, whatever completions_per_request asks for.
    # (convention, untested)
    writer_typical_completions: int = knob(3, lo=1)
    # pool_low_floor = 16, pool_low_cap = 64, pool_low_flight_divisor = 4:
    # the pool is "low" (refill now) under max(floor, min(cap, flight /
    # divisor)) eligible prompts. (convention, untested)
    pool_low_floor: int = knob(16, lo=0)
    pool_low_cap: int = knob(64, lo=0)
    pool_low_flight_divisor: int = knob(4, lo=1)
    # offline_topup_multiple = 2: the template writer tops up when fewer
    # than this many batches of prompts are eligible. (convention,
    # untested)
    offline_topup_multiple: int = knob(2, lo=1)
    # offline_bounce_limit = 20: extra template rounds tried before a
    # short batch is accepted. (convention, untested)
    offline_bounce_limit: int = knob(20, lo=0)
    # offline_bounce_stride = 17: the seed step between those rounds; any
    # stride coprime with the round count does. (convention, untested)
    offline_bounce_stride: int = knob(17, lo=1)
    # restart_seed_stride = 997: the round id advances by this per writer
    # restart so restarted draws never reuse a round's temperature and
    # tags. Any large prime does. (convention, untested)
    restart_seed_stride: int = knob(997, lo=1)
    # first_wave_writers = 2, first_wave_cards = 4, first_wave_tokens =
    # 320: the first waves are tiny so the first rollouts start in about
    # five seconds instead of after a full wave. (convention, untested)
    first_wave_writers: int = knob(2, lo=1)
    first_wave_cards: int = knob(4, lo=1)
    first_wave_tokens: int = knob(320, lo=1)
    # min_cards_per_wave = 2: a wave asks for at least two cards; one
    # card gives the writer nothing to contrast. (convention, untested)
    min_cards_per_wave: int = knob(2, lo=1)
    # tokens_per_card = 130, wave_tokens_base = 128, wave_tokens_floor =
    # 256, wave_tokens_cap = 2048: a wave's reply budget is clamp(130 x
    # cards + 128, 256, 2048), about 130 tokens per card so long-prompt
    # cards are realizable; the old 768 ceiling gave 12-card batches 64
    # tokens per message. (observed on the hosted writer; the per-card
    # figure is a convention, untested)
    tokens_per_card: int = knob(130, lo=1)
    wave_tokens_base: int = knob(128, lo=0)
    wave_tokens_floor: int = knob(256, lo=1)
    wave_tokens_cap: int = knob(2048, lo=1)
    # failing_seeds_cap = 40: failing asks mined from traces= that seed
    # the run; more than this crowds out the grid. (convention, untested)
    failing_seeds_cap: int = knob(40, lo=0)

    # --- selection --------------------------------------------------

    # select_oversample = 3: the diversity selector picks this many times
    # the batch so the family cap and the situation quota have slack.
    # (convention, untested)
    select_oversample: int = knob(3, lo=1)
    # family_cap_floor = 16, family_cap_per_phrasing = 4: near-copy
    # scenario families are capped at max(floor, n x per_phrasing) rows.
    # (convention, untested)
    family_cap_floor: int = knob(16, lo=1)
    family_cap_per_phrasing: int = knob(4, lo=1)
    # writer_context_items = 8: items of each kind (avoid, underexplored,
    # behavior gaps, axis gaps, tools) the writer prompt carries; more
    # made the card prompt longer than the cards. (convention, untested)
    writer_context_items: int = knob(8, lo=0)
    # writer_context_parents = 10: failing rows the writer mutates from.
    # (convention, untested)
    writer_context_parents: int = knob(10, lo=0)
    # family_avoid_items = 6: family-rejected prompts shown to the writer
    # as avoid pressure, before the concentrated ones. (convention,
    # untested)
    family_avoid_items: int = knob(6, lo=0)

    # --- search heuristics ------------------------------------------

    # allocation_gain = 4.0: see ALLOC_GAIN.
    allocation_gain: float = knob(ALLOC_GAIN, lo=0.0)
    # allocation_tool_weight = 0.6, allocation_condition_weight = 0.4: a
    # cell matching a hot region's tool scores 0.6, its tool condition
    # 0.4, both 1.0. (convention, untested)
    allocation_tool_weight: float = knob(0.6, lo=0.0, hi=1.0)
    allocation_condition_weight: float = knob(0.4, lo=0.0, hi=1.0)
    # region_novelty_smoothing = 0.5: weight on the newest novelty score
    # in a region's running novelty (an EMA; 1 - this on the old value).
    # (convention, untested)
    region_novelty_smoothing: float = knob(0.5, lo=0.0, hi=1.0)
    # gap_min_rows = 3: rows a region needs before "one signature only"
    # counts as stuck; gap_rich_signatures = 3: distinct signatures at
    # which a region counts as explored. (convention, untested)
    gap_min_rows: int = knob(3, lo=1)
    gap_rich_signatures: int = knob(3, lo=1)
    # gap_value_stuck = 1.0, gap_value_rich = 0.2, gap_value_unknown =
    # 0.5: the behavior-gap score for a stuck, an explored, and an
    # undecided region. (convention, untested)
    gap_value_stuck: float = knob(1.0, lo=0.0, hi=1.0)
    gap_value_rich: float = knob(0.2, lo=0.0, hi=1.0)
    gap_value_unknown: float = knob(0.5, lo=0.0, hi=1.0)
    # gap_weight = 0.7: share of a region's behavior value from the gap
    # score; the rest (0.3) from its fault rate. (convention, untested)
    gap_weight: float = knob(0.7, lo=0.0, hi=1.0)
    # adaptive_verify_explore_floor = 0.55: under mode="adaptive" a
    # short clock (explore share under this) re-rolls a prompt to peek
    # for a different outcome even when its region shows one behavior.
    # (convention, untested)
    adaptive_verify_explore_floor: float = knob(0.55, lo=0.0, hi=1.0)
    # pass_threshold = 0.5: see PASS_THRESHOLD.
    pass_threshold: float = knob(PASS_THRESHOLD, lo=0.0, hi=1.0)
    # smoothing_alpha = 1.0: Laplace's rule of succession, (s + a) /
    # (n + 2a), for the group hazard and the mixed rate; a = 1 is the
    # uniform prior on a rate, one pseudo-observation each way. Why
    # unanimous groups are stopped at all: they carry no gradient (DAPO,
    # arXiv 2503.14476; Lambert 2025, chapter Reasoning). No paper states a
    # prior for the decision; this is the engine's own, untested against
    # a = 0.5.
    smoothing_alpha: float = knob(1.0, lo=0.0)

    # --- run-level notes --------------------------------------------

    # short_share_floor = 0.08, long_share_floor = 0.10: under these
    # shares of short or long asks the writer is nudged toward that
    # length. (convention, untested)
    short_share_floor: float = knob(0.08, lo=0.0, hi=1.0)
    long_share_floor: float = knob(0.10, lo=0.0, hi=1.0)
    # followup_starved_min = 8, followup_starved_divisor = 4: the run is
    # marked followups_starved at 8 or more missed follow-ups that are
    # also at least rows / 4. (convention, untested)
    followup_starved_min: int = knob(8, lo=0)
    followup_starved_divisor: int = knob(4, lo=1)
    # semantic_duplicate_novelty = 0.05: a row under this semantic
    # novelty is a duplicate in the duplicate rate. (convention,
    # untested)
    semantic_duplicate_novelty: float = knob(0.05, lo=0.0, hi=1.0)
    # idle_judge_share = 0.1: an rl pool idle on verdicts for more than
    # this share of the run gets the "add situations" note. (convention,
    # untested)
    idle_judge_share: float = knob(0.1, lo=0.0, hi=1.0)
    # tier_mix_min_rows = 20, tier_mix_tolerance = 0.10: see the
    # constants TIER_MIX_MIN_ROWS and TIER_MIX_TOLERANCE above.
    tier_mix_min_rows: int = knob(TIER_MIX_MIN_ROWS, lo=1)
    tier_mix_tolerance: float = knob(TIER_MIX_TOLERANCE, lo=0.0, hi=1.0)
    # progress_every_s = 10.0, progress_every_rows = 10: never more than
    # this long, or this many finished rollouts, between progress lines.
    # (convention, untested)
    progress_every_s: float = knob(10.0, lo=0.0)
    progress_every_rows: int = knob(10, lo=1)
    # flush_report_rows = 25, flush_report_s = 5.0: the streamed-output
    # log line is written every 25 rows or 5 s. (convention, untested)
    flush_report_rows: int = knob(25, lo=1)
    flush_report_s: float = knob(5.0, lo=0.0)


def knob_names() -> tuple[str, ...]:
    """Every ``advanced`` key :class:`RunKnobs` reads, in field order."""
    return tuple(f.name for f in fields(RunKnobs))


def knob_default(name: str) -> int | float | None:
    """The default of the ``advanced`` key ``name`` (a :class:`RunKnobs`
    field), or ``None`` for a field with no default. ``KeyError`` for a
    name that is not a knob."""
    spec = RunKnobs.__dataclass_fields__[name]
    return None if spec.default is MISSING else spec.default


def knob_bounds(name: str) -> tuple[float | None, float | None]:
    """``(lo, hi)`` the key ``name`` is validated against, inclusive;
    ``None`` on either side is open. ``KeyError`` for a name that is not
    a knob."""
    meta = RunKnobs.__dataclass_fields__[name].metadata
    return meta.get("lo"), meta.get("hi")


def resolve_knobs(cfg: dict[str, Any]) -> RunKnobs:
    """Pop every :class:`RunKnobs` key out of ``cfg`` (the ``advanced``
    dict), coerce it to the field's type and check its bounds. What is
    left in ``cfg`` belongs to someone else."""
    values: dict[str, Any] = {}
    for spec in fields(RunKnobs):
        if spec.name not in cfg:
            continue
        raw = cfg.pop(spec.name)
        # ``from __future__ import annotations`` leaves the field type as
        # the string it was written as, so compare against the string.
        kind = int if spec.type == "int" else float
        try:
            if isinstance(raw, bool):
                raise TypeError
            value = kind(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"advanced[{spec.name!r}] is {'an integer' if kind is int else 'a number'}; "
                f"got {raw!r}. The default is {spec.default!r}."
            ) from None
        if kind is int and float(raw) != float(value):
            raise ValueError(
                f"advanced[{spec.name!r}] is an integer; got {raw!r}. "
                f"The default is {spec.default!r}."
            )
        lo, hi = spec.metadata.get("lo"), spec.metadata.get("hi")
        if lo is not None and value < lo:
            raise ValueError(
                f"advanced[{spec.name!r}]={value!r} is below its floor {lo!r}. "
                f"The default is {spec.default!r}."
            )
        if hi is not None and value > hi:
            raise ValueError(
                f"advanced[{spec.name!r}]={value!r} is above its ceiling {hi!r}. "
                f"The default is {spec.default!r}."
            )
        values[spec.name] = value
    return RunKnobs(**values)


def laplace(successes: float, trials: float, alpha: float = 1.0) -> float:
    """Laplace's rule of succession: ``(s + a) / (n + 2a)``. With ``a = 1``
    a rate seen 0 of 0 times is 1/2, 0 of 1 is 1/3, and so on."""
    return (float(successes) + alpha) / (float(trials) + 2.0 * alpha)


# DELIVERED_LONG_CONVERSATION_TURNS = 3: a conversation with at least this
# many user turns counts as long in ``coverage["delivered"]``
# (``user_turns_3plus_share``). Convention, untested.
DELIVERED_LONG_CONVERSATION_TURNS = 3
# DELIVERED_FAULT_SHORTFALL = 0.7: a delivered fault share under this
# fraction of the requested ``fault_rate`` is reported as a gap between
# what was asked for and what the rows carry. Convention, untested.
DELIVERED_FAULT_SHORTFALL = 0.7
# DELIVERED_FAULT_LEAK = 0.01: a fault share above this when ``fault_rate``
# was 0 is reported the same way. Convention, untested.
DELIVERED_FAULT_LEAK = 0.01
# DELIVERED_STANCE_MIN_SHARE = 0.5: the requested stances must cover at
# least this share of the delivered rows, else the gap is reported.
# Convention, untested.
DELIVERED_STANCE_MIN_SHARE = 0.5

# DELIVERED_TURNS_MIN_REQUEST = 2: an ``avg_turns`` request below this is
# not checked against the delivered mean (one turn cannot fall short).
# Convention, untested.
DELIVERED_TURNS_MIN_REQUEST = 2
# DELIVERED_TURNS_SHORTFALL = 0.5: a delivered mean under this fraction of
# the requested ``avg_turns`` is reported as a gap. Convention, untested.
DELIVERED_TURNS_SHORTFALL = 0.5

# ---------------------------------------------------------------------
# methods (OPD, OPSD, async RL): whileai/methods.py
# ---------------------------------------------------------------------

# OPD_DIVERGENCE = "reverse_kl": the per-token divergence on-policy
# distillation minimizes on the student's own samples. GKD (Agarwal et al.
# 2023, arXiv:2306.13649) offers forward KL, reverse KL and JSD and finds
# the best one task dependent; the reasoning-distillation line settled on
# reverse KL as a per-token advantage log p_teacher - log p_student
# (Thinking Machines 2025, tinker-cookbook train_on_policy; Qwen3,
# arXiv:2505.09388), which is what prime-rl's ``opd`` and TRL's
# DistillationTrainer (beta=1.0) compute. ``forward_kl`` and ``jsd`` are
# offered for a trainer that has them.
OPD_DIVERGENCE = "reverse_kl"
# OPD_TOP_K = 32: the teacher support the signal lives on. Li et al. 2026
# (arXiv:2604.13016) show on-policy distillation works by raising
# student/teacher top-k overlap and that k >= 4 matches the sampled-token
# loss while k = 1 fails; Fu et al. 2026 (arXiv:2603.25562) compute the KL
# over the teacher's top 32 to keep the signal off filler tokens.
OPD_TOP_K = 32
# OPD_SAMPLES = 4: student rollouts per prompt. The tinker-cookbook recipe
# and Li et al. 2026 (arXiv:2604.13016) both sample 4; Fu et al. 2026
# (arXiv:2603.25562) 8. Distillation forms no group baseline, so 4 is a
# throughput choice, not a variance one.
OPD_SAMPLES = 4
# OPD_TEMPERATURE = 1.0: sample and score at the same temperature so the
# teacher's log-probabilities are on the distribution the student drew
# from (Thinking Machines 2025; Li et al. 2026, arXiv:2604.13016).
OPD_TEMPERATURE = 1.0
# OPD_MAX_TOKENS = 8192: the response cap. Li et al. 2026
# (arXiv:2604.13016) measure the teacher signal decaying past about 7k
# tokens; Fu et al. 2026 (arXiv:2603.25562) find the log-probability gap
# widening late in long sequences.
OPD_MAX_TOKENS = 8192
# OPD_LEARNING_RATE_LORA = 1e-4 / OPD_LEARNING_RATE_FULL = 1e-6: the
# optimizer step for an adapter and for full weights. The tinker-cookbook
# recipe trains a rank-128 LoRA at 1e-4 (5e-5 full); Li et al. 2026
# (arXiv:2604.13016) and Fu et al. 2026 (arXiv:2603.25562) train full
# weights at 1e-6 and 2e-6.
OPD_LEARNING_RATE_LORA = 1e-4
OPD_LEARNING_RATE_FULL = 1e-6

# OPSD_PRIVILEGED = "demonstration": what the teacher sees that the student
# does not. A passing demonstration of the same task is the SDFT form
# (Shenfeld et al. 2026, arXiv:2601.19897) and the one prime-rl's ``opsd``
# implements; ``reference`` (the answer, Zhao et al. 2026,
# arXiv:2601.18734), ``hint`` (Penaloza et al. 2026, arXiv:2602.04942) and
# ``feedback`` (a successful rollout plus the environment's error text,
# Hübotter et al. 2026, arXiv:2601.20802) are the other forms.
OPSD_PRIVILEGED = "demonstration"
# OPSD_DIVERGENCE = "reverse_kl": SDFT (arXiv:2601.19897), SDPO
# (arXiv:2601.20802) and prime-rl's ``opsd`` use the reverse KL to the
# privileged teacher; Zhao et al. 2026 (arXiv:2601.18734) find forward KL
# with pointwise clipping better for answer-conditioned teachers, so
# ``forward_kl`` is offered for a trainer that has it.
OPSD_DIVERGENCE = "reverse_kl"
# OPSD_ANCHOR = "ema" / OPSD_ANCHOR_ALPHA = 0.01: what the teacher's
# weights are, and the moving-average rate. Unregularized self-distillation
# diverges (SDPO ablation, arXiv:2601.20802, 50.6 vs 36.1); SDFT
# (arXiv:2601.19897) holds the teacher as an exponential moving average of
# the student at 0.01 to 0.05 and SDPO at 0.01; Zhao et al. 2026
# (arXiv:2601.18734) freeze the initial weights (``initial``). ``live`` is
# the unanchored variant prime-rl runs.
OPSD_ANCHOR = "ema"
OPSD_ANCHOR_ALPHA = 0.01
# OPSD_SAMPLES = 1: rollouts per prompt. Self-distillation forms no group
# baseline, so one sample per prompt is what SDFT (arXiv:2601.19897) and
# Zhao et al. 2026 (arXiv:2601.18734) train with; SDPO (arXiv:2601.20802)
# samples 4 because its feedback is another rollout of the same prompt.
OPSD_SAMPLES = 1
# OPSD_TEMPERATURE = 1.0: SDPO and SRPO (arXiv:2604.02288) sample at 1.0,
# Zhao et al. 2026 at 1.1; the student and teacher are scored on the same
# draw either way.
OPSD_TEMPERATURE = 1.0
# OPSD_MAX_TOKENS = 4096: the response cap Kaur et al. 2026
# (arXiv:2607.05184) and SDFT (2048, arXiv:2601.19897) train under; the
# privileged teacher's signal is on the answer, not on a long trace.
OPSD_MAX_TOKENS = 4096
# OPSD_LEARNING_RATE = 5e-6: the step SDPO (arXiv:2601.20802), Zhao et al.
# 2026 (arXiv:2601.18734), SRPO (arXiv:2604.02288) and Kaur et al. 2026
# (arXiv:2607.05184) all train at.
OPSD_LEARNING_RATE = 5e-6
# OPSD_TEMPLATE = "Here is an example of an expert response: ...": the
# system message that carries the demonstration to the teacher; prime-rl's
# ``opsd`` default text, so a config written here and one written by hand
# put the same prompt in front of the teacher.
OPSD_TEMPLATE = (
    "Here is an example of an expert response:\n<demonstration>\n{demonstration}\n</demonstration>"
)

# ASYNC_OFF_POLICY_STEPS = 8: how many optimizer steps a rollout may lag the
# policy that trains on it. ScaleRL (Khatri et al. 2025, arXiv:2510.13786)
# runs PipelineRL with 8 off-policy steps and finds it raises speed, not
# the ceiling; AReaL (Fu et al. 2025, arXiv:2505.24298) bounds staleness
# at 4 for code and 8 for math; prime-rl's own default is 8, TRL's
# AsyncGRPOTrainer's is 4. One step is free (Noukhovitch et al. 2024,
# arXiv:2410.18252).
ASYNC_OFF_POLICY_STEPS = 8
# ASYNC_CORRECTION = "ipo" / ASYNC_IPO_EPS = 0.3 / ASYNC_ICEPOP_RATIO =
# (0.5, 5.0) / ASYNC_TIS_CAP = 2.0: the per-token correction for the gap
# between the sampler's and the trainer's log-probabilities, which exists
# even at zero staleness (Yao et al. 2025, "Your Efficient RL Framework
# Secretly Brings You Off-Policy RL Training"). ``ipo`` is prime-rl's
# default: mask a token whose probability moved more than 0.3 (prime-rl's
# default eps). ``icepop`` masks a token whose trainer/sampler ratio leaves
# 0.5 to 5.0, the band Ring-1T trained under (Ling Team 2025,
# arXiv:2510.18855; prime-rl's own default band is 0.2 to 5.0). ``tis``
# caps the ratio at 2.0, verl's default for Yao et al.'s truncated
# importance sampling.
ASYNC_CORRECTION = "ipo"
ASYNC_IPO_EPS = 0.3
ASYNC_ICEPOP_RATIO = (0.5, 5.0)
ASYNC_TIS_CAP = 2.0

# ---------------------------------------------------------------------
# single-rollout methods (FlashReinforce, SAO, BPCO): whileai/methods.py
# One trajectory per prompt and no group to take a baseline over, so each
# method brings its own baseline: the batch mean (FlashReinforce) or a
# critic (SAO, BPCO). Sections are per method so parallel work reconciles
# by section, not by line.
# ---------------------------------------------------------------------

# --- FlashReinforce (Hu et al. 2026, NVIDIA) ---------------------------

# FLASH_REINFORCE_TRUST = 0.1: the sequence trust region. A trajectory is
# admitted to the update when its mean sampled-action KL proxy to the
# policy that sampled it is at most this; above it the whole trajectory
# is masked, the drift a token-level ratio cannot correct (Hu et al.
# 2026, FlashREINFORCE, section on the sequence trust region).
# (convention, untested: the paper's own value is to be read off the
# PDF and written here)
FLASH_REINFORCE_TRUST = 0.1
# FLASH_REINFORCE_OFF_POLICY_STEPS = 8: the policy lag the method is built
# to absorb. The Qwen3-30B-A3B run trains at a lag of about eight updates
# and beats GRPO at a lag of one (Hu et al. 2026, FlashREINFORCE); the
# same bound ScaleRL (arXiv:2510.13786) and prime-rl default to.
FLASH_REINFORCE_OFF_POLICY_STEPS = 8
# FLASH_REINFORCE_LEARNING_RATE = 1e-6: the optimizer step on full
# weights. (convention, untested: to be read off the paper)
FLASH_REINFORCE_LEARNING_RATE = 1e-6
# FLASH_REINFORCE_TEMPERATURE = 1.0: sample at the temperature the ratio
# is taken at, so the behavior log-probabilities on the row are the ones
# the correction divides by. (convention, untested)
FLASH_REINFORCE_TEMPERATURE = 1.0
# FLASH_REINFORCE_MAX_TOKENS = 8192: the response cap. (convention,
# untested: to be read off the paper)
FLASH_REINFORCE_MAX_TOKENS = 8192

# --- SAO, single-rollout asynchronous optimization (Hou et al. 2026) ---
# Every number below is read off Hou, Li, Tang and Dong 2026,
# arXiv:2607.07508, section 4.1 unless the comment says otherwise; the
# paper releases no code, so the paper is the only source.

# SAO_RATIO = (0.7, 6.0) / SAO_RATIO_CODING = (0.2, 4.0): the token band
# of direct double-sided importance sampling, (1 - eps_low, 1 + eps_high).
# A token whose current/rollout probability ratio leaves the band is
# masked to zero, not clipped, whichever sign its advantage has (Hou et
# al. 2026, arXiv:2607.07508, section 3.1, eq. 3). The reasoning-with-
# Python run trains at eps_low 0.3 and eps_high 5.0; the SWE-Bench coding
# run at eps_low 0.8 and eps_high 3.0 with every other knob the same
# (section 4.1).
SAO_RATIO = (0.7, 6.0)
SAO_RATIO_CODING = (0.2, 4.0)
# SAO_GAE_ALPHA = 1.5: length-adaptive GAE, lambda_policy = 1 - 1/(alpha *
# L) for a response of L model-generated tokens (VAPO, Yue et al. 2025,
# arXiv:2504.05118, which trains at 0.4; Hou et al. 2026,
# arXiv:2607.07508, section 4.1, sets 1.5), so the weight of the terminal
# reward on the first token, lambda ** (L - 1), stays about exp(-1/alpha)
# whatever the length.
SAO_GAE_ALPHA = 1.5
# SAO_GAMMA = 1.0: the discount in the skip-observation TD residual delta =
# r + gamma V(next action token) - V(token) (Hou et al. 2026,
# arXiv:2607.07508, eq. 4 and 5). The paper writes gamma and gives it no
# value; 1 is what the PPO and VAPO lines it builds on train language
# models at, and the value that makes the lambda_critic = 1 target the
# undiscounted return. (convention, untested)
SAO_GAMMA = 1.0
# SAO_CRITIC_STEPS = 2: value-network updates per policy update, the
# "faster value update" K (Hou et al. 2026, arXiv:2607.07508, section
# 3.2 and 4.1, K = 2; one update per batch loses 2.3 points on AIME2025
# and 5 on BeyondAIME, Table 4).
SAO_CRITIC_STEPS = 2
# SAO_CRITIC_WARMUP = 10: the value model's warmup, "a 10-step warmup
# period" (Hou et al. 2026, arXiv:2607.07508, section 4.1). The paper
# does not say whether that is the critic optimizer's learning-rate
# warmup or critic-only steps before the policy moves; a trainer reads
# it as whichever it has.
SAO_CRITIC_WARMUP = 10
# SAO_LEARNING_RATE = 1e-6 / SAO_CRITIC_LEARNING_RATE = 5e-6: the policy
# and value-model optimizer steps on full weights (Hou et al. 2026,
# arXiv:2607.07508, section 4.1; no adapter run is reported).
SAO_LEARNING_RATE = 1e-6
SAO_CRITIC_LEARNING_RATE = 5e-6
# SAO_BATCH = 128: trajectories per policy update at a group size of 1,
# the same 128 the GRPO arms get as 16 prompts by 8 rollouts (Hou et al.
# 2026, arXiv:2607.07508, section 4.1).
SAO_BATCH = 128
# SAO_TEMPERATURE = 1.0: the sampler's temperature. The paper evaluates
# at temperature 1.0 and top-p 1.0 (section 4.1) and does not say what
# it samples training rollouts at; 1.0 keeps the rollout
# log-probabilities the ones the band divides by. (convention, untested)
SAO_TEMPERATURE = 1.0
# SAO_MAX_TOKENS = 131072: the trajectory's token budget, "a max-length
# of 128k tokens" for both the reasoning and the coding run (Hou et al.
# 2026, arXiv:2607.07508, section 4.1), the whole multi-turn context
# including tool output (up to 50 turns for math, 300 OpenHands turns for
# SWE-Bench Verified), not a per-response cap.
SAO_MAX_TOKENS = 131072

# --- BPCO, best practice critic optimization (Qi et al. 2026) ----------

# BPCO_CLIP = 0.2: the DPPO clip epsilon. The clip range on the ratio is
# eps divided by the behavior probability of the token, so a rare token
# gets a wider range (Qi et al. 2026, arXiv:2608.23566). (convention,
# untested: the paper's own value is to be read off the PDF)
BPCO_CLIP = 0.2
# BPCO_GAE_ALPHA = 0.4: length-adaptive GAE for the policy advantage,
# lambda_pi = 1 - 1/(alpha * L) (Qi et al. 2026, arXiv:2608.23566); the
# critic's own target is Monte Carlo, lambda_V = 1.
BPCO_GAE_ALPHA = 0.4
# BPCO_REWARD_RANGE = (0.0, 1.0): the interval the critic's prediction is
# bounded to through a scaled arctangent, R_min + (R_max - R_min)(1/2 +
# atan(z)/pi) (Qi et al. 2026, arXiv:2608.23566); a 0/1 outcome reward
# lives on this interval (PASS_REWARD).
BPCO_REWARD_RANGE = (0.0, 1.0)
# BPCO_CRITIC_WARMUP = 15: policy updates the critic trains alone before
# the policy moves (Qi et al. 2026, arXiv:2608.23566).
BPCO_CRITIC_WARMUP = 15
# BPCO_LEARNING_RATE = 1e-6 / BPCO_CRITIC_LEARNING_RATE = 1e-5: the policy
# and critic optimizer steps (Qi et al. 2026, arXiv:2608.23566).
BPCO_LEARNING_RATE = 1e-6
BPCO_CRITIC_LEARNING_RATE = 1e-5
# BPCO_TEMPERATURE = 1.0: sample at the temperature the ratio is taken at.
# (convention, untested)
BPCO_TEMPERATURE = 1.0
# BPCO_MAX_TOKENS = 8192: the response cap. (convention, untested: to be
# read off the paper)
BPCO_MAX_TOKENS = 8192

# PRIME_RL_GPUS = 2: the fewest GPUs a prime-rl run takes. It runs the
# inference engine and the trainer as separate processes on separate
# devices (INTELLECT-2, arXiv:2505.07291, section 2), so one of each is
# the floor; the writer splits a larger count half and half, the split
# the Qwen3.8-27B quant runs used (2026-09-14 to 2026-09-19).
PRIME_RL_GPUS = 2
# PRIME_RL_STEPS = 100: optimizer steps when none are given. Zhao et al.
# 2026 (arXiv:2601.18734) and SDFT (arXiv:2601.19897) report their
# self-distillation numbers at 100 steps; the tinker-cookbook OPD recipe
# at 200. Enough to read a curve, short enough to be a first run.
PRIME_RL_STEPS = 100
# PRIME_RL_BATCH = 64: prompts per optimizer step. SDFT (arXiv:2601.19897)
# trains at 16 to 64, Li et al. 2026 (arXiv:2604.13016) at 64, Kaur et al.
# 2026 (arXiv:2607.05184) at 64; a quarter of TRAINING_BATCH_PROMPTS.
PRIME_RL_BATCH = 64
# PRIME_RL_SEQ_LEN = 12288: the trainer's sequence cap, prompt plus
# response. It has to exceed the response cap plus the longest prompt;
# OPD_MAX_TOKENS plus a 4k prompt is this. The Qwen3.8-27B smoke at 12288
# truncated 11 to 28 percent of rollouts at a 2k response cap on an
# agentic task, so an agent with long tool output raises it.
PRIME_RL_SEQ_LEN = 12288
# PRIME_RL_LEARNING_RATE_LORA = 1e-5 / PRIME_RL_LEARNING_RATE_FULL = 1e-6:
# the GRPO step. DAPO (arXiv:2503.14476) and Dr. GRPO (arXiv:2503.20783)
# train full weights at 1e-6; the LoRA rate is the one the Qwen3.8-27B
# quant runs converged at on prime-rl (2026-09-14 to 2026-09-19).
PRIME_RL_LEARNING_RATE_LORA = 1e-5
PRIME_RL_LEARNING_RATE_FULL = 1e-6
# PRIME_RL_EVAL_EXAMPLES = 60 / PRIME_RL_EVAL_GROUP = 2: the in-run eval,
# 60 tasks by 2 rollouts every quarter of the run. Under the 160 by 4 the
# paper recipes hold out (recipes/papers), on purpose: an in-run eval
# reads the curve, the recipe's held-out delta is the result. (convention,
# untested)
PRIME_RL_EVAL_EXAMPLES = 60
PRIME_RL_EVAL_GROUP = 2

__all__ = [
    "AGENT_MAX_TOKENS_FLOOR",
    "ALLOC_GAIN",
    "ALPHA",
    "ASYNC_CORRECTION",
    "ASYNC_ICEPOP_RATIO",
    "ASYNC_IPO_EPS",
    "ASYNC_OFF_POLICY_STEPS",
    "ASYNC_TIS_CAP",
    "BASE_PASS_RATE",
    "BOOTSTRAP_DRAWS",
    "BPCO_CLIP",
    "BPCO_CRITIC_LEARNING_RATE",
    "BPCO_CRITIC_WARMUP",
    "BPCO_GAE_ALPHA",
    "BPCO_LEARNING_RATE",
    "BPCO_MAX_TOKENS",
    "BPCO_REWARD_RANGE",
    "BPCO_TEMPERATURE",
    "CEILING_PASS_RATE",
    "CHARS_PER_TOKEN",
    "CI_LEVEL",
    "DEAD_AGENT_BUDGET_MULTIPLE",
    "DEAD_AGENT_MIN_ERRORS",
    "DECISION_TIMEOUT_S",
    "DECISION_UNSURE_BAND",
    "DECONTAM_NGRAM",
    "DECONTAM_OVERLAP",
    "DEFAULT_AVG_TURNS",
    "DEFAULT_BUDGET",
    "DEFAULT_CARDS_PER_WAVE",
    "DEFAULT_COMPLETIONS_PER_REQUEST",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_EXTRA_CARDS",
    "DEFAULT_LLM_JUDGE_CONCURRENCY",
    "DEFAULT_MIN_USER_TURNS",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_PROBE",
    "DEFAULT_SEED",
    "DEFAULT_SELECT_TARGET",
    "DEFAULT_WRITER_FLIGHT",
    "DELIVERED_FAULT_LEAK",
    "DELIVERED_FAULT_SHORTFALL",
    "DELIVERED_LONG_CONVERSATION_TURNS",
    "DELIVERED_STANCE_MIN_SHARE",
    "DELIVERED_TURNS_MIN_REQUEST",
    "DELIVERED_TURNS_SHORTFALL",
    "DIFFICULTY_BAND",
    "DIFFICULTY_BAND_ROLLOUTS",
    "ENV_DECONTAMINATION_EXAMPLES",
    "ENV_DECONTAMINATION_NGRAM",
    "ENV_EVAL_EXAMPLES",
    "ENV_EVAL_ROLLOUTS",
    "ENV_HARNESS_MIX",
    "ENV_HARNESS_SEED",
    "ENV_HOLDOUT_FRACTION",
    "ENV_MAX_TURNS_FALLBACK",
    "FAULT_STATUSES",
    "FINGERPRINT_STEM_MIN_LEN",
    "FLASH_REINFORCE_LEARNING_RATE",
    "FLASH_REINFORCE_MAX_TOKENS",
    "FLASH_REINFORCE_OFF_POLICY_STEPS",
    "FLASH_REINFORCE_TEMPERATURE",
    "FLASH_REINFORCE_TRUST",
    "FLIP_FLAG",
    "GPU_PRICE_SOURCE",
    "GPU_USD_PER_HOUR",
    "HACK_THRESHOLD",
    "HOLDOUT_BUCKET_HEX_CHARS",
    "HUNG_SLOT_S",
    "JUDGE_CHECK_SAMPLE",
    "JUDGE_COMPARE_CONCURRENCY",
    "JUDGE_CONCURRENCY_CAP",
    "JUDGE_FINAL_TEXT_CHARS",
    "JUDGE_MAX_TOKENS",
    "JUDGE_PAYLOAD_CHARS",
    "JUDGE_POLICY_CHARS",
    "JUDGE_SITUATION_CHARS",
    "JUDGE_TEMPERATURE",
    "LEAK_MIN_QUOTE_CHARS",
    "LENGTH_GAP_FLAG",
    "LOCAL_MODEL_TEMPERATURE",
    "MAX_COMPLETIONS_PER_REQUEST",
    "MAX_GOLD_ASK",
    "MAX_SAMPLES_PER_CALL",
    "MAX_SKIPPED_SHARE",
    "MESSAGE_EXAMPLES",
    "MIN_AGREEMENT",
    "MIN_CI_TASKS",
    "MIN_GOLD",
    "MIN_KAPPA",
    "MIN_REPLY_TOKENS",
    "MIN_RERUNS",
    "MIN_TRAIN_SEEDS",
    "MONITOR_BUFFER",
    "MONITOR_CONCURRENCY",
    "MONITOR_DELTA",
    "MONITOR_EVERY",
    "MONITOR_K",
    "MONITOR_LENGTH_PCT",
    "MONITOR_MAX_NEW_TOKENS",
    "MONITOR_N_PROMPTS",
    "MONITOR_SAMPLE_BATCH",
    "MONITOR_SAMPLE_TEMPERATURE",
    "MONITOR_SAMPLE_TOP_P",
    "MONITOR_SCAN_MIN",
    "MONITOR_SCAN_PERMUTATIONS",
    "MONITOR_WINDOW",
    "MONITOR_WINDOW_BOOTSTRAPS",
    "OK_STATUSES",
    "OPD_DIVERGENCE",
    "OPD_LEARNING_RATE_FULL",
    "OPD_LEARNING_RATE_LORA",
    "OPD_MAX_TOKENS",
    "OPD_SAMPLES",
    "OPD_TEMPERATURE",
    "OPD_TOP_K",
    "OPSD_ANCHOR",
    "OPSD_ANCHOR_ALPHA",
    "OPSD_DIVERGENCE",
    "OPSD_LEARNING_RATE",
    "OPSD_MAX_TOKENS",
    "OPSD_PRIVILEGED",
    "OPSD_SAMPLES",
    "OPSD_TEMPERATURE",
    "OPSD_TEMPLATE",
    "PARENT_HEAD_CHARS",
    "PASS_REWARD",
    "PASS_THRESHOLD",
    "PLATFORM_CREDENTIAL_TTL_S",
    "PLATFORM_ERROR_DETAIL_CHARS",
    "PLATFORM_HF_DATASET_TIMEOUT_S",
    "PLATFORM_HF_MODEL_POLL_S",
    "PLATFORM_HF_MODEL_TIMEOUT_S",
    "PLATFORM_HF_POLL_S",
    "PLATFORM_HOLDOUT_PROVE_EFFECT",
    "PLATFORM_IMPORT_MAX_ROWS",
    "PLATFORM_IMPORT_POLL_S",
    "PLATFORM_IMPORT_TIMEOUT_S",
    "PLATFORM_PUT_S_PER_MB",
    "PLATFORM_PUT_TIMEOUT_S",
    "PLATFORM_REQUEST_TIMEOUT_S",
    "PLATFORM_REWARD_MODEL_BATCH",
    "PLATFORM_STUDIO_MAX_ROWS",
    "PLATFORM_TRACE_PAGE_SIZE",
    "PLATFORM_UNKNOWN_IDS_SHOWN",
    "PLATFORM_UPLOAD_TIMEOUT_S",
    "POSITION_FLIP_FLAG",
    "POWER",
    "PRIME_RL_BATCH",
    "PRIME_RL_EVAL_EXAMPLES",
    "PRIME_RL_EVAL_GROUP",
    "PRIME_RL_GPUS",
    "PRIME_RL_LEARNING_RATE_FULL",
    "PRIME_RL_LEARNING_RATE_LORA",
    "PRIME_RL_SEQ_LEN",
    "PRIME_RL_STEPS",
    "PROBE_MIN_N",
    "PROGRESS_MIN_BUDGET",
    "PROGRESS_MIN_ROWS_FOR_ESTIMATE",
    "PROVE_EFFECT",
    "REJECTION_SAMPLING_MIN_K",
    "REPORT_LIST_ITEMS",
    "RL_FAULT_RATE",
    "RL_ROLLOUTS_PER_ASK",
    "RL_ROLLOUTS_PER_PROMPT",
    "ROLLOUTS_PER_TASK",
    "RULE_AXIS_CAP_GRID",
    "RULE_AXIS_CAP_REPORT",
    "SAMPLING_TEMPERATURE_MAX",
    "SAO_BATCH",
    "SAO_CRITIC_LEARNING_RATE",
    "SAO_CRITIC_STEPS",
    "SAO_CRITIC_WARMUP",
    "SAO_GAE_ALPHA",
    "SAO_GAMMA",
    "SAO_LEARNING_RATE",
    "SAO_MAX_TOKENS",
    "SAO_RATIO",
    "SAO_RATIO_CODING",
    "SAO_TEMPERATURE",
    "SATURATION_CAP",
    "SCENARIO_ID_CHARS",
    "SECONDS_PER_HOUR",
    "SEMANTIC_SIMILARITY",
    "SFT_COMPLETIONS_PER_PROMPT",
    "SFT_PHRASINGS_PER_SITUATION",
    "SHORT_HASH_CHARS",
    "STOP_GRACE_S",
    "SYSTEM_PROMPT_HEAD_CHARS",
    "TEXT_HEURISTICS",
    "TIER_MIX_MIN_ROWS",
    "TIER_MIX_TOLERANCE",
    "TIMEOUT_TOKENS_PER_SECOND",
    "TOOL_SCHEMA_SPAN_CHARS",
    "TRACE_EXEMPLARS_PER_TOOL",
    "TRACE_EXEMPLAR_DICT_KEYS",
    "TRACE_EXEMPLAR_LIST_ITEMS",
    "TRACE_EXEMPLAR_MAX_CHARS",
    "TRACE_EXEMPLAR_STRING_CHARS",
    "TRACE_EXPLORATION_FLOOR",
    "TRACE_EXPLORATION_MAX",
    "TRACE_LEAK_EXAMPLES",
    "TRACE_LEAK_THRESHOLD",
    "TRACE_MIN_SUPPORT",
    "TRACE_PSEUDO_PRODUCTION_FRACTION",
    "TRACE_RATE_FLOOR",
    "TRACE_REPORT_LIST_CAP",
    "TRACE_STATE_PRIORITY",
    "TRACE_SUPPORT_SATURATION",
    "TRACE_TASK_HASH_CHARS",
    "TRAINING_BATCH_PROMPTS",
    "TRAINING_ERROR_CHARS",
    "TRAINING_FLUSH_EVERY",
    "TRAINING_FLUSH_SECONDS",
    "TRAINING_KNOBS",
    "TRAINING_LORA_ALPHA",
    "TRAINING_LORA_RANK",
    "TRAINING_MAX_BATCH",
    "TRAINING_POLL_MIN_S",
    "TRAINING_POLL_S",
    "TRAINING_SFT_EPOCHS",
    "TRAIN_MIN_MIXED_TASKS",
    "TRANSIENT_BACKOFF_S",
    "TRANSIENT_TRIES",
    "TRUNCATED_REPLY_CHARS",
    "WORLD_CI_FAIL_ONE_IN",
    "WORLD_CONDITION_MODES",
    "WORLD_CREATED_ID_MODULUS",
    "WORLD_DATE_YEARS",
    "WORLD_DEFAULT_FAULT_MODE",
    "WORLD_DEFAULT_FAULT_RATE",
    "WORLD_EXISTS_SHARE",
    "WORLD_EXPRESSION_CHARS",
    "WORLD_HINT_CHARS",
    "WORLD_ID_RANGE",
    "WORLD_ISSUED_ID_HEX",
    "WORLD_JITTER_DIVISOR",
    "WORLD_MALFORMED_PAYLOAD",
    "WORLD_REF_CHARS",
    "WORLD_SEARCH_HITS",
    "WORLD_SHELL_FLAVORS",
    "WORLD_STALE_AS_OF",
    "WORLD_TEMPLATE_HITS",
    "Z_95",
    "RunKnobs",
    "TextHeuristics",
    "knob",
    "knob_bounds",
    "knob_default",
    "knob_names",
    "laplace",
    "resolve_knobs",
]
