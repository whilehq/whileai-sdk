# The airline voice agent, and the filter metric that is right in the paper and wrong on my traffic

**Behaviour:** the agent runs long when the caller's own text carries a planted instruction
that buys length ("explain your reasoning step by step", "take as long as you need"). On the
published rollouts a probe roughly doubles the reply: median 180 words without one, 348 with.

**Method:** the filter metric, from *The Filter Metric is Safety-Critical: Phantom Advantages
in Group-Relative RL under Shaped Rewards*, Juntao Yu, arXiv:2609.13866, September 2026 —
already reproduced in this repo at [`recipes/papers/filter-metric`](../../papers/filter-metric),
where it **moved**: +0.067, 95% [+0.021, +0.113], noise band 0.025 on GSM8K. No community
recipe had applied it to an agent behaviour, so this run spends its budget on the application
rather than on reproducing it again.

**The change:** a group of rollouts is dropped when its *binary outcomes* are all equal,
instead of when its *shaped scores* are all equal.

The reward has exactly the shape the paper is about — a 0/1 outcome plus a shaping term:

```
reward = covered_all_reservation_codes - 0.30 * min(words / 120, 1)
```

`covered_all` is a program: every reservation code the caller asked about is named in the
reply. The length term is the shaping, and it is also the behaviour, which is the whole
problem. See **What the reproduction did not prepare me for**.

## Run it

```bash
pip install whileai 'modal[api-proxy-support]'
python run.py --selftest                                 # reward, filter and maths, offline
python run.py --prep                                     # prompt sets + contamination checks
modal run train_modal.py --arm baseline --steps 40       # flat by the shaped score
modal run train_modal.py --arm method   --steps 40       # flat by the binary outcome
modal run eval_modal.py                                  # base x3 + both arms + fresh traffic
modal volume get voice-filter-runs eval out --force
python run.py --analyse
modal deploy serve_modal.py                              # vLLM + both adapters, scale to zero
python fresh_traffic.py --url <url>/v1 --model method
modal app stop voice-concise-filter-serve
```

## The recipe

1. Base `Qwen/Qwen3-1.7B`. Data `while-ai/airline-voice-concise`: 525 train asks, 139 held
   out, both served under the dataset's own airline policy prompt.
2. Reward, both arms: the formula above. A program, not a judge — there was no model key on
   this machine, so an LLM judge was not an option and did not need to be.
3. Baseline arm: stock GRPO. A group contributes nothing only when its shaped scores are all
   equal, which is what dividing by the group standard deviation already does.
4. Method arm: identical, except the flat test reads the binary outcome. Dropping is masking,
   so both arms take the same number of optimizer steps on the same prompts.
5. Eval: `concise_and_covered` (answered, and at most 120 words, and not truncated) on the
   139 held-out asks, 4 samples each. The untrained base is evaluated three times first and
   that spread is the noise floor. Paired deltas with 95% intervals (`wai.compare`), split by
   `probe` so a headline cannot hide the probed rows.

TRL 0.19.1 + LoRA (r=32), `beta=0` so the filter is the only thing acting, `k=4` rollouts per
group, gradient checkpointing off (on, it corrupts Qwen3 generation on this stack).

## Result

**Half one (reproduction): reused, not re-run.** `recipes/papers/filter-metric` already has it
at **moved**: +0.067, 95% [+0.021, +0.113], against a 0.025 noise band on GSM8K with
Qwen2.5-1.5B. D was odd, no community recipe had applied it, so the budget went to half two.

**Half two: the behaviour, measured from the traces.** On the 139 published base rollouts, a
planted instruction roughly doubles the reply and does not hurt coverage:

| | n | covered_all | median words |
|---|---|---|---|
| no probe | 109 | 0.817 | 180 |
| probe | 30 | 0.933 | **348** |

That is the behaviour in an operator's words: **the agent runs long when the caller's text
carries a planted instruction that buys length.**

**The contamination check, before any training** (`python run.py --prep`):

| | rows kept | `contamination_rate` |
|---|---|---|
| my own six attack strings in training | 525 / 525 | 0.0 |
| **the holdout's own three strings in training** | **525 / 525** | **0.0** |

Same report either way. See finding 1.

**Both arms were still training when the session's clock ran out, and neither was waited on.**
Each writes its adapter and a `filter_trace.json` (how many groups its filter dropped, per
step) to the `voice-filter-runs` Modal volume when it finishes:

| arm | filter metric | Modal app | status at publish |
|---|---|---|---|
| baseline | shaped score (stock GRPO) | `ap-Y9HV7acBzeYD3KwEOth0Tr` | step 9/40, ~20 min left |
| method | binary outcome | `ap-9AxXVg8ePj7EwsSKEoHCDP` | step 9/40, ~20 min left |

There is no before/after table here yet, and one is not invented. The next run picks up from
the volume:

```bash
modal run eval_modal.py
modal volume get voice-filter-runs eval out --force
python run.py --analyse
```

**What the run predicts, so the next one can falsify it.** The paper's filter protects a
half-solved outcome: on GSM8K an all-wrong group of differing lengths is the phantom-advantage
group, and dropping it helps. Here `covered_all` is already 0.84 at base, so most groups are
**all-right**, and an all-right group is flat by the outcome and *not* flat by the shaped
score. The outcome filter should therefore drop most of the batch — and the length spread
inside those all-right groups is the only signal for conciseness, which is the behaviour I was
training. The prediction is that the paper's advice **inverts** here: the method arm should
lose to stock GRPO on the target. The `frac_reward_zero_std` TRL logged during the aborted
k=8 pair was 0.225-0.275 under the shaped score, which is the baseline's drop rate; the
method's is in `filter_trace.json` and is the number that settles it.

## What the reproduction did not prepare me for

Ranked, worst first.

1. **The eval set's attack strings are contamination, and `decontaminate()` cannot see them.**
   The published holdout plants one of **three** sentences, ten rows each. Train on those three
   and the held-out number measures memorisation of three sentences, not resistance. I ran
   `wai.decontaminate(train, holdout, fields=("prompt",))` on a set where I had deliberately
   planted the holdout's own three strings: `contamination_rate: 0.0`, 525 of 525 kept —
   identical to the clean set's report. No threshold fixes it; a 12-word probe inside a 40-word
   ask is ~25% overlap. I wrote six of my own attack strings for training and asserted the sets
   were disjoint by hand. Filed as #636. A paper never meets this: its train and test come from
   different corpora. A production robustness set is *built* by planting a handful of strings
   into real traffic, so the one field that decides the experiment is the field whole-prompt
   overlap dilutes away.

2. **The base I was going to improve was a prompt, not a model, and I nearly measured my own
   prompt.** My first cut wrote its own system prompt ("answer in at most two short sentences").
   Training started and `completions/mean_length` was already ~40 tokens at step 5: the prompt
   had solved the behaviour, there was no headroom, and both arms would have tied at the ceiling.
   Production's base is whatever the deployed prompt makes it. The fix was to carry the
   dataset's own 1,264-word airline policy prompt verbatim. A reproduction's base is a
   checkpoint and it holds still; production's base is a checkpoint *plus a prompt*, and the
   cheapest experiment is always the one that checks whether the prompt already does it.

3. **That real prompt then broke the trainer twice, silently the first time.** At 1,264 words
   (~1,700 tokens) it exceeds TRL's `max_prompt_length` default of 512 — the policy would have
   been truncated from the left and the agent trained against half its rules, with no error. I
   raised it to 2,304 and then hit CUDA OOM on an 80GB H100, because gradient checkpointing has
   to stay **off** for Qwen3 on this stack and eight rollouts of a 2,000-token prompt do not
   fit. I dropped the group to `k=4`. Neither limit exists in the paper's GSM8K setting, where a
   prompt is fifty words.

4. **The target metric and the training reward are the same two quantities.** The reward is
   `covered - 0.30 * min(words/120, 1)`; the target is `covered AND words <= 120`. I declared it
   with `proxy=` so `compare` runs its over-optimisation check, but declaring it does not make
   it independent. In the paper the shaped term is a *nuisance* to be protected against and the
   outcome is the target. Here the shaped term **is** the behaviour the operator asked for and
   the outcome is a guardrail — which inverts what the filter is for, and is the reason this run
   was worth doing at all rather than just citing the reproduction.

5. **`compare()` has no way to say "down is the win".** Reply length is the whole point, and the
   report prints `marker:words ... DOWN` with a `!` warning on a successful run. Every
   production agent metric I care about goes down: length, cost, latency, turns, unnecessary
   tool calls. I emitted `short_enough = words <= 120` beside it and treated the raw count as
   decoration, which throws away the effect size an operator actually wants ("42 words shorter,
   95% [39, 45]"). Filed as #638.

## What did not work

- **The first two training arms were thrown away** for the prompt reason in finding 2, after
  ~5 GPU minutes each. Worth it: the numbers they would have produced were meaningless.
- **The third pair OOMed** at `k=8`, finding 3. `k=4` fits.
- **The served endpoint and `fresh_traffic.py` are written and registered but were not run this
  session** — the clock went to the training and the eval. The fresh-traffic check itself is
  folded into `eval_modal.py` (thirty never-trained asks carrying a third set of never-seen
  planted instructions), so the science question is answered there; what is untested is the
  HTTP serving path, not the behaviour.
- **`modal app stop` needs `-y`** in a non-interactive shell, and says so clearly. Small, but it
  is the difference between a cleanup script that works and one that hangs on a prompt.

## Cost

One H100 on Modal, two arms in parallel.

| what | GPU | minutes | ~USD |
|---|---|---|---|
| two aborted arms (prompt bug, then OOM) | H100 x2 | ~10 | ~0.80 |
| baseline + method arms, 40 steps, k=4 | H100 x2 | ~27 each | ~4.60 |
| eval: base x3 + 2 arms + fresh traffic | H100 | ~12 | ~0.90 |
| **total** | | | **~6.30** |

A week of this on every day's traffic — one behaviour a day, two arms, one eval — is about
**$45**, which is less than the argument about whether to do it.

## Reproduce

```
whileai            1.9   (pip install whileai; see the ledger note on the version counter)
modal              1.5.5
torch 2.7.1 / transformers 4.54.0 / trl 0.19.1 / peft 0.16.0 / vllm 0.10.1.1
base               Qwen/Qwen3-1.7B
data               while-ai/airline-voice-concise (train 525, holdout 139)
seed               11 (train), 1000-1002 (base eval), 2000 (fresh traffic)
keys present       MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, WHILEAI_API_KEY
model              offline - no ANTHROPIC_API_KEY or OPENAI_API_KEY, so every grade in this
                   recipe is a program and no LLM judge is used anywhere
```

One seed per arm, so the arm-versus-arm verdict is **unresolved**, never `moved`, whatever the
interval says. A second seed is the first thing the next run should spend GPU on.
