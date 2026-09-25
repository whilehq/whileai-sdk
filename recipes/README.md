# Recipes

One post-training run, as five steps. Each folder is a step; each recipe inside
it is a runnable, self-contained script with a README that says what you
learn, what you need, and how long it takes. Read them in order the first
time; after that, jump to the step you are on.

```
recipes/
  01-simulate/   make rollouts: an agent, situations, a reward that is a program
  02-measure/    say what the numbers mean: pass@k, headroom, reward hacking, safety
  03-select/     turn graded rows into training data: SFT rows, pairs, RL groups
  04-train/      train it, hosted or on your own GPU, and watch the run page
  05-export/     ship the data and the adapter
  papers/        recent papers, each one change to a step recipe, with the number it moved
```

## Conventions

Every recipe follows the same shape, so a coding agent can run one without
reading the others:

- `README.md` opens with **what you learn**, **needs**, **takes**. "Offline"
  means no key and no network.
- The first command in the README runs the whole recipe. Scripts take
  `--help`. Anything long-running takes `--limit` or `--steps` for a smoke run.
- Paths are relative to the recipe folder unless the README says "from the
  repo root".
- Run a recipe from its own directory. From the repo root, `python -m`,
  `modal run` and a notebook put the working directory first on `sys.path`,
  so the clone's `whileai/` folder shadows the installed package with no
  message. The first line every recipe prints, `whileai <version> from
  <dir>`, says which one ran: `(source tree, not the installed wheel)` is the
  clone. To develop against the tree on purpose, `pip install -e .` (or `uv
  sync`); the line then says `installed editable`.
- Keys come from the environment, never from files: `WHILEAI_API_KEY`
  (platform), `OPENAI_API_KEY` / `OPENAI_BASE_URL` (any OpenAI-compatible
  model), `ANTHROPIC_API_KEY` (Claude), `FIREWORKS_API_KEY` (Fireworks). A recipe
  that trains on Modal or on Fireworks says so.
- Generated files land in the recipe's `out/` or `raw/`, both gitignored.
  Checked-in data is the exception and is named in the README.
- Every measured claim is a paired number with a 95% interval on a held-out
  set, produced by the SDK (`pass_at`, `delta_report`), never a mean alone.
- `smoke.sh` is the offline path through the recipe: no key, no GPU, no
  spend, under a minute. An offline recipe runs whole. A recipe that trains
  or calls the platform runs everything up to the paid step (the prompts,
  the reward, the split, the config, the no-key message) through `--dry-run`,
  `--offline` or a selftest, and its README's first command is that same
  free line with the paid one and its cost after it. CI runs every
  `smoke.sh` on every pull request. Recipes without one: the two-arm
  training replications under `papers/*`, because each `recipe.py` imports
  `modal` at the top and CI does not install it; `python recipe.py
  --selftest` is their offline check once it is. `papers/meta-harness` is
  the exception: a search loop, not a trained arm, so it is in this shape
  and carries a `smoke.sh`.

## Write one

Copy [`_template/`](_template) into the step it belongs to, replace the parts
in angle brackets, and open a pull request. The rest is in
[CONTRIBUTING.md](../CONTRIBUTING.md#contributing-a-recipe).

```bash
cp -r recipes/_template recipes/03-select/my-recipe
uv run sh recipes/03-select/my-recipe/smoke.sh
```

A recipe that trains on Modal we run on our own account before merging: a
fork's pull request gets no secrets from this repository, by design.

**Costs** in the tables below are the recipe's stated wall time at Modal's
on-demand GPU prices ([modal.com/pricing](https://modal.com/pricing), read
2026-09-20: A10G $1.10 an hour, L40S $1.95, H100 $3.95), rounded up. "Free"
means no GPU and no paid call; a hosted run uses the same GPUs through the
platform.

## 01-simulate

| Recipe | What you learn | Needs | Takes | Costs |
|---|---|---|---|---|
| [`bring-your-own-agent`](01-simulate/bring-your-own-agent) | the `agent(message) -> {steps, final_text}` contract, what a run says when the agent raises, why an `evaluate()` score must not become the reward | nothing | seconds | free |
| [`verifiers`](01-simulate/verifiers) | rewards that are programs: `MathEqual`, `All` (answer and format), `CodeExec` against hidden tests, `JSONSchema`, each honoring the judge contract | nothing | seconds | free |
| [`smol-data-envs`](01-simulate/smol-data-envs) | an outside environment as a verifier: 5,394 Kaggle data questions, the policy's program runs next to the tables, the dataset's grader is the reward; an environment failure is `None`, not `0`, and a shell `echo` earns nothing | nothing offline; any model spec live | seconds offline, ~1 min live | free offline; model calls live |
| [`swarm-rescue`](01-simulate/swarm-rescue) | on the tasks where all 8 rollouts fail, whether a particle swarm of rollouts that share their best attempts finds a passing answer that resampling does not; rescue rate paired by task, four arms at one budget | `WHILEAI_API_KEY` (offline with `--dry-run`) | ~2 hours on the hosted model | hosted |

## 02-measure

| Recipe | What you learn | Needs | Takes | Costs |
|---|---|---|---|---|
| [`eval-your-agent`](02-measure/eval-your-agent) | evals for the agent you already have: wrap it, write the policy as a judge, pass@1 with an interval per policy branch, the coverage warnings that catch a hollow run, a CI gate | nothing | seconds | free |
| [`is-your-eval-any-good`](02-measure/is-your-eval-any-good) | whether a number your eval produced means anything: ceiling, headroom, criteria that cannot fail, self-noise, the judge, contamination, and the three checks that void a base-vs-tuned comparison outright | nothing | seconds | free |
| [`character-to-the-wall`](02-measure/character-to-the-wall) | whether a persona holds when two of its own values collide: situations built so no reply can honor both principles, the spec's authority ordering as the answer key, held_wall and kept_lower graded apart to tell caving from rigidity, a judge checked against the set's own labels, before/after on held-out walls | nothing offline; a model endpoint and a judge for the live run | seconds offline, about 2 min live | free offline; model calls live |
| [`pass-at-k`](02-measure/pass-at-k) | pass@1 with its interval, pass^k, pass@k, the per-ask histogram the mean hides, and headroom = what a grouped update can learn | nothing | seconds | free |
| [`reward-hacking`](02-measure/reward-hacking) | reward hacking caught before, during and after training: the within-ask scan, the judge probes, the trajectory flags, the proxy-vs-target verdict | nothing | seconds | free |
| [`compare-judges`](02-measure/compare-judges) | six judges on the same 300 labeled rollouts, one ranked table: agreement with its interval, kappa, leak rate, unsure and unjudged counts, seconds per row; Jev, the hosted judge, Claude, and the policy judging itself | `TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY` or a login; `--dry-run` and `report` need nothing | ten minutes, or seconds offline | free offline; judge API calls live, no GPU |
| [`safety-evals`](02-measure/safety-evals) | a safety suite for a tool-using agent: prompt injection, exfiltration, secret leakage, unauthorized writes, benign controls; four trajectory markers as the judge, pass^k per attack class, a before/after that fails the fix which got safe by refusing | nothing | seconds | free |
| [`safety-evals-marketplace`](02-measure/safety-evals-marketplace) | the same eval where the untrusted text is user-generated content and the private data is per tenant; `live.py` runs it on a local model through Ollama | nothing offline; Ollama for `live.py` | seconds offline, minutes live | free; Ollama runs on your machine |
| [`public-benchmark`](02-measure/public-benchmark) | a public benchmark (200 GSM8K test questions) into the measurement: `wai.rows` with `MathEqual` as the reward, pass@1 with its interval, the eval's own noise over three passes, `holdout_size`, `select` dropping the groups that carry no gradient, and a `compare` report | nothing | seconds | free |

## 03-select

| Recipe | What you learn | Needs | Takes | Costs |
|---|---|---|---|---|
| [`schema`](03-select/schema) | one row file projected into eval, SFT, preference, GRPO prompts, OPSD and OPD targets; the `Task`/`Rollout`/`Judgment`/`Marker` split that makes that possible | nothing | seconds | free |
| [`prime-intellect-rl`](03-select/prime-intellect-rl) | `simulate(mode="rl")` for uniform groups, the gradient gate (`diagnose.py`) that catches a reward the policy can game before you train, prompts in the `verifiers` shape | an account key; `VLLM_API_KEY` for the shared pool | 3 min for 800 rollouts | free offline; hosted model calls live, no GPU |
| [`character`](03-select/character) | a constitution to traits, graded replies per trait, a judge checked against the spec's own labels, length-matched pairs and masked SFT rows, before/after on an adversarial holdout | nothing offline; a model endpoint for the live run | seconds offline, 2.5 min live | free offline; model endpoint calls live |

## 04-train

| Recipe | What you learn | Needs | Takes | Costs |
|---|---|---|---|---|
| [`hosted-loop`](04-train/hosted-loop) | push graded rows, `wai.train` SFT on Qwen3-4B, `wai.serve` the adapter, one chat completion from the endpoint | `WHILEAI_API_KEY` | about a minute of A10G, plus a cold start | about 5 cents (one A10G minute, plus the cold start) |
| [`report-run`](04-train/report-run) | the typed objects the platform tracks (a tracked agent with its harness, behaviors, runs, live traffic), why a harness is versioned by its fingerprint, and why a version is scored on every behavior | `WHILEAI_API_KEY` for the real thing; nothing for the smoke run | 10 seconds | free |
| [`identity`](04-train/identity) | a leak-free SFT set that teaches a name and maker, with Modal scripts for the LoRA and for the identity/leak eval | nothing to generate; Modal and an A10G to train | seconds to generate | free to generate; A10G minutes to train and eval, at $1.10 an hour |
| [`grpo`](04-train/grpo) | TRL `GRPOTrainer` with LoRA on a verifiable rule, `HackMonitor` and reward/KL on the run page, paired pass@1 before/after with per-category deltas, loss variants and `--balance` as flags | Modal, one A10G; the key is optional | under 15 min at 40 steps | about 30 cents (15 A10G minutes); the `--steps 10` check about 5 cents |
| [`dpo`](04-train/dpo) | on-policy pairs from `build_preference_pairs`, TRL `DPOTrainer`, the reward margin on the run page, iterated rounds with `--from-run`, constructed negatives | Modal, one A10G; the key is optional | about 10 min | about 20 cents (10 A10G minutes); the `--steps 10` check about 5 cents |
| [`fireworks`](04-train/fireworks) | export SFT rows and DPO pairs in Fireworks' shapes (`format="fireworks"`), the `firectl` commands that train on Fireworks GPUs and serve the result, the paired before/after through `wai.Fireworks` and `wai.compare` | nothing for the export; `FIREWORKS_API_KEY` and `firectl` for the job and the proof | export under a minute; the job is Fireworks' queue | export free; the job is billed per training token by Fireworks |
| [`sft`](04-train/sft) | LoRA SFT with TRL `SFTTrainer` on the rows lesson 7 exports (`select(mode="sft").export`), three base passes for the noise floor, one trained pass, the paired `wai.compare(run_std=)` on the held-out set; the step the course used to skip | Modal, one A10G; no key | about 10 min, under a dollar | about 11 cents (6 A10G minutes); the three runs behind the lesson about 37 cents |
| [`prime-rl`](04-train/prime-rl) | GRPO, OPSD and OPD on one taskset on prime-rl from `wai.prime_rl_config`, a launcher over Prime Intellect's published image, per-prompt held-out deltas with intervals from `wai.compare`; run e2e1: OPD matched GRPO with no reward, OPSD moved a fifth as far | Modal, two H100s an arm | about 15 min an arm | about $6 (three arms, two H100s each, 15 minutes an arm) |
| [`text-to-sql`](04-train/text-to-sql) | hill-climb a model on a schema with a verifier as the reward: a seeded Postgres, 741 execution-checked tasks, `SQLExec`, benchmarks through `simulate(tasks=)`, self-distillation, GRPO rounds on Modal with vLLM generation and Postgres in the container, every round measured on the same holdout | Postgres, `WHILEAI_API_KEY`; Modal and an H100 to train | minutes to benchmark, an hour a round | $4 to $8 a round (one to two H100 hours); the benchmark is hosted model calls |
| [`resist-planted-instruction`](04-train/resist-planted-instruction) | a behaviour rubric decided by code, the criterion promoted into the reward on probe evidence, rejection sampling from the base itself, a pre-registered random-selection control, three arms from one vLLM process with attack and clean halves apart | nothing offline; a vLLM serving Qwen3-4B to generate; Modal, one H100 and one L40S to train and eval | seconds offline; about an hour and five dollars end to end | about $5 end to end; free offline |

## 05-export

| Recipe | What you learn | Needs | Takes | Costs |
|---|---|---|---|---|
| [`hugging-face`](05-export/hugging-face) | rows to a Hub dataset repo (one split per purpose, commit tagged by dataset id), any Hub split onto the account with a profile, a run's adapter to a model repo | `WHILEAI_API_KEY` and a Hugging Face account connected on the platform | a minute | free; platform calls, no GPU |
| [`bedrock-import`](05-export/bedrock-import) | merge a LoRA adapter on Modal, import the weights into your AWS account with Bedrock Custom Model Import, measure the served model on the same held-out tasks as the vLLM run; a paired interval says the weights survived the move | a Modal account and AWS credentials | twenty minutes | Modal CPU minutes; Bedrock bills per Custom Model Unit while serving |

## papers

Recent post-training papers, each cut down to a run under an hour on one GPU:
a baseline arm, the paper's one change, the same holdout, a paired delta.
Index and contract in [`papers/README.md`](papers/README.md); the table there
is generated from each recipe's `results.json`.

## community

Runs contributed by people trying the SDK on their own problems: the script that
ran, the numbers with intervals, and what did not work. Not maintained by While;
each README names the version it ran against. Index in
[`community/README.md`](community/README.md).

| Recipe | What you learn | Needs | Takes | Costs |
|---|---|---|---|---|
| [`same-entrypoint-before-after`](community/same-entrypoint-before-after) | pin a task set across a model swap and run both arms of a before/after through one entry point, so the delta measures the model and not the SDK; the noise floor from base re-runs | `WHILEAI_API_KEY`; `--dry-run` needs nothing | about 33 minutes of warm A10G, seconds offline | about 60 cents (33 A10G minutes); free offline |
| [`force-the-branch`](community/force-the-branch) | force a policy branch with `result_shapes=` so a marker scores the decision and not the agent's mood; whether a reported regression survives a forced holdout | `WHILEAI_API_KEY`; `--dry-run` needs nothing | about 7 minutes of warm A10G, seconds offline | about 15 cents (7 A10G minutes); free offline |
| [`can-the-judge-be-trusted`](community/can-the-judge-be-trusted) | a gold label a machine can compute, what `judge_agreement` and `judge_trust` measure, why a judge's errors matter by shape more than by rate | `WHILEAI_API_KEY`; `--dry-run` needs nothing | about 25 minutes of warm A10G, seconds offline | about 50 cents (25 A10G minutes); free offline |
| [`hosted-grpo-vs-sft`](community/hosted-grpo-vs-sft) | what the hosted `sft`, `grpo` and `dpo` methods consume, hosted SFT and GRPO on the same rows against one base, pulling the adapter back into PEFT form | `WHILEAI_API_KEY`; `--dry-run` needs nothing | two hosted runs under $1, seconds offline | under $1 for the two hosted runs; free offline |
| [`how-much-contamination-survives`](community/how-much-contamination-survives) | how much human-labelled paraphrase contamination (QQP, PAWS) the lexical `decontaminate()` rule removes (about 9%) and the `embedder=` pass removes (about 90%), at what false-positive cost, with controls under every arm | `datasets` and one Hub download; `--dry-run` needs nothing | ten minutes of CPU, seconds offline | free (ten CPU minutes) |
| [`who-protects-the-holdout`](community/who-protects-the-holdout) | which `decontaminate()` rule carries the protection on a `simulate()` holdout (`same_task`, 98 to 100%) and what is left when the eval set has no ids (18 to 30%); why `contamination_rate: 0.0` does not certify an external holdout | nothing | under a minute | free |

## Where the main README's pieces live

- **Simulate and grade:** `01-simulate/bring-your-own-agent` (callable),
  `01-simulate/verifiers` (program as reward). Every recipe grades with a callable so it runs without a key;
  `data.grade(judge=...)` is the same call with the hosted judge.
- **pass@k and headroom:** `02-measure/pass-at-k`. The same `PassAt` object is
  `data.pass_at`, `ScoredData.pass_at`, and the per-trait lines in `character`.
- **Data for SFT, pairs for DPO, groups for GRPO:** `03-select/schema` (all
  six projections from one file), `03-select/prime-intellect-rl` (RL groups and
  the gate), `03-select/character` (pairs and SFT rows from graded replies),
  `04-train/dpo` (pairs from the policy being trained), `04-train/text-to-sql`
  (rejection sampling through `optimize(mode="sft")`).
- **Train:** `04-train/hosted-loop` (platform trainer, no GPU of yours);
  `identity`, `grpo`, `dpo`, `text-to-sql` (your trainer on Modal, reporting
  into the same run page through `wai.TrainerCallback`); `sft` (LoRA SFT on
  Modal from the course's own `train.jsonl`, with the before and after).
- **Before and after:** `grpo`, `dpo` and `text-to-sql` call `run.delta(...)`;
  `character/measure.py` and `safety-evals` call `delta_report` directly, the
  latter with `must_not_regress=["helpful_on_benign"]` so a fix that got safe
  by refusing fails. Every delta is a paired bootstrap over tasks with a 95%
  interval.
- **Trust checks:** judge trust against gold labels in `character`
  (`judge_vs_spec`) and `safety-evals` (hand-labeled transcripts, the refusal
  probe); reward hacking in
  `prime-intellect-rl` (effort correlation), `grpo` (`HackMonitor`) and `dpo`
  (constructed negatives); the `evaluate()` provenance guard in
  `bring-your-own-agent`.
- **Export:** `05-export/hugging-face` (Hub), `03-select/prime-intellect-rl/export_prompts.py`
  (the `verifiers` prompt shape), `03-select/schema/project.py` (JSONL per target).

## Adding a recipe

1. Put it under the step it belongs to: `recipes/<step>/<name>/`.
2. `README.md` first: what you learn, needs, takes, then the first command.
3. Register its scripts in `tests/recipes/test_offline_examples.py`
   (`CLI_EXAMPLES` for scripts that parse arguments and run `--help` with no
   key; `NEEDS_MODAL` for scripts that only compile), and add a row to the
   table above.
4. Data files it ships are ignored by default (`*.jsonl`); unignore them by
   path in `.gitignore`. Output folders go in `.gitignore` too.

Recipe READMEs cite primary papers as numbered references, resolved in a
"References" list at the end of the page, and cite the RLHF textbook
(Lambert 2025) by chapter title, never by chapter number.
