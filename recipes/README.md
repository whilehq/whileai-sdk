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
- Keys come from the environment, never from files: `WHILEAI_API_KEY`
  (platform), `OPENAI_API_KEY` / `OPENAI_BASE_URL` (any OpenAI-compatible
  model), `ANTHROPIC_API_KEY` (Claude). A recipe that trains on Modal says so.
- Generated files land in the recipe's `out/` or `raw/`, both gitignored.
  Checked-in data is the exception and is named in the README.
- Every measured claim is a paired number with a 95% interval on a held-out
  set, produced by the SDK (`pass_at`, `delta_report`), never a mean alone.
- `smoke.sh` runs the whole recipe with no key, no GPU and no spend, in under
  a minute. CI runs every one of them on every pull request.

## Write one

Copy [`_template/`](_template) into the step it belongs to, replace the parts
in angle brackets, and open a pull request. The rest is in
[CONTRIBUTING.md](../CONTRIBUTING.md#contributing-a-recipe).

```bash
cp -r recipes/_template recipes/03-select/my-recipe
sh recipes/03-select/my-recipe/smoke.sh
```

A recipe that trains on Modal we run on our own account before merging: a
fork's pull request gets no secrets from this repository, by design.

## 01-simulate

| Recipe | What you learn | Needs | Takes |
|---|---|---|---|
| [`bring-your-own-agent`](01-simulate/bring-your-own-agent) | the `agent(message) -> {steps, final_text}` contract, what a run says when the agent raises, why an `evaluate()` score must not become the reward | nothing | seconds |
| [`agent-behavior`](01-simulate/agent-behavior) | a coding agent with instructed bad habits, every turn on the platform as OTLP spans, a held-out test suite and an LLM judge disagreeing about the same turn, rows grouped by `scenario_id` for RL | `WHILEAI_API_KEY` and a model endpoint (`--dry-run` needs neither) | 8 min for 40 runs |
| [`verifiers`](01-simulate/verifiers) | rewards that are programs: `MathEqual`, `All` (answer and format), `CodeExec` against hidden tests, `JSONSchema`, each honoring the judge contract | nothing | seconds |

## 02-measure

| Recipe | What you learn | Needs | Takes |
|---|---|---|---|
| [`eval-your-agent`](02-measure/eval-your-agent) | evals for the agent you already have: wrap it, write the policy as a judge, pass@1 with an interval per policy branch, the coverage warnings that catch a hollow run, a CI gate | nothing | seconds |
| [`is-your-eval-any-good`](02-measure/is-your-eval-any-good) | whether a number your eval produced means anything: ceiling, headroom, criteria that cannot fail, self-noise, the judge, contamination, and the three checks that void a base-vs-tuned comparison outright | nothing | seconds |
| [`pass-at-k`](02-measure/pass-at-k) | pass@1 with its interval, pass^k, pass@k, the per-ask histogram the mean hides, and headroom = what a grouped update can learn | nothing | seconds |
| [`reward-hacking`](02-measure/reward-hacking) | reward hacking caught before, during and after training: the within-ask scan, the judge probes, the trajectory flags, the proxy-vs-target verdict | nothing | seconds |
| [`compare-judges`](02-measure/compare-judges) | six judges on the same 300 labeled rollouts, one ranked table: agreement with its interval, kappa, leak rate, unsure and unjudged counts, seconds per row; Jev, the hosted judge, Claude, and the policy judging itself | `TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY` or a login; `--dry-run` and `report` need nothing | ten minutes, or seconds offline |
| [`safety-evals`](02-measure/safety-evals) | a safety suite for a tool-using agent: prompt injection, exfiltration, secret leakage, unauthorized writes, benign controls; four trajectory markers as the judge, pass^k per attack class, a before/after that fails the fix which got safe by refusing | nothing | seconds |
| [`safety-evals-marketplace`](02-measure/safety-evals-marketplace) | the same eval where the untrusted text is user-generated content and the private data is per tenant; `live.py` runs it on a local model through Ollama | nothing offline; Ollama for `live.py` | seconds offline, minutes live |

## 03-select

| Recipe | What you learn | Needs | Takes |
|---|---|---|---|
| [`schema`](03-select/schema) | one row file projected into eval, SFT, preference, GRPO prompts, OPSD and OPD targets; the `Task`/`Rollout`/`Judgment`/`Marker` split that makes that possible | nothing | seconds |
| [`prime-intellect-rl`](03-select/prime-intellect-rl) | `simulate(mode="rl")` for uniform groups, the gradient gate (`diagnose.py`) that catches a reward the policy can game before you train, prompts in the `verifiers` shape | an account key; `VLLM_API_KEY` for the shared pool | 3 min for 800 rollouts |
| [`character`](03-select/character) | a constitution to traits, graded replies per trait, a judge checked against the spec's own labels, length-matched pairs and masked SFT rows, before/after on an adversarial holdout | nothing offline; a model endpoint for the live run | seconds offline, 2.5 min live |

## 04-train

| Recipe | What you learn | Needs | Takes |
|---|---|---|---|
| [`hosted-loop`](04-train/hosted-loop) | push graded rows, `wai.train` SFT on Qwen3-4B, `wai.serve` the adapter, one chat completion from the endpoint | `WHILEAI_API_KEY` | about a minute of A10G, plus a cold start |
| [`report-run`](04-train/report-run) | the typed objects the platform tracks (a tracked agent with its harness, behaviors, runs, live traffic), why a harness is versioned by its fingerprint, and why a version is scored on every behavior | `WHILEAI_API_KEY` for the real thing; nothing for the smoke run | 10 seconds |
| [`identity`](04-train/identity) | a leak-free SFT set that teaches a name and maker, with Modal scripts for the LoRA and for the identity/leak eval | nothing to generate; Modal and an A10G to train | seconds to generate |
| [`grpo`](04-train/grpo) | TRL `GRPOTrainer` with LoRA on a verifiable rule, `HackMonitor` and reward/KL on the run page, paired pass@1 before/after with per-category deltas, loss variants and `--balance` as flags | Modal, one A10G; the key is optional | under 15 min at 40 steps |
| [`dpo`](04-train/dpo) | on-policy pairs from `build_preference_pairs`, TRL `DPOTrainer`, the reward margin on the run page, iterated rounds with `--from-run`, constructed negatives | Modal, one A10G; the key is optional | about 10 min |
| [`text-to-sql`](04-train/text-to-sql) | hill-climb a model on a schema with a verifier as the reward: a seeded Postgres, 741 execution-checked tasks, `SQLExec`, benchmarks through `simulate(tasks=)`, self-distillation, GRPO rounds on Modal with vLLM generation and Postgres in the container, every round measured on the same holdout | Postgres, `WHILEAI_API_KEY`; Modal and an H100 to train | minutes to benchmark, an hour a round |
| [`resist-planted-instruction`](04-train/resist-planted-instruction) | a behaviour rubric decided by code, the criterion promoted into the reward on probe evidence, rejection sampling from the base itself, a pre-registered random-selection control, three arms from one vLLM process with attack and clean halves apart | nothing offline; a vLLM serving Qwen3-4B to generate; Modal, one H100 and one L40S to train and eval | seconds offline; about an hour and five dollars end to end |

## 05-export

| Recipe | What you learn | Needs | Takes |
|---|---|---|---|
| [`hugging-face`](05-export/hugging-face) | rows to a Hub dataset repo (one split per purpose, commit tagged by dataset id), any Hub split onto the account with a profile, a run's adapter to a model repo | `WHILEAI_API_KEY` and a Hugging Face account connected on the platform | a minute |

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

| Recipe | What you learn | Needs | Takes |
|---|---|---|---|
| [`same-entrypoint-before-after`](community/same-entrypoint-before-after) | pin a task set across a model swap and run both arms of a before/after through one entry point, so the delta measures the model and not the SDK; the noise floor from base re-runs | `WHILEAI_API_KEY`; `--dry-run` needs nothing | about 33 minutes of warm A10G, seconds offline |
| [`force-the-branch`](community/force-the-branch) | force a policy branch with `result_shapes=` so a marker scores the decision and not the agent's mood; whether a reported regression survives a forced holdout | `WHILEAI_API_KEY`; `--dry-run` needs nothing | about 7 minutes of warm A10G, seconds offline |
| [`can-the-judge-be-trusted`](community/can-the-judge-be-trusted) | a gold label a machine can compute, what `judge_agreement` and `judge_trust` measure, why a judge's errors matter by shape more than by rate | `WHILEAI_API_KEY`; `--dry-run` needs nothing | about 25 minutes of warm A10G, seconds offline |
| [`hosted-grpo-vs-sft`](community/hosted-grpo-vs-sft) | what the hosted `sft`, `grpo` and `dpo` methods consume, hosted SFT and GRPO on the same rows against one base, pulling the adapter back into PEFT form | `WHILEAI_API_KEY`; `--dry-run` needs nothing | two hosted runs under $1, seconds offline |
| [`how-much-contamination-survives`](community/how-much-contamination-survives) | how much human-labelled paraphrase contamination (QQP, PAWS) the lexical `decontaminate()` rule removes (about 9%) and the `embedder=` pass removes (about 90%), at what false-positive cost, with controls under every arm | `datasets` and one Hub download; `--dry-run` needs nothing | ten minutes of CPU, seconds offline |
| [`who-protects-the-holdout`](community/who-protects-the-holdout) | which `decontaminate()` rule carries the protection on a `simulate()` holdout (`same_task`, 98 to 100%) and what is left when the eval set has no ids (18 to 30%); why `contamination_rate: 0.0` does not certify an external holdout | nothing | under a minute |

## Where the main README's pieces live

- **Simulate and grade:** `01-simulate/bring-your-own-agent` (callable),
  `01-simulate/agent-behavior` (traces), `01-simulate/verifiers` (program as
  reward). Every recipe grades with a callable so it runs without a key;
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
  into the same run page through `wai.TrainerCallback`).
- **Before and after:** `grpo`, `dpo` and `text-to-sql` call `run.delta(...)`;
  `character/measure.py` and `safety-evals` call `delta_report` directly, the
  latter with `must_not_regress=["helpful_on_benign"]` so a fix that got safe
  by refusing fails. Every delta is a paired bootstrap over tasks with a 95%
  interval.
- **Trust checks:** judge trust against gold labels in `character`
  (`judge_vs_spec`), `safety-evals` (hand-labeled transcripts, the refusal
  probe) and `agent-behavior` (held-out suite vs judge); reward hacking in
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
