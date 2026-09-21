# The hosted loop: push, train, serve, call

Four calls from graded rows to a chat completion from the trained model,
all on the platform. One key, one A10G run, no GPU of your own.

What you will learn: the shape of a train set and a task-disjoint holdout
on the platform, what `wai.train` returns and how to wait on it, what
`wai.serve` gives you back, and how to call the served adapter. The rows
are deliberately small; this is the wiring check, not a result. You need
`WHILEAI_API_KEY` (or `wai login`); no model key, since the rows
come from the template writer and a scripted agent.

```bash
uv add whileai
wai login                 # or export WHILEAI_API_KEY=...
cd recipes/04-train/hosted-loop
python run.py --dry-run         # free: simulate, grade, split, push nothing; no key
python run.py                   # data -> train -> serve -> call: about a minute of A10G plus a cold start, about 5 cents
python run.py train --method sft --epochs 2   # any step alone; state is in hosted-loop.json
python run.py models            # what the account hosts
```

| flag | default | what it does |
|---|---|---|
| `step` | all | `data`, `train`, `serve`, `call`, or `models`; `all` runs the first four |
| `--name` | hosted-loop | dataset, agent and model name |
| `--method` | sft | `sft`, `grpo` or `dpo` |
| `--base` | Qwen/Qwen3-4B | a served base, or nothing serves |
| `--epochs` | 1.0 | SFT only |
| `--steps` | 20 | GRPO and DPO only |
| `--budget` | 96 | rollouts to simulate in `data` |
| `--seed` | 1 | the template writer's and the split's seed |
| `--timeout` | 1800 | seconds `train` waits for the run before giving up |

## What each step does

| step | call | what comes back |
|---|---|---|
| `data` | `simulate` (template writer, scripted agent), `run_judge`, `split_pseudo_production`, `push_rows` x2 | a train set and a task-disjoint holdout on the platform, with the publish gate's warnings |
| `train` | `wai.train(train_id, method="sft", base_model="Qwen/Qwen3-4B", holdout=...)`, then `run.wait(timeout=--timeout)` | a finished run: held-out loss before and after, the adapter's location, the curve at `run.url` |
| `serve` | `wai.serve("hosted-loop", run)` | a model row: `endpoint` (OpenAI-compatible base URL) and `name` (the model id to send) |
| `call` | `POST {endpoint}/chat/completions` with the account key as bearer | the trained model's reply |

One run of `python run.py` with the defaults (`--seed 1 --budget 96`,
SFT on `Qwen/Qwen3-4B`, one epoch): 24 train rows over 7 tasks, and a
72-row holdout because the splitter seeds the held-out side by failure
signature first (see below). The loss numbers are one run's; the
platform trainer is not seeded, so yours will differ:

```
== train
started sft run run_726d53b769141506: https://withwhile.com/platform/training/run_726d53b769141506
done in 47s: loss 5.0094 -> 4.1311 on 72 held-out rows
adapter: volume whileai-train-runs:/run_726d53b769141506/adapter
== serve
serving hosted-loop v1 on Qwen/Qwen3-4B
endpoint https://zeroproofai--zeroproof-serve-qwen3-4b.modal.run/v1
== call
HTTP 200 in 287s
I cannot process your request. The order ID "88213" is not valid or does not exist in our system. Please provide a valid order ID, and I will assist you accordingly.
```

The rows here are small on purpose (a scripted agent, template situations)
so the loop finishes in minutes. The loss drop shows the wiring works; it
says nothing about the agent. The reply under `== call` is the same kind
of wiring check: it shows the endpoint answers, and yours will differ. Replace `scripted_agent` and `judge` with
yours, or point `data` at rows you already graded.

## What to know before you run it

- **Only two bases serve.** `Qwen/Qwen3-4B` and `microsoft/phi-4`. The
  trainer's defaults (Qwen2.5-0.5B for SFT, 1.5B for GRPO and DPO) train
  faster but cannot be hosted; `wai.train` warns and `wai.serve` refuses.
  SFT runs on an A10G and takes about a minute here; GRPO and DPO run
  on an L40S (`--method grpo --steps 10` took 137 s on Qwen3-4B).
- **Cold starts.** The serving GPU scales to zero. The first call after
  idle can take a few minutes; `call` waits up to fifteen, and a 502, 503
  or 504 while the container is still waking is retried inside that window
  rather than raised.
- **Thinking mode.** Qwen3 reasons before it answers unless told not to.
  `call` sends `chat_template_kwargs: {"enable_thinking": false}` so the
  reply is the answer, not the reasoning.
- **Cost.** SFT here is about a minute of A10G, GRPO a few minutes of L40S, and
  `run.training["cost_usd"]` says what that came to: an estimate at Modal's list price
  (`cost_basis` names the rate and the day, `estimate: A10G at $1.10/h, modal.com/pricing
  2026-09-20`), so a run this size is a few cents; `print(run)` shows it as
  `about $0.02 (A10G, 56 s, estimate)`. Serving bills while the GPU is awake; the
  endpoint idles back to zero on its own, and rollouts and judge calls are not priced.
- **Holdout.** `split_pseudo_production` moves whole tasks and seeds the
  held-out side with one task per failure signature first, so on a tiny
  set (7 tasks here) the holdout ends up larger than the fraction asks.
  That is fine for a wiring check; a real set has hundreds of tasks.

## Where it shows up

`run.url` is the loss curve and the before/after on the platform.
`wai.models()` lists what the account hosts, `wai.get_run(run_id)` returns
the points, and the dataset cards link to the run. Docs:
https://docs.withwhile.com/api/training
