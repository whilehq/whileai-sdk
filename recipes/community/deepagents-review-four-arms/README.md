# Fine-tune a Deep Agents reviewer from its LangSmith traces, with and without While

A stock LangChain Deep Agents agent on Qwen3.8-27B reviews real patches and says approve or
reject. Its traces go to LangSmith. We fine-tune it twice on those traces with smithtune's own
renderer, split and schedule: once on the traces smithtune's judge council keeps, once on the
traces `wai.simulations.optimize` keeps. Nothing else differs. Both models are scored on SWE-bench Verified
patch review, a benchmark nobody here built, and on 84 repositories the traces never touched.

What you will learn: that which traces you train on moves a fine-tune as much as the trainer
does, how to hold a comparison to one variable (same trainer, same server, same sampling), and
why a single base run is not a baseline. You need `OPENROUTER_API_KEY` and `LANGSMITH_API_KEY`
for traces, Docker for smithtune (it imports `fcntl`, so Linux or macOS), and a Modal account for
training and serving. `python run.py --dry-run` needs none of it.

## Result

SWE-bench Verified patch review, 250 reviews on 125 issues, 50/50 approve/reject, labels from the
official evaluation. Every arm on one vLLM server, thinking off, same sampling:

| arm | correct | model calls per review |
|---|---|---|
| base, three runs | 11.6%, 12.0%, 9.6% | 36.4 |
| tuned Deep Agents profile, no training | 20.4% | 31.4 |
| SFT on smithtune's pick (258 traces) | 46.0% | 23.5 |
| **SFT on While's pick (173 traces)** | **56.8%** | **18.2** |

While's pick against smithtune's, paired by review: **+10.8 points [+4.8, +16.8]** and **5.3 fewer
calls [−6.7, −4.0]**, on 65% fewer training tokens per epoch (3.6M against 10.2M). The noise floor
from three base runs (`wai.eval_variance`, t at 2 df) is 7.8 points; the gain clears it.

On the 84 held-out repositories (246 reviews): 61.4% against 63.8%, a tie (−2.4 [−7.3, +2.4]),
and 3.1 fewer calls [−4.3, −2.0] for While's pick. All numbers are in `results.json`
(`python make_results.py`).

## How it works

1. `data.py` builds the review tasks from `nebius/SWE-agent-trajectories` and
   `nebius/SWE-bench-extra` (pinned revisions): for each issue with a passing and a failing
   SWE-agent patch, one of each. Splits are by repository. `wai.decontaminate` checks train and dev
   against the holdout (0 of 1,306 dropped, every rule ran). Four snapshots deleted from GitHub are
   listed in `unavailable.json` and left out everywhere.
2. `collect.py` runs the stock agent (`agent.py`: `create_deep_agent`, read-only filesystem over a
   tarball of the base commit) and writes LangSmith feedback `correctness` and `model_calls` on every
   root run. 400 training reviews: 67% correct.
3. Without While: `smithtune dataset pull` with the correctness filter, then `smithtune dataset
   triage` with `rubric.md`. The council kept 258 of 260.
4. With While: `pick.py` runs `wai.simulations.optimize(mode="sft")` with reward 1 only for a correct review
   within 8 model calls, and writes feedback `while_pick = 1`; smithtune pulls it with that filter.
   173 kept.
5. `export_sft.py` runs inside the smithtune image and calls smithtune's `prepare_sft_rows`
   (reasoning omitted, its default), `split_rows` (80/10/10) and `render_row_tokens` (the Baseten
   renderer: one datum per assistant turn, history at weight 0). `smithtune prepare` itself stops
   without a Baseten Loops or Fireworks account; this skips only that preflight.
6. `train_modal.py` trains LoRA on one H200 with smithtune's Baseten defaults: rank 8, batch 32,
   learning rate 1e-4, seed 42, up to 5 epochs, stop when validation loss does not improve, keep the
   best epoch. Both arms stopped after epoch 3 and kept epoch 2.
7. `serve_modal.py` serves the base and both adapters from one vLLM 0.26 server with the token-level
   Qwen3.5 tool parser (`qwen35_tool_parser.py`) and `enable_thinking=false` for every request.
8. `public_bench.py` builds the SWE-bench Verified set from six public leaderboard submissions
   (`s3://swe-bench-submissions`, `patch.diff` and `report.json`), none of whose twelve repositories
   is in the training traces. `report.py` and `make_results.py` do the paired comparisons.

## Run it

```bash
cd recipes/community/deepagents-review-four-arms
python run.py --dry-run                    # offline: the comparison and the pick rule on fixture rows
python data.py && python public_bench.py   # tasks (byte-identical from the pinned revisions)
python run.py traces                       # stock agent over train -> LangSmith
python pick.py                             # While's pick -> feedback while_pick=1
docker build -f smithtune.Dockerfile -t smithtune:0.1.0 .
# smithtune dataset pull / triage / push, then export_sft.py in the image (see step 3-5)
modal deploy train_modal.py                # then spawn train for sft-with and sft-without
REVIEW_ADAPTERS=sft-with,sft-without modal deploy serve_modal.py
python collect.py --arm srv-sft-with --split public --base-url <server>/v1 --model sft-with --api-key-env REVIEW_SERVER_KEY
python make_results.py
```

## Caveats

- One training seed per arm. The public-benchmark gain clears the three-run noise floor; a second
  seed per arm is the next run.
- The two training sets differ in size as well as in which traces they hold (173 against 258). A
  random 173 from the council's set would separate the pick from the size.
- smithtune omits reasoning from training rows by default, so both tuned models are non-thinking
  models and are scored against the base with thinking off. With thinking on, through OpenRouter,
  the base scored 70.3%, 75.1% and 74.6% on the held-out repositories; that run on SWE-bench
  Verified is still to do. Re-running the same base once moved it 4.5 points with an interval
  that excluded zero, which is why every comparison here carries the three-run floor.
- Training ran on Modal, not Baseten Loops (not enabled for our workspace), with smithtune's
  renderer, split and schedule.

## Next

`python collect.py --arm base --split public` through OpenRouter for the thinking-on base, then a
second seed per SFT arm and the size-matched random control. Traces keep flowing to LangSmith, so
the same `pick.py` picks the next round.
