# ParseBench: hill-climb an open document-parsing agent

An open-weight agent that turns PDF pages into markdown, HTML tables and layout
boxes, scored on ParseBench [1]. It is Qwen3.8-27B served by vLLM plus the
PP-DocLayoutV3 layout detector [2], both on Modal in your own workspace. On
ParseBench it scores **78.89** on the full set and **78.53** on held-out test
documents. That puts while.ai first among open-weight entries on the
leaderboard (rakedoc-nano, 77.23, sits below the full-set interval), and
Claude Opus 5.5 at high effort (79.85) sits inside it.

What you will learn: how to hill-climb an agent harness on a public benchmark
without fooling yourself (a document-level dev/test split, tune on dev, report
test and full), which harness pieces moved the score and by how much, and why
two RL runs on synthetic chart data did not transfer. You need a Modal account
with an H200 quota, a Hugging Face token, and a few GPU hours. `python
selftest.py` needs nothing.

## Results

Overall is the mean of the five ParseBench dimensions, the number the
leaderboard reports. Intervals are 95% percentile bootstraps over documents,
2,000 draws, from `modal run serve/bench.py::bands`. Every number here is in
[`results.json`](results.json).

| Entry | Split | Overall | 95% CI | Tables | Charts | Text content | Formatting | Layout |
|---|---|---|---|---|---|---|---|---|
| while.ai (`wai_agent_v4`) | full | **78.89** | 77.82 to 79.95 | 80.50 | 80.79 | 86.34 | 70.32 | 76.51 |
| while.ai (`wai_agent_v4`) | test | **78.53** | 77.33 to 79.64 | 80.51 | 79.90 | 86.34 | 71.09 | 74.81 |
| LlamaParse Agentic | full | 87.01 | | | | | | |
| Claude Opus 5.5, high | full | 79.85 | | | | | | |
| rakedoc-nano (best open-weight before this) | full | 77.23 | | | | | | |
| Qwen3.8-27B thinking, one call per page | full | 70.79 | | | | | | |

Leaderboard rows are from ParseBench's `leaderboard.csv`, read 2026-09-25. The
base model on its own reproduces its leaderboard row on our dev split (70.84
against 70.79), so the gain is the harness, not a different server.

## The honesty rules

ParseBench ships no training split, so every number above could be a
memorized benchmark if we were careless. Three rules keep it honest:

1. **No ParseBench page is ever trained on.** The RL data is ChartNet [3]
   (CDLA-Permissive-2.0), and any ChartNet chart whose values cover 30% or more
   of a ParseBench chart page's rules is dropped (281 of 29.7k).
2. **Tune on dev only.** `waiparse/split.py` hashes the source report name
   (the page suffix `_pN` removed) and sends one report in five to dev. Every
   page of a report lands on the same side, so a report's fonts and layout
   cannot leak from dev into test.
3. **Report test and full.** Test is the other four reports in five, never
   used to pick a setting. Full is for leaderboard parity. One piece of the
   headline, the detector chart trigger in v4, was motivated by a failure seen
   on test (a chart the layout pass read as bare label lines). Its settings
   were tuned on dev only, and `waiparse/pipelines.py` says so.

## Run it

Needs: Python 3.11+, `pip install modal`, `modal setup`, and these in your
own Modal workspace:

```bash
# A key the agent and the servers share. Any long random string.
modal secret create docparse-vllm-key VLLM_API_KEY=$(openssl rand -hex 24)
# A Hugging Face token (read access) for the model, detector and dataset downloads.
modal secret create huggingface-secret HF_TOKEN=hf_...
# Your workspace name: the part before "--" in any URL `modal deploy` prints.
export DOCPARSE_WORKSPACE=<your-workspace>
export PYTHONUTF8=1
```

Every server URL the agent calls is built from `DOCPARSE_WORKSPACE`
(`waiparse/endpoints.py`). `DOCPARSE_SERVER`, `DOCPARSE_LAYOUT_URL` and
`DOCPARSE_TUNED_SERVER` override one URL each. Modal shortens a label longer
than 63 characters, so with a long workspace name, copy the URL `modal deploy`
printed.

From this directory:

```bash
python selftest.py                                   # free: split, endpoints, layout mapping, configs
modal deploy serve/serve_vlm.py                      # Qwen3.8-27B on H200, scales to zero
modal deploy serve/serve_layout.py                   # PP-DocLayoutV3 on L4
modal run serve/bench.py::download                   # ParseBench into the docparse-data volume
modal run serve/bench.py::run --pipeline wai_agent_v4 --split smoke   # a few pages, end to end
modal run serve/bench.py::run --pipeline wai_agent_v4 --split dev     # tune here
modal run serve/bench.py::run --pipeline wai_agent_v4 --split test    # report this
modal run serve/bench.py::run --pipeline wai_agent_v4 --split full    # and this
python scores.py wai_agent_v4:dev wai_agent_v4:test wai_agent_v4:full
modal run serve/bench.py::bands --specs wai_agent_v4:test --pairs "wai_agent_full:test|wai_agent_v4:test"
```

The first run on a split writes `/data/split_dev` or `/data/split_test` from
the full set. Results land in the `docparse-runs` volume at
`/runs/bench/<pipeline>/<split>`. `--group table` (or `chart`, `text`,
`layout`) runs one dimension. Name a function after `bench.py::`: a bare
`modal run serve/bench.py` is ambiguous.

## What each harness piece bought

The agent (`waiparse/agent.py`) runs per page: a layout pass, then specialist
passes on regions, then emits markdown and layout boxes in the shape the
scorers read. Each row is one change, scored on dev against the row above it.

| Pipeline | What changed | Dev overall | Paired gain, 95% CI |
|---|---|---|---|
| `wai_baseline_thinking` | the leaderboard's Qwen3.8-27B layout prompt, one call per page | 70.84 | |
| v1 (not kept) | layout pass, then crop every Table and Picture to a specialist | 69.60 | not paired: tables +7.4, charts -16.6 |
| `wai_agent` | charts read from the whole page in one thinking pass, so titles and legends stay in view | 74.60 | +3.64 (1.04 to 6.17) over baseline |
| `wai_agent_style` | re-read text blocks from crops and copy only the bold, strikethrough, superscript and subscript spans onto the page text | 77.18 | +2.58 (0.93 to 4.20) |
| `wai_agent_full` | layout boxes from PP-DocLayoutV3, text from the overlapping layout-pass items, empty or split boxes re-read from crops | 79.97 | +2.80 (1.42 to 4.23) |
| `wai_agent_full_sc3` | three chart samples, a vote per cell (self-consistency [4]) | 80.14 | +0.18 (-1.04 to 1.39) |

On test, the same pieces hold: `wai_agent_full` 77.12, and `wai_agent_v4`
(full, plus three-sample voting on tables as well as charts, plus the detector's
chart label as a second trigger for the page chart pass) 78.53, a paired +1.41
(0.87 to 2.01). `wai_agent_v5` (v4 with medium chart effort) won on dev
charts, 86.7 against 83.5 to 85.2 at xhigh, but scored 78.19 on test, a paired
-0.34 (-0.80 to 0.10): no difference, so v4 stays the headline.

Dev runs are one pass each. Tables swung 76 to 81 between single-sample runs,
which is why voting is on for tables in v4. The gap from dev to test (79.97 to
77.12 for `wai_agent_full`) is mostly layout and charts; the detector
thresholds were tuned on dev.

Detector ablations run without new model calls: `serve/relayout.py` attaches
detector boxes to existing raw outputs and re-scores the layout group
(`wai_agent_det_*` in `waiparse/pipelines.py`).

## What did not work

- **Chart RL on crops.** `train/configs/charts-v1.toml`: LoRA r32 on
  Qwen3.8-27B, GRPO advantages [5] with the CISPO loss [6] as ScaleRL runs it
  [7], reward = the share of ParseBench chart rules the reply passes, in the
  spirit of olmOCR 2's unit-test rewards [8]. Validation reward rose from 0.569
  to 0.624 by step 100, but ParseBench dev charts fell to 82.1 against the
  base's 83.5 to 85.2.
- **Chart RL on whole pages.** `train/configs/charts-pages-v1.toml` trains on
  synthetic report pages composed from ChartNet charts with the page chart
  prompt the agent uses. Validation reward rose from 0.539 to 0.559 by step 50;
  ParseBench dev charts at medium effort were 85.9 against the base's 86.7.
- **Medium chart effort** won on dev and did not hold on test (above).
- **A lower detector threshold** keeps raising the layout score (0.1 gives 83.5
  dev layout at 52 boxes a page against 35 in the ground truth) because the
  headline has no false-positive penalty. We stayed at 0.2 on purpose.

Both RL runs say the same thing: synthetic ChartNet charts do not teach what
ParseBench's report charts need. `train/README.md` has the launcher, the
checks to run before a launch, and how to resume and serve an adapter.

## Where to hill-climb next

- **Tables.** 80.5 here against 94.3 for Claude Opus 5.5. The largest gap to
  the best closed model.
- **Formatting.** 70.3, the lowest dimension. `modal run
  serve/bench.py::fmt_failures --spec wai_agent_v4:dev` prints the failing
  rules with the markdown around each.
- **The dev to test drop in layout and charts.** Tune the detector knobs
  (`det_*` in `agent.DEFAULTS`) with an interval, not a single dev run.
- **Chart data that looks like report charts** for RL: rendered charts with
  styled legends, data labels and small multiples, or rewards on real
  unlabeled pages through self-consistency.

## Cost and time

Modal on-demand prices ([modal.com/pricing](https://modal.com/pricing), read
2026-09-28): H200 $4.54 an hour, L4 $0.80 an hour. The vLLM app runs up to four H200 containers
(`DOCPARSE_MAX_CONTAINERS`) and scales to zero after ten idle minutes; the
detector runs up to four L4s; the benchmark runner is a CPU container. A
single-dimension dev run is the cheap loop; test and full are the expensive
ones. RL runs on four H200s ($18.16 an hour): `charts-v1` took about 1.6 minutes a
step, so its 100 steps were under three hours and about $50;
`charts-pages-v1` took about 3 minutes a step, about $45 for 50 steps.

## Traps

- **Reasoning parser.** vLLM needs `--reasoning-parser qwen3`, or the thinking
  text lands in the content and breaks the JSON the layout pass returns.
- **Model name.** ParseBench's Qwen layout adapter keys on the served model
  name starting with `qwen3.`, hence `--served-model-name qwen3.8-27b`.
- **pdfium is not thread-safe.** ParseBench runs documents in threads.
  `waiparse/render.py` takes a global lock and closes bitmaps and pages inside
  it; left to the garbage collector, the full-set run segfaulted.
- **Image size.** The chart RL stack starved at 2048 px pages (over half the
  rollouts timed out); page tasks are 1400 px. The agent reads charts at 2048
  px (86.7 dev charts against 82.5 at 1400), so evaluate an adapter at 2048.
- **Weight broadcast.** Loading thousands of inlined page images outlasts
  prime-rl's default 1200 s broadcast wait. Put `[weight_broadcast]
  timeout = 3600` at the top level of the config: prime-rl rebuilds the
  trainer and orchestrator sections from it and overwrites their own
  timeouts. Cap `env.taskset.max_tasks` too.
- **Router queue.** At 1,024 in-flight rollouts the vllm-router queue (100
  requests, 60 s) expired about half of them as "Request timed out"; the page
  config uses 128.
- **Long requests.** A Modal web endpoint answers a long request with a 303;
  follow redirects (`curl -L`).
- **`modal volume get` with an empty path** downloads the whole volume.
- **Use `--spawn` for RL, not `modal run --detach`.** A detached run dies with
  the local client.

## References

1. Zhang et al. ParseBench: A Document Parsing Benchmark for AI Agents. arXiv:2604.08538, 2026.
2. Cui et al. PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing. arXiv:2601.21957, 2026. PP-DocLayoutV3 is its layout stage; weights at `PaddlePaddle/PP-DocLayoutV3_safetensors`, Apache-2.0.
3. Kondic et al. ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding. arXiv:2603.27064, 2026.
4. Wang et al. Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv:2203.11171, 2022.
5. Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
6. MiniMax. MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention. arXiv:2506.13585, 2025.
7. Khatri et al. The Art of Scaling Reinforcement Learning Compute for LLMs. arXiv:2510.13786, 2025.
8. Poznanski et al. olmOCR 2: Unit Test Rewards for Document OCR. arXiv:2510.19817, 2025.
