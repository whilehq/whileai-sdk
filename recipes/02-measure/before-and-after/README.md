# Did my prompt rewrite really help?

Run the old prompt and the new one on the same questions, on a model on
your own machine, and get one verdict: the gain, its 95% range over tasks,
and PASS only when the range clears zero. Free: no key, no credits.

What you will learn: how to turn "the new prompt feels better" into a
paired number with an interval, why both prompts draw on the same seeds,
and what an honest null looks like. You need nothing for the offline run;
[Ollama](https://ollama.com) for a real model. Seconds offline; an external
case study ran this shape on `qwen3:4b-instruct` through Ollama on one
laptop in 22 minutes
([gentlyventures.com](https://gentlyventures.com/casestudies/whileai)).

## Run it

```bash
uv add whileai
wai compare --demo                         # the built-in scripted model, offline
cd recipes/02-measure/before-and-after
python run.py                              # the same check, in Python, on a fake model
python run.py --same                       # old prompt on both arms: the null
python run.py --model ollama:qwen3:4b-instruct   # a real model, free, on this machine
```

The command line takes your own files:

```bash
ollama pull qwen3:4b-instruct
wai compare --model ollama:qwen3:4b-instruct \
  --before before.txt --after after.txt --tasks tasks.jsonl --reward Numeric
```

| flag | default | what it does |
|---|---|---|
| `--model` | scripted | where the prompts run: `ollama:<model>`, `vllm:<model>@<url>`, any spec |
| `--k` | 4 | replies per task per arm in each run |
| `--runs` | 3 | times the whole eval runs per arm, so the report can measure its own noise |
| `--seed` | 0 | sampling seeds and bootstrap; same seed, same output |
| `--same` | off | the old prompt on both arms |

`tasks.jsonl` is one `{"id", "prompt", "reference"}` per line. `--reward`
is any `wai.verify` class (`Numeric`, `ExactMatch`, `Includes`,
`MathEqual`) or `module:function`.

## What you get

```text
before-and-after check: 24 tasks x 4 replies x 3 runs per arm, model scripted_model, temperature 0.7, seed 0 (same seed, same output)
  before 419b766a94cd  pass@1 0.55 [0.47..0.63]
  after  53313aa05fd5  pass@1 0.85 [0.80..0.90]

pass_at_1: moved (+0.306, 95% +0.253..+0.358, 24 paired tasks)
PASS
eval noise: run_std 0.015, a delta under 0.035 is noise (t(df=4)=2.78 x run_std x sqrt(1/3 + 1/3); 3 eval runs before, 3 after)
answered: 100.0% before, 100.0% after
  pass_at_1                    0.549 -> 0.854  +0.306 [+0.253..+0.358]  up  (24 paired)  noise<0.035
```

The number that matters is the gain, +0.306, with its 95% range
+0.253..+0.358: the range clears zero and the gain clears the eval's own
re-run noise (0.035, measured from the three runs a side), so PASS. The compare block is
`wai.compare` printing itself; nothing in this check computes a statistic
of its own. With `--same` the gain is +0.000 and the verdict is
NO DIFFERENCE, which is what a check that cannot be fooled by its own
noise has to say. Each arm got draw `j` of a question on the same seed,
so the sampler's luck is shared and the paired difference is the
prompt's.

Why three runs: one run per side is one draw of the eval, and
`wai.compare` reads a gain on one draw as INCONCLUSIVE, because it cannot
tell the prompt from the luck of that run. Three is the fewest re-runs a
spread can be read from. `--runs 1` is a third of the time and says so.
The case study's 22 minutes was one run; budget about three times that
for the default on the same laptop.

## Next

Swap in your prompts and your test set, then
`wai compare --model ollama:<your-model> ...`. The rows are on the report
(`report.before_rows`, `report.after_rows`) for `wai.pass_at`,
`wai.select` or `wai.harness.attribute`.
