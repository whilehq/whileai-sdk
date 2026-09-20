# Character training from a constitution

How a model talks when nobody told it how to talk. This example takes the
style section of the [OpenAI Model Spec](https://github.com/openai/model_spec)
as a constitution and runs it through the pipeline Anthropic describes for
Claude's character [1]
and [Maiya et al. 2025](https://arxiv.org/abs/2511.01689) open-sourced:
traits, prompts that exercise each trait, several replies per prompt, a
judge that reads the trait's principle, then preference pairs and SFT rows.
The measurement comes with it. A trait you cannot measure is a trait you
cannot train.

What you will learn: how a constitution becomes graded rows, how to check
the judge against the spec's own labels before reading a pass rate, where
preference pairs and SFT rows come from, and how to measure a trait before
and after training with a guard on the behaviors that must not regress.
You need nothing for the offline run; the live run needs an
OpenAI-compatible endpoint for the student and, ideally, a second one for
the judge.

## Run it

```bash
uv add whileai
cd recipes/03-select/character
python run.py                 # scripted student, offline, seconds
python measure.py --demo      # before vs after on the adversarial holdout
```

Output of `python run.py` with the defaults (`--seed 0 --k 4`). The
student and the judge are both scripted, so this is deterministic:

```
traits 8 | train 57 prompts x 4 = 228 rows | adversarial 120 | control 24 | spec 35
judge reference vs spec labels: agreement 1.00 (n=35, kappa 1.00)
pass@1 0.58 | pass^4 0.39 | pass@4 0.79 | headroom 0.21 | mixed prompts 23/57
  avoid_being_condescending                  pass@1 0.44 headroom 0.56  (4 prompts)
  avoid_sycophancy                           pass@1 0.73 headroom 0.27  (11 prompts)
  be_clear                                   pass@1 0.00 headroom 0.00  (4 prompts)
  be_rationally_optimistic                   pass@1 1.00 headroom 0.00  (4 prompts)
  be_warm                                    pass@1 0.50 headroom 0.00  (6 prompts)
  do_not_make_unprompted_personal_comments   pass@1 0.48 headroom 0.10  (12 prompts)
  love_humanity                              pass@1 0.50 headroom 0.50  (8 prompts)
  refusal_style                              pass@1 0.84 headroom 0.16  (8 prompts)
markers: no_filler 0.64 [0.57,0.70] | on_task 1.00 [1.00,1.00] | trait 0.58 [0.48,0.67]
controls on_task 1.00 | adversarial trait 0.25
corr(reward, reply length) +0.30 ok
pairs 20 (chosen longer 0.7) -> out/pairs.jsonl | sft 133 -> out/sft.jsonl
```

`python measure.py --demo` (`--seed 0`, k=4) plays the untrained and the
"trained" scripted student on the same 36 holdout and control tasks and
runs `delta_report(target="marker:trait", must_not_regress=["on_task",
"no_filler"])`:

```
marker:trait: moved (+0.500, 95% +0.408..+0.592, 30 paired tasks)
PASS
  pass_at_1                    0.375 -> 0.792  +0.417 [+0.319..+0.521]  up  (36 paired)
  marker:no_filler             0.681 -> 0.986  +0.306 [+0.229..+0.389]  up  (36 paired)
  marker:on_task               1.000 -> 1.000  +0.000 [+0.000..+0.000]  flat  (36 paired)
  marker:trait                 0.250 -> 0.750  +0.500 [+0.408..+0.592]  up  (30 paired)
```

Offline, the student replays the spec's own GOOD and BAD replies at a fixed
rate per prompt and the judge is a lookup against those labels. The numbers
are real; the model is not. `--after` plays a student that landed the
training, which is what the measure demo compares against.

## With a model

```bash
python run.py --model-url https://host/v1 --model qwen3-4b --key $KEY \
    --judge-url https://judge/v1 --judge-model phi-4 \
    --k 4 --write-prompts 8 --teacher
```

Same code, three changes: the student is sampled with the deployment prompt
only (`You are Sol, an assistant.`, no constitution), the judge is an LLM
that gets the principle in its system prompt, and `judge_vs_spec` becomes a
real number. `--write-prompts N` has the model write N more prompts per
trait, few-shot from the spec's, split train/holdout by hash. `--teacher`
samples one more reply per train prompt with the constitution in the system
prompt, the distillation teacher of Maiya et al.; pairs then carry
`same_policy=false` where the chosen side is the teacher's.

Leave `--judge-url` off and the model grades itself. The report still runs;
the `judge_vs_spec` line is where self-preference shows up: a model grading
its own replies favors them [2].

One live run, hosted Qwen3-4B-Instruct as the student and hosted Phi-4 as
the judge, `--no-texture --k 4`, 239 rows in 148 seconds:

```
judge llm:microsoft/phi-4 vs spec labels: agreement 0.69 (n=35, kappa 0.40)
pass@1 0.78 | pass^4 0.73 | pass@4 0.80 | headroom 0.02 | mixed prompts 1/15
markers: no_filler 1.00 [1.00,1.00] | on_task 0.93 [0.80,1.00] | trait 0.85 [0.65,1.00]
controls on_task 0.88 | adversarial trait 0.97
corr(reward, reply length) -0.16 ok
pairs 1 (chosen longer 0.0) -> out/pairs.jsonl | sft 47 -> out/sft.jsonl
```

The rows from that run are on Hugging Face as
[while-ai/character-training-model-spec](https://huggingface.co/datasets/while-ai/character-training-model-spec)
(splits `train`, `holdout`, `eval`; the `eval` split is the spec's labeled replies
with `gold_reward`) and on the [platform catalog](https://huggingface.co/while-ai).

Two things that run says, neither visible without the spec rows and the
markers:

- **The judge is lenient.** Phi-4 passed 10 of the spec's 20 BAD replies
  (`pass_when_gold_fail` 0.50). Those are exactly the rows a preference set
  would train toward. Fix the judge prompt, or use a stronger judge, before
  reading the pass rates.
- **The spec is this model's default character.** Qwen3-4B lands the
  spec's traits 78% of the time and holds them under "drop the act" (0.97).
  One mixed prompt, one pair: there is nothing here to train on. That is
  the expected result for an instruct model on the industry-default spec.
  A distinct persona, or `--write-prompts` for harder situations, is where
  the contrast comes from. Writing the constitution is the small part.

## What each number is for

| line | call | what to do with it |
|---|---|---|
| `judge vs spec labels` | `judge_agreement` on the spec's GOOD/BAD replies | below about 0.8, fix the judge before reading anything else |
| `pass@1` per trait | `pass_at` | a trait at 0.00 or 1.00 gives no pairs; write harder or easier prompts for it |
| `headroom` | `pass@k - pass@1` | what a grouped update can learn; zero means demonstrations, not rollouts |
| `mixed prompts` | `group_signal` | the prompts that produce pairs; everything else is supply for SFT or nothing |
| `markers` | `marker_summary` | `trait` is the target; `on_task` and `no_filler` are the guards |
| `adversarial trait` | markers on the holdout | how much of the trait survives "drop the act"; the number training should move |
| `corr(reward, reply length)` | `reward_correlations` | above 0.3 the judge is paying for length; `length_match` on the pairs is the second line of defense |
| `chosen longer` | `build_preference_pairs` | the spec's own GOOD replies run longer than its BAD ones 70% of the time; DPO will learn length first if you let it |

## The pipeline

| step | source | here |
|---|---|---|
| constitution | Askell: "constructing character traits that the model should have" [1]; the principles come from a written constitution [3] | `from_model_spec.py` parses each style heading into a principle and its GOOD/BAD comparisons; `spec_id` on every row names the heading |
| prompts | Askell: "get the model to generate queries that humans might give it that are relevant to that trait" [1] | the spec's prompts, four wordings each; `--write-prompts` for model-written ones |
| replies | sample k per prompt, the group of GRPO [4] and the candidates of rejection sampling [5] | `k` replies under the deployment prompt; `--teacher` for constitution-prompted chosen sides |
| judge | a separate judge that reads the principle [3], length-neutral [6], checked against labels | principle as privileged context, one few-shot comparison from another prompt of the same trait, `judge_agreement` on the spec rows |
| markers | phrase monitors, "removing common phrases like `Certainly`" [1], because a proxy reward drifts toward cheap features [7] | `trait`, `on_task`, `no_filler` |
| pairs and SFT | on-policy pairs for DPO [8], length-matched so length is not the first thing learned [6] | `build_preference_pairs(length_match=True)` and `export_preference` with the deployment prompt; `export_training` on passes with a loss mask |
| before and after | post-training on one thing forgets others [9] | `measure.py`: `delta_report(target="marker:trait", must_not_regress=["on_task", "no_filler"])` |

The reward on a trait prompt is `trait AND on_task`. The spec is explicit
that style "enhances rather than distracts from" helpfulness, and the
steroids example in the character chapter makes the same point: every persona
still refuses [1]. A reply
that has the character and drops the task is a 0.

## What is not here

- **A trainer of your own.** `out/pairs.jsonl` is `prompt`, `chosen`,
  `rejected` as message lists, what a DPO trainer reads; `out/sft.jsonl`
  carries a loss mask. `recipes/04-train/identity/train_modal.py` is a LoRA
  pattern to copy. The hosted path is `wai.train(ds_id, method="dpo")` on
  the pushed rows.
- **Maiya's third stage.** Introspective SFT (the trained model writing
  about its own values) needs the trained model. Run it after DPO with the
  same judge.
- **Persona vectors, activation capping, persona subnetworks** [1].
  No gradient, no data; a different tool.
- **A prompt-disjoint holdout offline.** The adversarial set reuses the
  train prompts with a "drop the act" suffix (Maiya's robustness test).
  `--write-prompts` gives a real one.

## Files

| file | what |
|---|---|
| `from_model_spec.py` | spec markdown to `constitution.json`; one explicit-content example skipped by default |
| `constitution.json` | 8 traits, 15 comparisons, 15 GOOD and 20 BAD replies (the 35 `spec` rows), with the spec commit |
| `run.py` | tasks, students (scripted or live), judges (reference or LLM), markers, pairs, SFT, report |
| `measure.py` | before/after delta on `holdout.jsonl` |
| `out/` | `rows.jsonl`, `pairs.jsonl`, `sft.jsonl`, `holdout.jsonl`, `report.json` |

Tests: `pytest tests/api/test_character_example.py tests/recipes/test_character.py -q`.

## References

1. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Model Character and Products*.
2. Panickssery, A., Bowman, S. R., Feng, S. LLM Evaluators Recognize and Favor Their Own Generations. arXiv:2404.13076, 2024.
3. Bai, Y. et al. Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073, 2022.
4. Shao, Z. et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
5. Yuan, Z. et al. Scaling Relationship on Learning Mathematical Reasoning with Large Language Models. arXiv:2308.01825, 2023.
6. Zheng, L. et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023. arXiv:2306.05685.
7. Gao, L., Schulman, J., Hilton, J. Scaling Laws for Reward Model Overoptimization. ICML 2023. arXiv:2210.10760.
8. Rafailov, R. et al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS 2023. arXiv:2305.18290.
9. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Regularization*.
