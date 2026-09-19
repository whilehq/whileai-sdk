---
title: "Character training"
sidebarTitle: "Character training"
description: "Change the weights so a model has a stable way of talking without a system prompt: sources, the recipe, and what to measure."
---

Character training changes the weights so a model has a stable way of
talking without a system prompt. It is the same machinery as any
post-training run [1], aimed at the manner of a reply, and mostly a data
pipeline: which phrases never appear, which replies get chosen. Worked
example:
[`recipes/03-select/character`](https://github.com/whilehq/whileai-sdk/tree/main/recipes/03-select/character)
(offline by default).

<img className="block dark:hidden" src="/figures/character-pipeline-light.svg" alt="Constitution, prompts, k replies under the deployment prompt, a judge that alone sees the principle, markers, then pairs, SFT, train and a paired delta" />
<img className="hidden dark:block" src="/figures/character-pipeline-dark.svg" alt="Constitution, prompts, k replies under the deployment prompt, a judge that alone sees the principle, markers, then pairs, SFT, train and a paired delta" />

## What the research says

Character training is the subset of post-training designed around
crafting traits within a model, and fine-tuning on trait data beats
prompting and activation steering for robustness [1]. Anthropic's process,
as Amanda Askell describes it: write the traits, generate queries per
trait, generate responses, rank by the trait. Much of the work is
controlling the language in the data [2]. The OpenAI Model Spec gives each
trait as a principle plus GOOD and BAD comparisons on real prompts, which
is a constitution with labeled pairs attached [3]. Maiya et al. build DPO
pairs (chosen versus rejected, no reward model) from a teacher with the
constitution in its prompt against a student without, and evaluate
revealed trait words, robustness to "ignore role-play and respond
genuinely", and whether capabilities stayed unchanged [4].

## The recipe

1. **Constitution.** One principle per trait, plus labeled examples
   (`prompt`, `good`, `bad`) if you have them.
   `recipes/03-select/character/from_model_spec.py` builds one from the spec.
2. **Prompts.** Situations that make the trait matter, with wording variants
   so the judge grades the trait, not the phrasing.
3. **Replies.** `k` per prompt under the deployment prompt only, which names
   the persona and nothing else. Constitution in the sampling prompt means
   you are measuring prompting, not character.
4. **Judge.** The principle goes in the judge's system prompt and nowhere
   else (`Task.privileged.principle`). Different family from the policy.
   Grade the spec's own GOOD/BAD replies: below 0.8 agreement or 0.6
   kappa, fix the judge first.
5. **Markers.** `trait` and `on_task` from the judge, `no_filler` from a
   phrase list the judge never sees. Reward is `trait AND on_task`.
6. **Pre-flight.** `pass_at` per trait. A trait the student always lands, or
   never, yields no pairs; the mixed prompts are the signal.
   `reward_correlations` warns above 0.3 on length.
7. **Pairs and SFT.** `build_preference_pairs(rows, length_match=True)`, then
   `export_preference(pairs, "pairs.jsonl", system_prompt=DEPLOY_PROMPT)`;
   `export_training` on the passes.
8. **Train.** `wai.train(dataset_id, method="dpo")`, or any DPO trainer on
   `pairs.jsonl`.
9. **Measure.** The same prompts with a "drop the act" suffix, plus plain
   tasks the persona must not distort.
   `delta_report(before, after, target="marker:trait", must_not_regress=["on_task", "no_filler"])`
   gives the headline with an interval and fails on a regression.

## Run it

Offline, seconds; student and judge are scripted, so the numbers are real
and the model is not.

```bash
cd recipes/03-select/character
python run.py             # constitution -> rows -> pairs and SFT
python measure.py --demo  # before vs after on the adversarial holdout
```

`run.py` with the defaults (`--seed 0 --k 4`):

```text
traits 8 | train 57 prompts x 4 = 228 rows | adversarial 120 | control 24 | spec 35
judge reference vs spec labels: agreement 1.00 (n=35, kappa 1.00)
pass@1 0.58 | pass^4 0.39 | pass@4 0.79 | headroom 0.21 | mixed prompts 23/57
markers: no_filler 0.64 [0.57,0.70] | on_task 1.00 | trait 0.58 [0.48,0.67]
controls on_task 1.00 | adversarial trait 0.25
corr(reward, reply length) +0.30 ok
pairs 20 (chosen longer 0.7) -> out/pairs.jsonl | sft 133 -> out/sft.jsonl
```

`measure.py --demo`, the headline lines:

```text
marker:trait: moved_unreplicated (+0.500, 95% +0.408..+0.592, 30 paired tasks)
PASS
  pass_at_1                    0.375 -> 0.792  +0.417 [+0.319..+0.521]  up  (36 paired)
  marker:no_filler             0.681 -> 0.986  +0.306 [+0.229..+0.389]  up  (36 paired)
  marker:on_task               1.000 -> 1.000  +0.000 [+0.000..+0.000]  flat  (36 paired)
  marker:trait                 0.250 -> 0.750  +0.500 [+0.408..+0.592]  up  (30 paired)
```

`moved_unreplicated` means one run per side; `simulate(tasks=..., runs=3)`
before calling it proven. `on_task` is 1.0 on both sides, so that guard
cannot fail.

## The rows from one run

The live run (hosted Qwen3-4B-Instruct student, Phi-4 judge) is public:
[while-ai/character-training-model-spec](https://huggingface.co/datasets/while-ai/character-training-model-spec),
splits `train` (60), `holdout` (144), `eval` (35 spec replies with
`gold_reward`). Grade `eval` with your judge first: Phi-4 passed 10 of the
20 BAD replies (agreement 0.69, kappa 0.40) and failed that check.

## Things that go wrong

- **The judge likes long replies.** In the spec's comparisons GOOD is longer
  70% of the time; hence length-matched pairs and the correlation line.
- **The judge is the policy.** A model prefers its own writing [5, 6], so
  the pairs encode the model's taste.
- **Character costs helpfulness.** `on_task` is a hard guard; the controls
  carry no trait marker.
- **No contrast.** pass@1 of 0 or 1 yields nothing to pair. Use a teacher
  for the chosen side and accept off-policy pairs (`same_policy=false`).
- **The holdout is the training set.** Adversarial variants of train prompts
  test robustness, not generalization. Split by hash; `decontaminate`
  checks the overlap.

## What the SDK does not do

Persona vectors, activation capping, persona subnetworks, Maiya's
introspection stage. The SDK makes the rows, pairs, judge check and
before/after; `pairs.jsonl` feeds any trainer.

## References

1. Lambert, N. Reinforcement Learning from Human Feedback. arXiv:2504.12501, 2025. Chapter *Model Character and Products*.
2. Anthropic. Claude's Character. 2024. [anthropic.com/research/claude-character](https://www.anthropic.com/research/claude-character).
3. OpenAI. Model Spec. [github.com/openai/model_spec](https://github.com/openai/model_spec).
4. Maiya, S. et al. Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI. arXiv:2511.01689, 2025.
5. Panickssery, A., Bowman, S. R., Feng, S. LLM Evaluators Recognize and Favor Their Own Generations. arXiv:2404.13076, 2024.
6. Zheng, L. et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023. arXiv:2306.05685.
