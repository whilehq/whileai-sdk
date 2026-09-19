# Constitution

What `whileai` is, what we believe, and how that shows up in the code.
Read it before you add a public name, write a page, or run a recipe. The
routines that maintain this repo read it too.

## What we are

`whileai` is a scientific post-training library for language models: SFT
and RL, on open models, with the measurement that says whether training
helped. Simulate, grade, measure with intervals, select, train, prove on a
held-out set, serve, and feed the new traces back in. Build self-improving
systems.

It is for AI researchers, ML engineers and applied-AI developers, and the
goal is that it sits in every applied-AI and research department the way
PyTorch does. The platform (`whileai.platform`) is a separate, optional
service for hosted training and serving. The library needs no account.

## What we believe

1. **Repeatable science.** A number is a result only with its interval, its
   noise floor, its seed and the versions that produced it. A mean alone is
   not a result. A flat result is a result. (`pass_at`, `eval_variance`,
   `delta_report`, `holdout_size`.)
2. **Replicated papers are the proof.** We show the library works by
   reproducing recent post-training research in it, one recipe per paper,
   under an hour on one GPU, with the number it moved and the number it did
   not. Every reproduced paper is a post. The proof point is the recipe,
   not the pitch. (`recipes/papers/`.)
3. **The book is the map, the paper is the citation.** Every default is
   named, sourced and tunable from the call. [rlhfbook.com](https://rlhfbook.com)
   (Lambert) is the map of the field; the originating paper is the
   reference. A default with no source says "convention, untested".
   (`defaults.py`, `scripts/check_no_hardcoding.py`.)
4. **Bring your own keys.** Your models, your compute, your accounts.
   Modal and Prime Intellect are first-class: a `whileai` environment
   becomes a `verifiers` environment and back, selected rows become a
   trainer's prompt set, eval results flow back into measurement with
   intervals. Nothing in the loop requires our hosting.
5. **Developer ergonomics are the product.** The code reads like PyTorch,
   DSPy and Unsloth: one import, objects carry configuration, calls carry
   data, reports print themselves, errors name the fix, and a first-time
   reader can guess the next line. Rigor lives behind a default, never
   behind a flag. (`docs/reference/style.md`, the ratchet test.)
6. **Plain words, then the mechanism, then the proof.** Every page, every
   docstring, every README section in that order. The first thing a reader
   sees is the loop as five lines, one per step, each the step's name and
   one sentence saying why the step exists in the reader's own words
   ("Measure. One run proves nothing. Ask whether a change is real or
   noise before you ship it or train on it."). Nothing goes in front of
   that list: a manufactured hook ("Break it. Score it. Prove it.") and a
   paragraph about the problem both read worse than the list, and were
   cut the same day they shipped. Book vocabulary (interval, rollout,
   band, gradient) stays in the docstring that cites the chapter, never in
   a lead sentence or a public name. The step names are Simulate, Grade,
   Measure, Select, Train. Measure is evaluation with intervals and Select
   is data curation; they are not merged, and neither is renamed to Eval.
7. **Mass experimentation.** A PhD or an engineer runs many experiments
   from one import, on their own compute, and every run leaves a record
   that a person can decide from.
8. **Never big-bang.** The internals carry the science and the tests.
   Change the front door, migrate callers mechanically, keep the old name
   working for one release with a warning that says the new one.
9. **One name.** The company is While, the package is `whileai`, the import
   is `import whileai as wai`, the hosts are withwhile.com,
   app.withwhile.com, api.withwhile.com and docs.withwhile.com, the
   variables are `WHILEAI_*`, the config dir is `~/.whileai`. ZeroProof
   was the name before 2026-09-16; the cutover finished on 2026-09-19 and
   nothing new is written under it: no code, page, prompt, routine,
   dataset card or post. What still carries the old name is wire protocol
   and infrastructure that would break users if renamed (the `zp_` key
   prefix, `zeroproof.*` span attribute keys, Modal app hostnames, volume
   and table names) and the history in `CHANGELOG.md`. Those are pinned,
   not permitted: `scripts/old_name_baseline.json` counts them per file,
   a count may fall and never rise, and a new file may not add one. The
   `ZEROPROOF_*` variables and `~/.zeroproof` are not read.

## How it shows up

| Belief | Where it is enforced |
|---|---|
| Repeatable science | `recipes/papers/check.py` refuses "moved" without an interval that excludes zero, three base re-runs, a clean holdout, and a proxy-vs-target verdict |
| Replicated papers | `recipes/papers/README.md`: one paper, one recipe, one command, one `post.md` |
| Sourced defaults | `scripts/check_no_hardcoding.py` in CI; `tests/api/test_readme_defaults.py` |
| Ergonomics | `docs/reference/style.md`; `tests/api/test_style_ratchet.py` pins the retired shapes |
| Docs order | `docs/` on Mintlify; the docs routine's one PR a day; the five-line loop list on the website home (`components/quickstart.tsx` in whilehq/website) is the reference wording |
| Bring your own keys | `wai.configure(agent=, judge=, api_key=)`, backend objects whose repr names the key source; the Modal and Prime Intellect researcher routines run on their own accounts twice a day |
| One name | `scripts/check_old_name.py` in CI lint pins the count of the old name per file from `scripts/old_name_baseline.json`; the docs, site and style routines fix any old-name string in a file they touch |

## Who reads this

People: contributors, before their first public name. Agents: the style
guide routine, the docs and site routines, and the Modal researcher
routine (the Prime Intellect one when it has a key), at the top of every
run. When this file and another file disagree, this file wins and the
other file gets a PR.
