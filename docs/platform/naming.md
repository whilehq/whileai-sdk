---
title: "Name things after your work"
sidebarTitle: "Naming"
description: "The platform shows the names your code sends. A coding agent names the agent, behaviors, versions and experiments in the team's own words, and the test by its content."
---

Every name on [withwhile.com/platform/runs](https://withwhile.com/platform/runs) comes from your code, and the page marks it so. The coding agent that writes the `track(...)` calls decides what the team reads for months. One rule covers it: **read the repo, then use its words.** The package name, the prompt files, the policy doc, the existing test names, the deploy tags and the changelog already hold the vocabulary. The SDK's example names (`refund-agent`, `refund_policy`, `v1`) are for the playbooks, not for a real repo.

| Thing | Rule | From the repo | Not this |
|---|---|---|---|
| Agent, `track("…")` | the product or service the agent is, kebab-case, one per deployed system; the model is a field, not part of the name | `checkout-support` | `my-agent`, `refund-agent-haiku` |
| Behavior, `Behavior(name=)` | a verb phrase from the team's policy or spec, snake_case, what a pass looks like | `refunds_when_eligible`, `escalates_over_limit` | `test1`, `quality` |
| Version, `tracked.run("…")` | what the team already calls a release: git tag, PR, date or prompt label | `2026-09-20`, `pr-412`, `policy+dates` | `v1`, unless the team ships as v1 |
| Harness variant, `Harness(label=)` | `prompt@model`; the prompt part is the prompt file or the change | `policy+dates@claude-sonnet-5` | `variant-3` |
| Test, `test_version=` | `"t-" + sha256(asks)[:8]`, the SDK's one fixed convention, because the name must change when the asks do | `t-5f57ed8d` | `final`, `v2` |
| Method, `method=` | as the SDK spells it | `eval`, `GRPO`, `SFT`, `DPO` | `training` |
| Experiment, `tracked.experiment(question=)` | the question in the team's words, one sentence | "Does the dates rule cut wrong refunds?" | "Experiment 3" |

Three checks before the first post:

- A teammate who has never opened the platform would recognise the agent id.
- Every behavior name appears, in some form, in the repo's own docs or tests.
- Versions sort the way the team's releases sort.

Why the test is the exception: a score is comparable only with the setup held constant (rlhfbook.com, "Evaluation"), so the test's name is its content and changes on its own when an ask changes. Everything else is a label a person reads, and the person is on the team.

The `AGENTS.md` block that `wai init` writes carries this in one line: *you know this repo best; name the agent, behaviors, versions and experiments in its words, and the test by its content.*
