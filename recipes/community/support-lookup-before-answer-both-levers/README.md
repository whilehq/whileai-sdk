# The support agent that would not look it up, and which lever fixed it

**Seat:** a post-training engineer on a team that ships one customer support
agent, trying to make it stop answering account questions out of its own head.

**Behaviour:** *looks it up before answering* — when a customer asks about
their own account, order, reservation or bill, call the right tool first
instead of answering, or asking again for something the customer already gave.

**Capability twin, reported beside it every time:** *stays in scope* — on the
asks the reference agent correctly refused (another customer's data, a policy
the tools do not carry, a list of everyone's lines), still do not call a tool.
Neither number means anything alone: an agent that always calls a tool scores
1.00 on the behaviour and 0.00 on the twin, and an agent that never calls one
does the reverse.

**Method:** the harness-and-weights grid, reused rather than reproduced —
[`recipes/papers/harness-and-weights`](../../papers/harness-and-weights)
(HASE, Luo et al., [arXiv:2607.03935](https://arxiv.org/abs/2607.03935); SIA,
Hebbar et al., [arXiv:2605.27276](https://arxiv.org/abs/2605.27276); Prime
Agent, Karten et al., [arXiv:2608.23552](https://arxiv.org/abs/2608.23552)).
That recipe put `both` six points over `weights` (+0.060 [+0.028, +0.095]) on
quant coding tasks and found the harness explained 95% of the spread. No
community recipe had applied it to an agent behaviour, so the budget went to
the application instead of a second reproduction.

**The change:** four cells on one frozen holdout. `neither` is the base weights
under the deployed policy prompt. `harness` is the base weights under a
searched harness — a skills text, no GPU at all. `weights` is the busy
engineer's default, LoRA SFT on the good rows under the deployed prompt.
`both` is the same SFT under the searched harness. `wai.harness.attribute`
reads the 2x2 and says which lever moved it.

## The agent and its traffic

`while-ai/tau2-simulated` (telecom, retail, airline: 1,057 conversations,
every one graded `reward = 1` by the published grader). The traffic is the
environment — the policy, the 13 to 16 tools, the customer's turns. The agent
is mine: `Qwen/Qwen3-4B`, offline, on my own Modal.

`prep.py` cuts each conversation at the moment the reference agent acted and
keeps what it did: 927 decision points where it called a tool (30 distinct
tools) and 130 where it correctly answered without one. `split.py` splits by
scenario, not by row — 400 of 654 scenario ids repeat across rows, so a row
split would put near-copies on both sides.

## Run it

```bash
pip install whileai 'modal[api-proxy-support]'
cd recipes/community/support-lookup-before-answer-both-levers

python run.py --dry-run              # offline: fixture tasks, scripted agent, no key, no GPU
python prep.py && python split.py    # published traces -> decision points, split, decontaminate
modal run --detach support_modal.py  # search, two arms, the grid
modal volume get support-lookup-runs / out
python run.py --analyse              # compare, attribute, results.json

modal deploy serve_modal.py          # vLLM, --enable-lora, scale to zero
python fresh_traffic.py --url <printed url> --arm both
modal app stop support-lookup-serve
```

| flag | default | what it does |
|---|---|---|
| `--dry-run` | off | offline: 6 fixture decision points, a scripted agent, no network |
| `--analyse` | off | read `out/` and write `results.json` |
| `SUPPORT_HOLDOUT` | 320 | held-out decision points played per cell |
| `SUPPORT_GATE` | 160 | train-split tasks the harness search runs on |
| `SUPPORT_K` | 2 | rollouts per task; the interval is over tasks |
| `SUPPORT_BASE` | `Qwen/Qwen3-4B` | the base model |
| `SUPPORT_GPU` | `L40S` | the GPU |

## Result

<!-- RESULTS -->

## Checks

<!-- CHECKS -->

## From paper to production

<!-- RANKED -->

## Learned

<!-- LEARNED -->
