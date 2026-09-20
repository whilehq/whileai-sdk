# Verifiers: verifiable rewards

A verifier is a reward that is a program, not an opinion, the reward of
reinforcement learning with verifiable rewards [1]. It reads a rollout and
returns pass, fail, or a partial score, with no model call. Every verifier
honors the judge contract, so it plugs into `data.grade(judge=...)`,
`evaluate`, `optimize` and a gated `push` exactly where an LLM judge would.

From the repo root:

```bash
python recipes/01-simulate/verifiers/run.py
```

No key, no model, seconds. What you will learn: where a verifier reads the
candidate and the gold, how to compose checks, and how a verifier's rows
feed `optimize` and a gated `push` unchanged. The script shows math (`MathEqual`),
an answer-and-format gate (`All([...])`), code execution against hidden tests
(`CodeExec`), a JSON-schema check (`JSONSchema`), and then hands the math rows
to `wai.optimize(mode="rl")` and `wai.training_rows` to show where the answer
key stops travelling.

## The pieces

```python
from whileai.simulations.verify import (
    ExactMatch,
    Includes,
    Regex,
    MultipleChoice,  # text
    Numeric,
    MathEqual,  # math
    JSONValid,
    JSONSchema,
    JSONField,  # structured output
    CodeExec,  # run code against tests
    All,
    Any,
    Weighted,
    verifier,  # compose / wrap
)
```

- **Where the answer comes from.** The candidate is the rollout's
  `final_text` (or the last assistant turn). The gold is read from the row's
  `privileged.reference` — which the training export never projects, so the
  answer key cannot leak into a training file — with flat fields (`answer`,
  `target`, `solution`, ...) as a fallback. Point any verifier at another
  column with `field=`.
- **Compose.** `All` needs every check to pass (right answer *and* right
  format), `Any` needs one, `Weighted` is a graded rubric in [0, 1].
- **Your own.** `@verifier def f(candidate, reference, row): ...` returns a
  bool or a score, or a `(score, reason)` pair.
- **Tool calls.** `wai.verify.tool_calls(row)` (or one message) returns the
  calls as `ToolCall(name, arguments, id)` with `arguments` always a dict,
  from a rollout row's flat `{"name", "arguments"}` and from an exported
  row's OpenAI wire shape alike, so a reward that checks "did it open with
  `get_order`" scores the same rows the same wherever it runs.

## In the loop

The gold travels with the task, not the spec: `simulate()` writes prompts,
rollouts and world state, never an answer key, so a verifiable task set is
rows you bring that already carry `privileged.reference` (or
`privileged.tests` for `CodeExec`). Score them with the verifier, run the
RL gates (reward band, unanimous groups, duplicates), push:

```python
import whileai.simulations as wai
from whileai.simulations.score.judging import run_judge
from whileai.simulations.verify import MathEqual

rows = [...]  # k rollouts per prompt, each with privileged.reference
scored = run_judge(rows, MathEqual())  # the verifier IS the reward
rows, _ = wai.optimize(scored, mode="rl")  # GRPO data, gradient checked
wai.push_rows(rows, "math-rl-v1", gate=True, mode="rl")
```

On a `SimulationData` the same step is `data.grade(judge=MathEqual())`. The
optimized rows still carry `privileged` (they are SDK rows, and a verifier
has to be able to re-score them); `wai.training_rows(rows)` is the export
that never projects it, which is what `run.py` prints at the end.

## Code execution safety

`CodeExec` runs the candidate in a fresh subprocess with isolated mode, a
private temp directory, a wall-clock timeout, and CPU/memory caps on POSIX.
That stops runaway loops and accidents. It is **not** a security boundary
against hostile code — for untrusted policies, run the verifier inside a
container or the hosted sandbox.

## References

1. Lambert, N. et al. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124, 2024.
