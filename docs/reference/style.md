---
title: "Style: how the language reads"
sidebarTitle: "Style"
description: "The coding standard for every public name: PyTorch and DSPy ergonomics, one import, objects carry configuration, calls carry data, reports print themselves."
---

`whileai` is a scientific SDK ([CONSTITUTION.md](https://github.com/whilehq/whileai-sdk/blob/main/CONSTITUTION.md)
says what that means). The code a user writes with it should read
the way PyTorch and DSPy read: a few nouns, a few verbs, objects that carry
their configuration, and one line per idea. This page is the coding
standard for every public name in the package. New code follows it. Old
code is brought under it one PR at a time, and the ratchet in
`tests/api/test_style_ratchet.py` refuses any PR that moves the other way.

The two models we copy:

- **PyTorch.** One universal noun (the tensor), and every op returns one.
  Stateful things are objects: `nn.Module` holds config in `__init__` and
  does its one job in `forward`; `optim.SGD(params, lr=)` then `.step()`.
  Namespaces are few and orthogonal: `torch`, `torch.nn`, `torch.optim`.
  Usability over cleverness, Python first, explicit over implicit
  ([PyTorch design philosophy](https://pytorch.org/docs/stable/community/design.html)).
- **DSPy.** Fifteen names at `dspy.*`. `dspy.configure(lm=)` once, then
  modules: `dspy.Predict(signature)`, `dspy.Evaluate(devset=, metric=)`,
  `optimizer.compile(program, trainset=)`. Behaviour is declared
  (`"question -> answer"`), not toggled with flags.

## The target front page

This is what the first twelve lines of the README should be. It is the bar
every public call is measured against. It runs a model on both sides, so it
needs `WHILEAI_API_KEY` in the environment, or `wai login`, plus the key
for whichever provider the specs name.

```python
import whileai as wai

wai.configure(agent="openai:gpt-4.1-mini", judge="anthropic:claude-haiku-4-5")

data = wai.simulate(tools=TOOLS, system_prompt=POLICY, mode="rl", repeats=8)
scored = data.grade(wai.verify.MathEqual())  # any judge object or callable
print(scored.pass_at)  # pass@1 0.67 [0.55..0.78] ...

print(wai.judge_trust(scored.rows))  # kappa vs people, length bias
rows = scored.select(mode="rl")  # 20..80% band, drop unanimous groups
rows.export("train.jsonl")  # or rows.push("my-agent-rl-v1")
```

Every line above runs today. Three of them did not when this page was
written: `pass_at` is a property, not a call; `judge_trust` takes rows and
a judge, and its `gold=` names the label key rather than the labels; and
`export` takes a path, with `format="trl"` for the shape `SFTTrainer`
loads. A bar nobody can run is a wish, so the snippet is kept executable
and the delta below says what is still missing.

Today the same program is `import whileai.simulations as wai`, a
`(rows, report)` tuple out of `optimize`, and `format_*` twins to print
anything. The rules below are the delta.

## Rules

Each rule names the PyTorch or DSPy habit it copies, then what it means
here. "Public" means any name in an `__all__` or documented under
`docs/api/`.

**1. One import, one prefix.** `import whileai as wai`. The top level
holds the loop verbs (`simulate`, `grade`, `judge_trust`, `select`,
`train`, `serve`), the nouns they pass around (`Rows`, `Report`, `Judge`,
`Verifier`), `configure`, and the sub-namespaces `wai.verify`, `wai.data`,
`wai.platform`. Under thirty names. Everything else lives one dot down,
grouped by stage, never by implementation file. `whileai.simulations` is
the legacy path and gains no new names. *(torch / torch.nn / torch.optim;
dspy.\* is fifteen names.)*

The alias is the mark, so it is also the only import shape anyone writes.
"while" is heard as "whale", wai the whale is While's mark, and the
definition sentence is reused verbatim wherever a person or a machine
asks: **wai is While's whale and the alias of the whileai SDK: `import
whileai as wai`.** Every example shows that line and no other: the README,
a module docstring, a docstring example, a recipe, a skill, an error
message that quotes code. `from whileai import x`, a bare `import whileai`
and `import whileai.simulations as wai` are style findings, and
`tests/api/test_alias_surface.py` counts them the way the ratchet counts
the retired shapes. Nothing else in the package is named `wai`: not a
module, a class, a CLI command or a flag, and the alias is lowercase
always. The places that state it, because a search or an answer engine
reads them before a reader does: the README banner `alt` text, the first
README paragraph, `whileai.__doc__`, the PyPI description and keywords,
and the GitHub repository topics. *(one shape per library: every code
example on dspy.ai opens `import dspy`, every PyTorch page `import
torch`.)*

A name a page spells `wai.X` resolves on `wai`. One import shape means
`wai.` is the only prefix a reader can type, so a page that writes
`wai.eval_variance` for a name that is not there hands the reader an
`AttributeError` and no path to the call. Two lists do two jobs, and
neither is the other: `__all__` is the front door, under thirty names, the
loop and its nouns, and the ratchet pins its size; `_LAZY` in
`whileai/__init__.py` is what the one import reaches, and it is already
larger (the platform client's own names live there, documented under
`wai.platform`). A name enters `_LAZY` when a doc, a recipe or a skill
writes it as `wai.X`, and enters `__all__` when it is also rule 5 clean
and another name comes off. A call listed in one table beside front-door
calls makes the same promise a front-door call makes, whichever list it
sits on. The count of `wai.X` spellings under `docs/`, `recipes/` and
`skills/` that do not resolve is not pinned yet; most of them are in files
that rebind `wai` to `whileai.simulations`, so the alias ratchet has to
fall first.

The first README paragraph also carries the ownership sentence once,
verbatim: **You own the model, the data and the weights: the datasets are
built from your production traces, the model is an open model post-trained
with SFT and RL, and the trained weights are yours to download and serve
anywhere.** It lives there and nowhere in code. A docstring and an error
message carry mechanism and a citation, never a thesis; the API backs the
sentence with ergonomics instead (a training call says where the weights
landed, keys ride on `configure()` and `context()`, and no training path
needs a hidden platform key).

**2. Objects carry configuration; calls carry data.** Anything a user sets
up once and applies many times is a class: judges, verifiers, selectors,
trainers, exporters. The constructor takes the configuration; one verb
method, or `__call__`, takes the rows. `Verifier` already does this. A
function is for a pure transform over rows with no reusable setup.
*(nn.Module.\_\_init\_\_ then forward; dspy.Evaluate(devset=, metric=)(program).)*

```python
# no
scored = grade(rows, model="...", rubric=r, temperature=0, field="answer", n=3, ...)
# yes
judge  = wai.Judge(model="...", rubric=r, temperature=0)
scored = data.grade(judge)
```

**3. Eight parameters.** A public function, method or constructor takes
at most eight, keyword-only past the first. If it needs more, it is two
things: split the object, or accept a typed options object. `simulate`
takes forty-five today; that is the ratchet's starting line, not a
licence, and a call already over the cap does not get to keep growing:
the ratchet pins the widest signature and the total overage as well as
the count. *(optim.AdamW has eight; dspy.Predict has three.)*

**4. One noun flows through every stage.** `Rows` (today `SimulationData`
and `ScoredData`) is the tensor. Every stage takes it and returns it, or
returns a `Report`. What you do to rows is a method on rows:
`data.grade()`, `scored.select()`, `rows.export()`, `rows.push()`,
`rows.decontaminate(eval_set)`. A stage that only works on rows is not a
free function with rows as the first argument. `attach_*` and `stamp_*`
mutate rows in place, so they become methods and stop being free
functions. *(every torch op returns a tensor; x.mean(), x.to().)*

**5. Results are objects that print themselves.** A measurement returns a
dataclass with `__str__` for the terminal and `_repr_html_` for a
notebook. There is no `format_x` twin for a `x_report`; printing is the
object's job. No public call returns a `(rows, report)` tuple or a bare
dict the user has to know the keys of. *(PassAt already prints
`pass@1 0.67 [0.55..0.78]`; do that everywhere.)*

A number the call could not compute says **why, and the fix, next to the
number it is missing from**. `None` alone is a silence the reader fills in
with confidence: a mean printed with no interval and no reason reads as a
result, which is the one thing the constitution says it is not (belief 1,
"a mean alone is not a result"). This is rule 10 for a value rather than an
exception, and it is the same sentence: what happened, and the one call or
field that changes it. Every report already has the place to put it
(`note`, `notes`, `warning`); the rule is that it is filled.

A report that covers part of a space **names the part it does not cover**.
This is the same sentence again for a measurement that was never taken: a
report over five of thirteen markers, printing five clean rows and nothing
else, reads as a verdict on the agent rather than on five phrase lists. The
reader cannot see the edge of the instrument from inside it, so the
instrument says where the edge is, and the call that measures past it
(`style_report` ends with the markers `trace_markers` and `mark_grounding`
stamp). A marker that came out the same on every row is a third case of the
same thing: no interval, and a warning that says a detector that cannot
fail and a behavior that never happened look identical (#270, #760).

A report that was a dict first becomes a `whileai.report.Report`, which
*is* a dict: every key, `.get`, `json.dumps` and `==` against a plain dict
keep working, and `__str__` is the block the `format_*` twin writes. That
is the migration step that costs a caller nothing. `judge_trust`,
`hack_scan`, `delta_report` (`wai.compare`), `leak_report`,
`eval_variance`, `holdout_size` and `style_report` are through it;
`decontaminate`, `export` and `preflight` are not.

**6. Public names are the verb a scientist says.** `simulate`, `grade`,
`select`, `compare`, `train`, `serve`, `push`. Implementation words are
private: `run_judge`, `normalize_judge_result`, `resolve_topology`,
`stamp_stage`, `build_*`, `load_*` get a leading underscore or move under
a namespace the user never imports from. A name that needs a suffix to
say what it returns (`*_rows`, `*_of`) is two names doing one job.

**7. Declare behaviour; do not toggle it.** A mode is one argument with a
small set of values (`mode="rl"`), or an object. It is never three booleans
that interact (`grade=`, `llm_grade=`, `simulator=False`). If two flags
have to be read together to know what happens, replace them with one
value. *(dspy signatures: `"question -> answer"`.)*

**8. Settings once, override per call.** `wai.configure(agent=, judge=,
api_key=)` sets the session. A `with wai.context(judge=...)` block scopes
an override. A kwarg on the call wins over both. No public call reads an
environment variable the user did not name in the docs. *(dspy.configure
and dspy.context.)*

**9. Every default is named, sourced and tunable from the call.** The
existing rule, kept: a number lives in `defaults.py` with a `# NAME =
value: why (source)` comment, and the call that uses it exposes it as a
kwarg. `scripts/check_no_hardcoding.py` enforces the first half; the
ratchet enforces the second on new calls. *(every torch.optim default is in
the signature and the docstring.)*

**10. Errors and warnings name the fix, from the call that took the bad
value.** A `ValueError` says the kwarg, the bound and the value. Whichever
call accepted the value raises: `wai.configure(agent="openai/gpt-4.1-mini")`
refuses the string on that line and names `"openai:gpt-4.1-mini"`, rather
than storing it for `simulate` to fail on later, in a frame the user did
not write. A warning says the one call that changes the outcome. No
warning is emitted twice for the same cause in one run. The near miss is
worth naming in the message when it is a neighbour's spelling: DSPy and
LiteLLM write `dspy.LM("openai/gpt-4o-mini")` with a slash, so the error
for a slash says which colon form to type.

**11. Typed, importable, cheap.** Every public signature is fully typed
and `py.typed` ships. `import whileai` takes under 200 ms, makes no
network call, and pulls in nothing beyond `requests` and `pydantic`; numpy
and torch are imported inside the functions that need them.

**12. Docstrings teach then prove.** Line one says what the call computes.
Then the parameters in prose, the return type, and a `Reference` line with
the numbered source from the README. A method with no source says
"convention, untested". The example in the docstring runs offline.

## What the ratchet checks

`tests/api/test_style_ratchet.py` pins today's counts and fails when any
grows:

| Count | Rule | Today |
|---|---|---|
| names in `whileai.simulations.__all__` | 1 | 208 |
| names in `whileai.__all__` (the front door) | 1 | 31 |
| front-door calls returning a bare `dict` or tuple | 5 | 3 |
| public calls or constructors with more than 8 parameters (record dataclasses exempt) | 3 | 27 |
| parameters on the widest public call (`simulate`) | 3 | 45 |
| parameters over the cap, summed across those 27 calls | 3 | 194 |
| public names starting `format_` | 5 | 12 |
| public names starting `attach_` or `stamp_` | 4 | 6 |
| public names starting `build_`, `load_`, `run_` or ending `_of`, `_rows` | 6 | 17 |
| imports that are not `import whileai as wai` (`tests/api/test_alias_surface.py`) | 1 | 98 |

Lower a number in the test when you retire a name. Never raise one. A PR
that has to raise one says why in the body and gets a second reviewer.

The numbers in this table are read off the test, not maintained by hand.
Four of them had drifted by the time #459 and #460 were re-checked on
0.123: `simulate` had gone 43 -> 45 and `delta_report` 17 -> 18 without
moving the count of wide calls, which is why the widest-signature and
total-overage rows exist.

## Migration

The path from today's surface to the front page above, in the order that
pays off first:

1. `whileai/__init__.py` re-exports the loop verbs and nouns; README and
   docs switch to `import whileai as wai`. Nothing is removed.
2. `optimize` returns a `Selection` object with `.rows` and `__str__`;
   `SimulationData.select()` wraps it. `format_*` functions become
   `__str__` on their report and are deprecated with a one-release warning.
3. `Judge(model=, rubric=, ...)` absorbs the sixteen kwargs of `grade`.
4. `configure()` and `context()` land; per-call `agent=`/`judge=` keep
   working.
5. `simulate` keeps its signature for one release behind a `Simulation`
   object that holds the forty kwargs in named groups (`world=`,
   `sampling=`, `budget=`).
