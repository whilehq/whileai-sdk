---
name: harness-search
description: >
  How a coding agent runs the Meta-Harness loop as the proposer: read
  out/proposal.md, write the next candidates/<n>_<name>.py, run the recipe,
  read the ledger, stop when the gate passes or after N rounds, then report
  Changed / Moved / Why / Learned / Reproduce and post every candidate as a
  harness version. Use when the thing to improve is the harness (the prompt,
  the tools, the turn cap, the retry) and not the weights, on a closed model
  or an open one. No GPU, no key for the dry run.
metadata:
  version: "1.0.0"
---

# Harness search

Lee et al. 2026 (Meta-Harness, arXiv:2603.28052) search over harness code
with an agentic proposer that reads the source, scores and traces of every
prior candidate through a filesystem, and check the pick on held-out tasks
and held-out models. The recipe `recipes/papers/meta-harness` is the inner
half: it scores every file in `candidates/`, writes the ledger, the
proposal and the gate's answer. You are the outer half. Every step below is
in `check.py`, which runs the recipe's dry run on a temporary copy in under
a minute; its setup defines `RECIPE` (that copy), `MODEL` (the search
model), `ROUNDS` (3), `run(*flags)` (the recipe with `--dry-run`),
`propose_candidate(proposal)` (your job, scripted there), `load_harness`
(a candidate file as `wai.Harness`) and `fake` (a platform that answers
like the API).

## 1. Run the loop once and read what it wrote

Three files carry the state: the ledger (one line per candidate, the score
with its interval on the train and holdout splits), the proposal (every
candidate's source, score and five worst rows, then the one instruction),
and the selection (the gate's answer). Read all three; never score by hand.

```python
def read_ledger(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def state() -> tuple[str, list[dict], dict]:
    out = RECIPE / "out"
    return (
        (out / "proposal.md").read_text(encoding="utf-8"),
        read_ledger(out / "ledger.jsonl"),
        json.loads((out / "selected.json").read_text(encoding="utf-8")),
    )


run("--propose", "--select", "--fresh")
proposal, ledger, selected = state()
```

## 2. Propose, run, read, stop

The proposal names the file to write. Read the worst rows of the best
candidate so far and change the one thing they say is wrong: a line in the
instructions, a tool, a `Disclosure` field. The candidate is a file that
defines `harness(model) -> wai.Harness`; copy the shape of the file before
it. Then run again and read the selection. Stop when `selected` names a
candidate, or after `ROUNDS`: a search that never clears the gate is a
result too, and the report says so. The train split picks; a candidate
written from the train split's worst rows has seen them, so only the
holdout and the held-out model decide (rlhfbook.com, "Evaluation").

```python
NEXT = re.compile(r"write candidates/(\d\d)_<name>\.py")

for _round in range(ROUNDS):
    if selected["selected"]:
        break
    number = NEXT.search(proposal).group(1)
    name, body = propose_candidate(proposal)  # you: read the worst rows, write the file
    (RECIPE / "candidates" / f"{number}_{name}.py").write_text(body, encoding="utf-8")
    run("--propose", "--select")
    proposal, ledger, selected = state()
```

## 3. Report so a person can decide

Five lines, from the ledger and the selection, never from memory. Changed
is the file and its fingerprint. Moved is the holdout, in points, with the
paired delta and its interval. Why is what the baseline's worst rows said.
Learned closes the question and names the paper. Reproduce is the command
and the seed. Post every candidate as a harness version so the Runs page
groups the dots by harness; `load_harness` gives the same fingerprint the
ledger carries. Archive nothing: a candidate that lost is the record of
what was tried.

```python
pick_name = selected["selected"] or selected["best_on_train"]
base, pick = ledger[0], next(e for e in ledger if e["candidate"] == pick_name)
check = selected["checks"][MODEL]
lo, hi = check["ci95"]
worst = read_ledger(RECIPE / "out" / base["worst"])
note = (
    f"Changed: {pick['candidate']}, fingerprint {pick['fingerprint']}, against {base['candidate']}.\n"
    f"Moved: {100 * base['holdout']['pass_at_1']:.0f} to {100 * pick['holdout']['pass_at_1']:.0f} "
    f"points on {pick['holdout']['n_tasks']} held-out tasks, "
    f"{100 * check['delta']:+.0f} [{100 * lo:+.0f}, {100 * hi:+.0f}].\n"
    f"Why: the baseline's worst rows were {'; '.join(sorted({w['why'] for w in worst}))}.\n"
    f"Learned: {selected['reason']} (Meta-Harness, Lee et al. 2026, arXiv:2603.28052).\n"
    f"Reproduce: cd recipes/papers/meta-harness && python run.py --dry-run --propose --select --seed 0"
)

tracked = track("support-bot", model=MODEL, transport=fake)  # drop transport= for real
tracked.behavior(
    Behavior(
        name="plain_answer",
        test_version="t-" + base["fingerprint"][:8],
        n=pick["holdout"]["n_tasks"],
        judge=Judge(name="filler, fault and leak checks as a program"),
        reward_is_judge=False,
    )
)
for entry in ledger:
    version = tracked.run(
        entry["label"],
        method="eval",
        harness=load_harness(entry["candidate"], MODEL),
        targets=["plain_answer"],
    )
    ci = entry["holdout"]["ci95"]
    version.score(
        "plain_answer",
        100 * entry["holdout"]["pass_at_1"],
        ci=50 * (ci[1] - ci[0]),
        n=entry["holdout"]["n_tasks"],
    )
    if entry["candidate"] == pick["candidate"]:
        version.note(note)
    version.finish(say=False)
print(note)
print("verdict:", tracked.verdict())
```

## What the person sees

One dot per candidate on the Runs page, grouped by harness, each with its
interval. Under the picked one, five lines that say what changed, by how
much on tasks it never saw, why, what it taught, and how to run it again.
Under the others, the score that lost. A search that stopped at `ROUNDS`
with nothing selected says so in Learned, and the next round starts from
the same proposal.
