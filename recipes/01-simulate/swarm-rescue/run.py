"""Rescue the prompts GRPO throws away: does a swarm find a passing answer where
sampling alone finds none?

On a task where the policy fails every one of G rollouts, GRPO has zero
advantage and dynamic sampling drops the prompt (Yu et al. 2025, DAPO). Those
are the tasks most worth learning from. This recipe gives every such task the
same extra budget four ways and asks which one finds a passing answer:

  resample  B fresh independent samples (what "sample more" does)
  solo      N particles, R rounds; each rewrites its own best attempt with the
            visible-test feedback, no sharing (PSO with the social term off)
  ring      solo, plus each particle sees the best attempt of its two ring
            neighbours (PSO lbest)
  star      solo, plus each particle sees the best attempt of the whole swarm
            (PSO gbest)

B = N x R for every arm, so the budget matches. The number is rescue rate: the
share of all-fail tasks that got at least one answer passing every test,
paired across arms with a bootstrap interval and a sign-flip p-value
(``wai.compare_runs``). Fitness inside the swarm is the visible tests; a
rescue is the hidden tests, which no arm ever sees.

Run: python run.py --dry-run             # three toy tasks, a fake model, no key
     python run.py --limit 20            # hosted Qwen3-4B, twenty code_contests tasks
     python run.py                       # every task, ~2 hours on the hosted model
     python run.py --reuse --post        # the numbers again from out/, then the Runs page
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import whileai.simulations as wai
from whileai.config import provenance

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "out"

# --- the budget -------------------------------------------------------------
G = 8  # base rollouts per task; all G fail = the task GRPO would drop
N = 8  # particles
R = 3  # rounds: round 0 is N independent samples, rounds 1..R-1 refine
B = N * R  # extra samples every arm gets per all-fail task
ARMS = ("resample", "solo", "ring", "star")

# --- the tests ---------------------------------------------------------------
# code_contests gives public tests (in the statement), private tests and
# generated tests. Visible = public + the first VIS_GEN generated; hidden =
# private + the next HID_GEN generated. Fitness reads visible; a rescue
# needs visible and hidden. Capped so grading stays under a few seconds a
# sample on one machine.
VIS_GEN = 8
HID_GEN = 32
TEST_TIMEOUT = 6.0  # seconds per test; the contest limit is 1-2 s for C++
FEEDBACK_CHARS = 300

# --- the model ---------------------------------------------------------------
# The model While hosts on the account key (the SDK's default agent route).
HOSTED_URL = "https://zeroproofai--zeroproof-serve-qwen3-4b.modal.run/v1"
HOSTED_MODEL = "Qwen/Qwen3-4B"
MAX_TOKENS = 2048
TEMPERATURE = 1.0
CONCURRENCY = 48
GRADERS = 8

SYSTEM = (
    "You are a competitive programmer. Solve the problem in Python 3. Read the "
    "input from standard input and write the answer to standard output, exactly "
    "in the format the statement asks for. Think briefly, then reply with one "
    "```python code block containing the whole program."
)

CODE_FENCE = re.compile(r"```(?:python|py)?\s*(.+?)```", re.S)
OPEN_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n")
INTERACTIVE = re.compile(r"interactive problem|interactor|flush the output|fflush", re.I)


# ============================================================ tasks


@dataclass
class Task:
    id: str
    name: str
    statement: str
    visible: list[tuple[str, str]]
    hidden: list[tuple[str, str]]
    tags: dict[str, str] = field(default_factory=dict)


DRY_TASKS = [
    Task(
        "toy-sum",
        "Sum",
        "Read two integers a and b on one line. Print a + b.",
        [("1 2\n", "3\n"), ("5 7\n", "12\n")],
        [("100 -3\n", "97\n"), ("0 0\n", "0\n")],
    ),
    Task(
        "toy-max",
        "Max",
        "The first line has n. The second line has n integers. Print the largest.",
        [("3\n1 5 2\n", "5\n")],
        [("1\n-4\n", "-4\n"), ("4\n9 9 1 0\n", "9\n")],
    ),
    Task(
        "toy-rev",
        "Reverse",
        "Read one word. Print it reversed.",
        [("abc\n", "cba\n")],
        [("x\n", "x\n"), ("hello\n", "olleh\n")],
    ),
]

PARQUET = {
    "test": "https://huggingface.co/datasets/deepmind/code_contests/resolve/"
    "refs%2Fconvert%2Fparquet/default/partial-test/0000.parquet",
    "valid": "https://huggingface.co/datasets/deepmind/code_contests/resolve/"
    "refs%2Fconvert%2Fparquet/default/partial-valid/0000.parquet",
}


def load_tasks(limit: int | None, seed: int) -> list[Task]:
    """code_contests test + valid splits (Li et al. 2022, AlphaCode), 282
    Codeforces-style problems, median rating 1900. Downloaded once into raw/."""
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover
        sys.exit("needs pyarrow: uv add pyarrow")
    RAW.mkdir(exist_ok=True)
    rows: list[dict] = []
    for split, url in PARQUET.items():
        path = RAW / f"{split}.parquet"
        if not path.exists():
            print(f"downloading code_contests {split} split ...", file=sys.stderr)
            urllib.request.urlretrieve(url, path)
        for r in pq.read_table(path).to_pylist():
            r["split"] = split
            rows.append(r)
    tasks: list[Task] = []
    for r in rows:
        pub = list(zip(r["public_tests"]["input"], r["public_tests"]["output"]))
        priv = list(zip(r["private_tests"]["input"], r["private_tests"]["output"]))
        gen = list(zip(r["generated_tests"]["input"], r["generated_tests"]["output"]))
        visible = pub + gen[:VIS_GEN]
        hidden = priv + gen[VIS_GEN : VIS_GEN + HID_GEN]
        if not visible or not hidden:
            continue
        # An interactive problem talks to a judge; against static tests the
        # program waits forever and every arm times out on it.
        if INTERACTIVE.search(r["description"]):
            continue
        rating = r.get("cf_rating") or 0
        band = (
            "unrated"
            if not rating
            else ("<1500" if rating < 1500 else "1500-2199" if rating < 2200 else "2200+")
        )
        tasks.append(
            Task(
                id=f"cc-{r['split']}-{r['name'].split('.')[0].replace(' ', '_')}",
                name=r["name"],
                statement=r["description"].strip(),
                visible=visible,
                hidden=hidden,
                tags={"rating": band, "split": r["split"]},
            )
        )
    random.Random(seed).shuffle(tasks)
    return tasks[:limit] if limit else tasks


# ============================================================ grading


def extract_code(text: str) -> str:
    """The last closed ```python block; a reply cut off at the token cap has
    an open fence and no close, so everything after the last open fence."""
    text = text or ""
    blocks = CODE_FENCE.findall(text)
    if blocks:
        return blocks[-1].strip()
    opens = list(OPEN_FENCE.finditer(text))
    if opens:
        return text[opens[-1].end() :].strip()
    return text.strip()


FLOAT_TOL = 1e-6


def outputs_match(got: str, expected: str) -> bool:
    """Whitespace-insensitive token match; a token that parses as a number
    matches within 1e-6 absolute or relative, the usual contest checker, so
    ``4.500000000`` passes against ``4.5`` and ``18.666666667`` against
    ``18.666666666666668``."""
    a, b = got.split(), expected.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if abs(fx - fy) > FLOAT_TOL * max(1.0, abs(fy)):
            return False
    return True


_grade_slots = threading.BoundedSemaphore(GRADERS)


def run_tests(code: str, tests: list[tuple[str, str]]) -> tuple[int, dict | None]:
    """Run the program on each test in a fresh subprocess; stop at the first
    failure. Returns (passed, first_failure) where the failure carries the
    input, the expected output and what the program did instead."""
    if not code:
        return 0, {"input": "", "expected": "", "got": "no code in the reply"}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sol.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code)
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        for i, (inp, exp) in enumerate(tests):
            with _grade_slots:
                for attempt in range(4):
                    try:
                        proc = subprocess.run(
                            [sys.executable, "-I", "-S", path],
                            input=inp,
                            cwd=tmp,
                            env=env,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=TEST_TIMEOUT,
                        )
                        break
                    except subprocess.TimeoutExpired:
                        return i, {
                            "input": inp,
                            "expected": exp,
                            "got": f"timed out after {TEST_TIMEOUT:.0f}s",
                        }
                    except OSError:
                        # Windows "paging file is too small" (1455) under many
                        # concurrent spawns: a transient of the grader, not of
                        # the program. Back off and try again.
                        if attempt == 3:
                            raise
                        time.sleep(2.0 * (attempt + 1))
            if proc.returncode != 0:
                tail = (proc.stderr or "").strip().splitlines()
                got = "error: " + (tail[-1] if tail else f"exit {proc.returncode}")
                return i, {"input": inp, "expected": exp, "got": got}
            if not outputs_match(proc.stdout, exp):
                return i, {"input": inp, "expected": exp, "got": proc.stdout}
    return len(tests), None


def grade(text: str, task: Task) -> dict[str, Any]:
    """Visible tests give the fitness the swarm sees; hidden tests decide a
    rescue. Hidden tests run only when every visible test passed."""
    code = extract_code(text)
    v_pass, fail = run_tests(code, task.visible)
    out = {
        "visible_passed": v_pass,
        "visible_total": len(task.visible),
        "fitness": v_pass / len(task.visible),
        "correct": False,
        "feedback": fail,
    }
    if fail is None:
        h_pass, h_fail = run_tests(code, task.hidden)
        out["correct"] = h_fail is None
        out["hidden_passed"] = h_pass
    return out


# ============================================================ the model

_slots = threading.BoundedSemaphore(CONCURRENCY)


class Model:
    """One OpenAI-compatible chat endpoint. Thinking is off: the fitness
    signal is the tests, and 1,500 tokens is enough for a program."""

    def __init__(self, base_url: str, model: str, api_key: str | None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.truncated = 0
        self._lock = threading.Lock()

    def chat(self, messages: list[dict], *, seed: int) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), headers=headers
        )
        for attempt in range(5):
            try:
                with _slots, urllib.request.urlopen(req, timeout=600) as resp:
                    data = json.load(resp)
                usage = data.get("usage") or {}
                with self._lock:
                    self.calls += 1
                    self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    self.completion_tokens += int(usage.get("completion_tokens") or 0)
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":
                    with self._lock:
                        self.truncated += 1
                return choice["message"]["content"] or ""
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                if attempt == 4:
                    raise
                time.sleep(2**attempt + random.random())
                last = exc  # noqa: F841
        raise RuntimeError("unreachable")


class FakeModel(Model):
    """Offline stand-in for ``--dry-run``: wrong on a bare ask, sometimes right
    once it sees test feedback, more often once it sees another attempt. It
    exists so the wiring runs in CI; its numbers mean nothing."""

    SOLUTIONS = {
        "Sum": "a, b = map(int, input().split())\nprint(a + b)",
        "Max": "n = int(input())\nprint(max(map(int, input().split())))",
        "Reverse": "print(input().strip()[::-1])",
    }

    def __init__(self) -> None:
        super().__init__("fake://", "fake", None)

    def chat(self, messages: list[dict], *, seed: int) -> str:
        user = messages[-1]["content"]
        rng = random.Random(hashlib.sha256(f"{seed}:{user[:2000]}".encode()).hexdigest())
        p = 0.0
        if "It failed on this test" in user:
            p = 0.25
        if "Another attempt" in user:
            p = 0.6
        name = next((k for k in self.SOLUTIONS if k in user.split("\n", 1)[0]), None)
        with self._lock:
            self.calls += 1
        if name and rng.random() < p:
            return f"Here is the fix.\n```python\n{self.SOLUTIONS[name]}\n```"
        return "I will print zero.\n```python\nprint(0)\n```"


# ============================================================ prompts


def base_messages(task: Task) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{task.name}\n\n{task.statement}"},
    ]


def _clip(s: str) -> str:
    s = s if isinstance(s, str) else str(s)
    return s if len(s) <= FEEDBACK_CHARS else s[:FEEDBACK_CHARS] + " ..."


def describe_attempt(label: str, attempt: dict) -> str:
    g = attempt["grade"]
    lines = [
        f"{label} (passed {g['visible_passed']} of {g['visible_total']} sample tests):",
        "```python",
        extract_code(attempt["text"]),
        "```",
    ]
    fb = g.get("feedback")
    if fb:
        lines += [
            "It failed on this test.",
            f"Input:\n{_clip(fb['input'])}",
            f"Expected output:\n{_clip(fb['expected'])}",
            f"Your program's output:\n{_clip(fb['got'])}",
        ]
    return "\n".join(lines)


def swarm_messages(task: Task, own: dict, social: dict | None) -> list[dict]:
    parts = [
        f"{task.name}\n\n{task.statement}",
        "",
        describe_attempt("Your best attempt so far", own),
    ]
    if social is not None:
        parts += [
            "",
            describe_attempt(
                "Another attempt, from a teammate working on the same problem", social
            ),
        ]
    parts += [
        "",
        "Write an improved solution. Keep what works, fix what fails, and if the "
        "approach is wrong change it. Reply with one ```python code block.",
    ]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(parts)}]


# ============================================================ arms


def sample(model: Model, task: Task, messages: list[dict], seed: int) -> dict:
    text = model.chat(messages, seed=seed)
    return {"text": text, "grade": grade(text, task)}


def _fan(fn, items):
    with cf.ThreadPoolExecutor(max_workers=len(items) or 1) as ex:
        return list(ex.map(fn, items))


def arm_resample(model: Model, task: Task, seed: int) -> dict:
    """B fresh independent samples, laid out as R blocks of N so the
    by-round curve is comparable with the swarms. Stops at the first rescue."""
    samples: list[dict] = []
    rescued_round = None
    for r in range(R):
        batch = _fan(
            lambda i, r=r: sample(model, task, base_messages(task), seed * 1000 + r * N + i),
            range(N),
        )
        for i, s in enumerate(batch):
            samples.append({"round": r, "particle": i, **s})
        if any(s["grade"]["correct"] for s in batch):
            rescued_round = r
            break
    return {"rescued": rescued_round is not None, "round": rescued_round, "samples": samples}


def neighbours(topology: str, i: int) -> list[int]:
    if topology == "solo":
        return []
    if topology == "ring":
        return [(i - 1) % N, (i + 1) % N]
    if topology == "star":
        return [j for j in range(N) if j != i]
    raise ValueError(topology)


def arm_swarm(model: Model, task: Task, seed: int, topology: str) -> dict:
    """PSO in prose. Position = the particle's current program. Personal best
    = its highest-fitness attempt (fitness: share of visible tests passed;
    a tie goes to the newer attempt so the particle keeps moving). The
    social term = the best personal best in the neighbourhood, shown in
    the prompt beside the particle's own. The LLM is the velocity update.
    Kennedy & Eberhart 1995; the ring (lbest) vs star (gbest) contrast is
    Kennedy & Mendes 2002."""
    rng = random.Random(seed)
    samples: list[dict] = []
    round0 = _fan(lambda i: sample(model, task, base_messages(task), seed * 1000 + i), range(N))
    pbest = list(round0)
    for i, s in enumerate(round0):
        samples.append({"round": 0, "particle": i, **s})
    if any(s["grade"]["correct"] for s in round0):
        return {"rescued": True, "round": 0, "samples": samples}
    for r in range(1, R):

        def step(i: int, r: int = r) -> dict:
            social = None
            nb = neighbours(topology, i)
            if nb:
                best = max(pbest[j]["grade"]["fitness"] for j in nb)
                pick = rng.choice([j for j in nb if pbest[j]["grade"]["fitness"] == best])
                social = pbest[pick]
            return sample(
                model, task, swarm_messages(task, pbest[i], social), seed * 1000 + r * N + i
            )

        batch = _fan(step, range(N))
        for i, s in enumerate(batch):
            samples.append({"round": r, "particle": i, **s})
            if s["grade"]["fitness"] >= pbest[i]["grade"]["fitness"]:
                pbest[i] = s
        if any(s["grade"]["correct"] for s in batch):
            return {"rescued": True, "round": r, "samples": samples}
    return {"rescued": False, "round": None, "samples": samples}


def run_arm(model: Model, arm: str, tasks: list[Task], seed: int, workers: int) -> dict[str, dict]:
    def one(task: Task) -> tuple[str, dict]:
        t0 = time.time()
        if arm == "resample":
            res = arm_resample(model, task, seed)
        else:
            res = arm_swarm(model, task, seed, arm)
        res["seconds"] = round(time.time() - t0, 1)
        return task.id, res

    out: dict[str, dict] = {}
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for tid, res in ex.map(one, tasks):
            out[tid] = res
            done += 1
            flag = f"rescued in round {res['round']}" if res["rescued"] else "not rescued"
            print(
                f"  [{arm}] {done}/{len(tasks)} {tid}: {flag} ({res['seconds']}s)", file=sys.stderr
            )
    return out


# ============================================================ base pass


def base_pass(model: Model, tasks: list[Task], seed: int, workers: int) -> dict[str, list[dict]]:
    """G rollouts per task, the group GRPO would see."""

    def one(task: Task) -> tuple[str, list[dict]]:
        batch = _fan(lambda i: sample(model, task, base_messages(task), seed * 7919 + i), range(G))
        return task.id, batch

    out: dict[str, list[dict]] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for n, (tid, batch) in enumerate(ex.map(one, tasks), 1):
            out[tid] = batch
            k = sum(s["grade"]["correct"] for s in batch)
            print(f"  [base] {n}/{len(tasks)} {tid}: {k}/{G} pass", file=sys.stderr)
    return out


# ============================================================ measure


def task_rows(arm_result: dict[str, dict]) -> list[dict]:
    """One row per task: reward 1 if the arm rescued it. Paired by task_id."""
    return [{"task_id": tid, "reward": int(res["rescued"])} for tid, res in arm_result.items()]


def by_round(arm_result: dict[str, dict]) -> list[float]:
    """Cumulative rescue rate after each round, in points."""
    n = len(arm_result) or 1
    return [
        round(
            100
            * sum(1 for r in arm_result.values() if r["round"] is not None and r["round"] <= k)
            / n,
            1,
        )
        for k in range(R)
    ]


def rate_ci(rows: list[dict]) -> tuple[float, float]:
    """Mean and 95% half-width in points, bootstrap over tasks (``pass_at``)."""
    pa = wai.pass_at(
        [{"task_id": r["task_id"], "reward": r["reward"], "rollout_index": 0} for r in rows],
        k=1,
        min_k=1,
    )
    mean = float(pa.pass_at_1 or 0.0)
    lo, hi = pa.ci95 or (mean, mean)
    return round(100 * mean, 1), round(100 * (hi - lo) / 2, 1)


def measure(
    base: dict[str, list[dict]], arms: dict[str, dict[str, dict]], noise: list[dict[str, dict]]
) -> dict:
    hist = [0] * (G + 1)
    for batch in base.values():
        hist[sum(s["grade"]["correct"] for s in batch)] += 1
    all_fail = [tid for tid, batch in base.items() if not any(s["grade"]["correct"] for s in batch)]
    report: dict[str, Any] = {
        "tasks": len(base),
        "all_fail": len(all_fail),
        "p0_histogram": hist,
        "budget": {"G": G, "N": N, "R": R, "B": B},
        "arms": {},
        "deltas": {},
    }
    ref = task_rows(arms["resample"]) if "resample" in arms else None
    for arm, res in arms.items():
        rows = task_rows(res)
        mean, half = rate_ci(rows)
        calls = sum(len(r["samples"]) for r in res.values())
        report["arms"][arm] = {
            "rescue_rate": mean,
            "ci": half,
            "n": len(rows),
            "by_round": by_round(res),
            "samples": calls,
            "samples_per_task": round(calls / max(1, len(res)), 1),
        }
        if ref is not None and arm != "resample":
            c = wai.compare_runs(ref, rows)
            report["deltas"][arm] = {
                "vs": "resample",
                "delta": round(100 * c["delta"], 1),
                "ci95": [round(100 * x, 1) for x in c["ci95"]] if c.get("ci95") else None,
                "p_value": c.get("p_value"),
                "verdict": c["verdict"],
                "n_paired": c["n_paired"],
            }
    if noise and "resample" in arms:
        runs = [task_rows(arms["resample"])] + [task_rows(n) for n in noise]
        ev = wai.eval_variance(*runs)
        report["noise"] = {
            "runs": int(ev["n_runs"]),
            "means": [round(100 * m, 1) for m in ev["means"].values() if m is not None],
            "run_std": round(100 * float(ev["run_std"] or 0.0), 2),
            "band": round(100 * float(ev["noise_band"] or 0.0), 1),
        }
    return report


def print_report(rep: dict) -> None:
    print(f"\ntasks {rep['tasks']}, all-fail at G={G}: {rep['all_fail']}")
    print("per-task passes out of 8 (0..8): " + " ".join(str(x) for x in rep["p0_histogram"]))
    print(f"\nextra budget per task: {B} samples ({N} particles x {R} rounds)")
    print(f"{'arm':10} {'rescued':>9} {'95% +/-':>8} {'by round':>20} {'samples/task':>13}")
    for arm, a in rep["arms"].items():
        print(
            f"{arm:10} {a['rescue_rate']:>8.1f}% {a['ci']:>8.1f} {a['by_round']!s:>20} {a['samples_per_task']:>13}"
        )
    for arm, d in rep["deltas"].items():
        ci = f"[{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]" if d["ci95"] else "no interval"
        p = f"p={d['p_value']:.3f}" if d.get("p_value") is not None else ""
        print(
            f"{arm} vs resample: {d['delta']:+.1f} points {ci} {p} -> {d['verdict']} over {d['n_paired']} tasks"
        )
    if rep.get("noise"):
        n = rep["noise"]
        print(
            f"noise: resample re-run {n['runs']} times {n['means']}, run_std {n['run_std']}, a delta under {n['band']} points is noise"
        )


# ============================================================ files


def dump(path: Path, obj: Any) -> None:
    OUT.mkdir(exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def export_rescued(tasks: dict[str, Task], arms: dict[str, dict[str, dict]]) -> int:
    """The rescued answers as bare rows: prompt in, passing program out, the
    swarm context stripped. This is the privileged export a later SFT or
    distillation run trains on; no arm's prompt leaks into it."""
    rows = []
    seen: set[tuple[str, str]] = set()
    for arm, res in arms.items():
        for tid, r in res.items():
            for s in r["samples"]:
                if s["grade"]["correct"]:
                    code = extract_code(s["text"])
                    if (tid, code) in seen:
                        continue
                    seen.add((tid, code))
                    rows.append(
                        {
                            "task_id": tid,
                            "prompt": base_messages(tasks[tid])[-1]["content"],
                            "final_text": f"```python\n{code}\n```",
                            "reward": 1,
                            "source": arm,
                            "round": s["round"],
                        }
                    )
    OUT.mkdir(exist_ok=True)
    with (OUT / "rescued.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return len(rows)


# ============================================================ platform


def post(
    rep: dict, tasks: dict[str, Task], arms: dict[str, dict[str, dict]], model_name: str
) -> None:
    from whileai.platform import Example, Harness, track

    all_fail = rep["all_fail"]
    test_version = (
        "cc-"
        + hashlib.sha256("\n".join(sorted(next(iter(arms.values())).keys())).encode()).hexdigest()[
            :8
        ]
    )
    tracked = track(
        "swarm-rescue",
        model=model_name,
        harness=Harness(label="base@qwen3-4b", instructions=SYSTEM, model=model_name),
    )
    tracked.experiment(
        question="On the tasks where the policy fails all 8 rollouts, does letting the copies share their best attempts find a passing answer that trying more times alone does not?",
        hypothesis="Ring beats plain resampling; star beats resampling by less, since every particle chasing one leader collapses the swarm early; solo refinement sits in between. Same budget for every arm.",
        method=f"code_contests test+valid, hosted Qwen3-4B, thinking off. {G} base rollouts per task; the all-fail tasks get {B} more samples per arm as {N} particles x {R} rounds. Fitness inside a swarm is the visible tests; a rescue passes the hidden tests too.",
        measure="Rescue rate: the share of all-fail tasks with at least one passing answer, per arm, paired by task, bootstrap interval and sign-flip p-value.",
        decide="An arm whose delta against resampling clears its interval and the re-run noise band is the rollout rule for the training run; a flat result says the hard tasks are hard for lack of knowledge, not lack of sharing.",
    )
    noise_floor = (rep.get("noise") or {}).get("band")
    tracked.behavior(
        "rescued",
        graded_by="program",
        reward_is_judge=False,
        test_version=test_version,
        n=all_fail,
        noise_floor=noise_floor,
        description=f"Share of the {all_fail} all-fail code_contests tasks (0 of {G} base rollouts pass every test) where the arm found a program passing every visible and hidden test within {B} extra samples.",
        rubric="Pass = the program's stdout, split on whitespace, equals the expected output on every visible and hidden test, each within the time limit. No judge.",
    )
    for arm, a in rep["arms"].items():
        run = tracked.run(
            f"{arm}",
            harness=Harness(label=f"{arm}@qwen3-4b", instructions=SYSTEM, model=model_name),
            method="rollout",
        )
        for r, cum in enumerate(a["by_round"]):
            run.log(r, rescued=cum)
        examples = []
        for tid, res in arms[arm].items():
            best = max(res["samples"], key=lambda s: (s["grade"]["correct"], s["grade"]["fitness"]))
            examples.append(
                Example(
                    prompt=tasks[tid].name,
                    reply=extract_code(best["text"])[:1200],  # Example.reply cap
                    ok=bool(res["rescued"]),
                    why=(
                        f"rescued in round {res['round']}"
                        if res["rescued"]
                        else f"best attempt passed {best['grade']['visible_passed']} of {best['grade']['visible_total']} visible tests"
                    ),
                    tags={
                        **tasks[tid].tags,
                        "round": str(res["round"]) if res["rescued"] else "none",
                    },
                )
            )
        run.score("rescued", a["rescue_rate"], ci=a["ci"], n=a["n"], rows=examples)
        d = rep["deltas"].get(arm)
        moved = f"Moved: {a['rescue_rate']} points (+/-{a['ci']}) on {a['n']} all-fail tasks"
        if d:
            moved += f"; {d['delta']:+.1f} vs resample [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}], {d['verdict'].replace('_', ' ')}"
        changed = {
            "resample": f"Changed: {B} fresh independent samples per task, nothing shared",
            "solo": f"Changed: {N} particles x {R} rounds, each rewrites its own best attempt with the failing visible test; no sharing",
            "ring": "Changed: solo, plus each particle also sees the best attempt of its two ring neighbours",
            "star": "Changed: solo, plus each particle also sees the best attempt in the whole swarm",
        }[arm]
        run.note(
            "\n".join(
                [
                    changed,
                    moved,
                    f"Why: cumulative rescue by round {a['by_round']}; {a['samples_per_task']} samples per task used (arms stop at the first rescue)",
                    "Learned: see the README result",
                    f"Reproduce: python recipes/01-simulate/swarm-rescue/run.py --arms {arm} --seed 0",
                ]
            )
        )
        run.finish()
    fig = {
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": arm,
                "x": list(range(R)),
                "y": a["by_round"],
            }
            for arm, a in rep["arms"].items()
        ],
        "layout": {
            "title": "Rescued tasks by round, in points",
            "xaxis": {"title": "round"},
            "yaxis": {"title": "rescued, % of all-fail tasks"},
        },
    }
    tracked.figure(
        "rescue-by-round",
        fig,
        caption="Cumulative share of all-fail tasks with a passing answer after each round, one line per arm, same budget.",
    )
    print(f"\nposted: {tracked.url()}" if hasattr(tracked, "url") else "\nposted")


# ============================================================ main


def main(argv: list[str] | None = None) -> int:
    print(provenance(), file=sys.stderr)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="three toy tasks and a fake model; no key, no network",
    )
    p.add_argument("--limit", type=int, default=None, help="tasks to load (default: all 282)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", default=",".join(ARMS), help="comma list from resample,solo,ring,star")
    p.add_argument(
        "--noise-runs", type=int, default=0, help="extra resample re-runs for the noise band"
    )
    p.add_argument(
        "--workers", type=int, default=6, help="tasks in flight at once (x N model calls each)"
    )
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or HOSTED_URL)
    p.add_argument("--model", default=HOSTED_MODEL)
    p.add_argument(
        "--reuse", action="store_true", help="read out/*.json instead of calling the model"
    )
    p.add_argument("--post", action="store_true", help="post the arms as runs on the platform")
    args = p.parse_args(argv)
    arms_wanted = [a for a in args.arms.split(",") if a]
    for a in arms_wanted:
        if a not in ARMS:
            sys.exit(f"unknown arm {a!r}; choose from {', '.join(ARMS)}")

    global OUT
    if args.dry_run:
        # Never over a real run's files: the toy run gets its own directory.
        OUT = HERE / "out-dry"
        tasks = DRY_TASKS
        model: Model = FakeModel()
    else:
        tasks = load_tasks(args.limit, args.seed)
        key = os.environ.get("WHILEAI_API_KEY") or os.environ.get("ZEROPROOF_API_KEY")
        if args.base_url == HOSTED_URL and not key:
            sys.exit(
                "the hosted model needs WHILEAI_API_KEY (wai login), or pass --base-url for your own endpoint"
            )
        model = Model(
            args.base_url,
            args.model,
            key if args.base_url == HOSTED_URL else os.environ.get("OPENAI_API_KEY"),
        )
    by_id = {t.id: t for t in tasks}
    print(f"{len(tasks)} tasks, model {model.model} at {model.base_url}", file=sys.stderr)

    t0 = time.time()
    base_path = OUT / "base.json"
    if args.reuse and base_path.exists():
        base = load(base_path)
    else:
        base = base_pass(model, tasks, args.seed, args.workers)
        dump(base_path, base)
    all_fail = [
        by_id[tid] for tid, batch in base.items() if not any(s["grade"]["correct"] for s in batch)
    ]
    print(f"all-fail tasks: {len(all_fail)} of {len(base)}", file=sys.stderr)

    arms: dict[str, dict[str, dict]] = {}
    for arm in arms_wanted:
        path = OUT / f"arm-{arm}.json"
        if args.reuse and path.exists():
            arms[arm] = load(path)
            continue
        arms[arm] = run_arm(model, arm, all_fail, args.seed, args.workers)
        dump(path, arms[arm])
    noise: list[dict[str, dict]] = []
    for i in range(args.noise_runs):
        path = OUT / f"noise-{i}.json"
        if args.reuse and path.exists():
            noise.append(load(path))
            continue
        noise.append(run_arm(model, "resample", all_fail, 100 + i, args.workers))
        dump(path, noise[-1])
    if args.reuse:
        for path in sorted(OUT.glob("noise-*.json")):
            if (
                len(noise) < 10
                and all(load(path) is not n for n in noise)
                and path.name not in {f"noise-{i}.json" for i in range(args.noise_runs)}
            ):
                noise.append(load(path))

    rep = measure(base, arms, noise)
    rep["model"] = model.model
    rep["seed"] = args.seed
    rep["calls"] = model.calls
    rep["truncated"] = model.truncated
    rep["tokens"] = {"prompt": model.prompt_tokens, "completion": model.completion_tokens}
    rep["seconds"] = round(time.time() - t0)
    rep["rescued_rows"] = export_rescued(by_id, arms)
    dump(OUT / "results.json", rep)
    print_report(rep)
    print(
        f"\n{rep['rescued_rows']} rescued answers written to out/rescued.jsonl (bare prompt -> passing program)"
    )
    if args.post and not args.dry_run:
        post(rep, by_id, arms, model.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
