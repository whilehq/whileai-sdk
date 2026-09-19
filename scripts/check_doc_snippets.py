#!/usr/bin/env python
"""Run the code in the docs and check it still works.

Every ```python block under docs/ is executed against the installed package,
except on the two generated trees: docs/api/ (from the package's docstrings)
and docs/recipes/ (from the recipe READMEs, whose scripts live in a clone and
are smoke-tested by their own CI job). A page is one program: blocks run in
order, in one namespace, in one scratch directory, so a later block can use a
name an earlier block defined. That is how a reader reads the page, so that is
how it is checked.

No keys are set. A block that needs one is skipped, and it only earns the skip
if the page says so first: a sentence, or an `export` line, naming the variable
(WHILEAI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, MODAL_TOKEN_ID) somewhere
above the block. An undeclared block that reaches for the network fails here.

When a page quotes a block's output in the fence right after it, the quoted
text has to match what the block printed on stdout. Warnings go to stderr and
are compared against nothing: a page quotes the output of its own code, and a
library warning is not that. The run still names every block that warned, so a
warning arriving from a release cannot pass unseen. The report lines the
package itself prints (`warning: ...` inside a selection report) are stdout and
stay checked. A block written as a session (`>>> call`, then what it printed or
raised) is checked line by line the same way, with an exception rendered as
the page shows it (`ValueError: message`) and whitespace normalised.

A guide is allowed to write `agent`, `TOOLS` or `POLICY` without defining them:
that is the reader's part, and spelling it out in every block would bury the
call being taught. Those names come from a fixture under `scripts/doc_snippets/`, mirroring the page's
path below `docs/`, which runs in the page's namespace before its first block. The
fixture is the reader's side of the page, written out once. The blocks still
run, so a renamed argument or a changed signature still fails here.

A block that is a sketch rather than a program (a signature, a shape, a live
training run) opts out in `scripts/doc_snippets/skips.json`, by page and by a
substring of its own code. The list is deliberately off to one side: nothing a
reader sees changes, and every entry has to carry a reason.

```bash blocks are not run. Any `whileai <subcommand>` they invoke is checked
against the CLI's own help, which catches a renamed or deleted command.

    uv run python scripts/check_doc_snippets.py
    uv run python scripts/check_doc_snippets.py --python /tmp/rel/bin/python
    uv run python scripts/check_doc_snippets.py --page docs/evals.md -v
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
FIXTURES = Path(__file__).resolve().parent / "doc_snippets"

# Named in recipes/README.md as the keys a reader is expected to export.
KEY_VARS = (
    "WHILEAI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MODAL_TOKEN_ID",
    "TYPESAFE_API_KEY",
)
# Cleared before anything runs, so a block that needs one fails loudly here
# instead of quietly billing whoever runs the check. Named ones first, then
# anything that looks like a credential: a developer machine with a key in its
# environment must get the same result as CI, or the check is only as honest as
# whoever last ran it.
CLEARED = (
    *KEY_VARS,
    "MODAL_TOKEN_SECRET",
    "VLLM_API_KEY",
    "OPENAI_BASE_URL",
    "HF_TOKEN",
)
CREDENTIAL_ISH = re.compile(r"API_KEY|_TOKEN|TOKEN_|CREDENTIAL|SECRET", re.I)


def clean_env() -> dict[str, str]:
    """os.environ without anything that could let a block reach a real service."""
    return {
        k: v for k, v in os.environ.items() if k not in CLEARED and not CREDENTIAL_ISH.search(k)
    }


# How the package says "you have no key". A block that dies this way is
# excused, but only if the page warned the reader first.
NEEDS_KEY = re.compile(
    r"No credential|No API key|No TypeSafe API key|Hosted models need a key|need a key:|"
    r"run `whileai login`|set (?:OPENAI|ANTHROPIC|VLLM|MODAL|TYPESAFE)_[A-Z_]*(?:KEY|ID|SECRET)|"
    r"rejected the API key \(401\)",
    re.I,
)

# Three backticks or more, closed by the same count, so a ````markdown block
# that quotes a ```python fence is one block of markdown, not a python block.
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,})(?P<info>[^`\n]*)$")
# simulate() prints a progress meter while it runs. A reader watches it go by
# and quotes what is left, so the comparison drops it too.
PROGRESS = re.compile(r"^\s*\d+/\d+ rollouts,.*$", re.M)


def printed(text: str) -> str:
    """What a page would quote: the block's output without the progress meter.

    Dedented, because a fence nested inside a Mintlify component carries that
    component's indentation and the program's output does not.
    """
    lines = [ln.rstrip() for ln in PROGRESS.sub("", text).splitlines() if ln.strip()]
    pad = min((len(ln) - len(ln.lstrip()) for ln in lines), default=0)
    return "\n".join(ln[pad:] for ln in lines).strip()


SKIPS = json.loads((FIXTURES / "skips.json").read_text(encoding="utf-8"))["skips"]
# Generated trees under docs/, never run, because a broken block in either is
# fixed at its source and not on the page: docs/api/ comes from __all__ and the
# docstrings, docs/recipes/ from recipes/**/README.md. A recipe page also
# excerpts a script that runs inside its own directory, with that directory's
# files importable and names bound higher up the script; run here, in a
# scratch directory with only the package, those excerpts fail for reasons the
# page never claimed otherwise. Running them honestly needs a per-recipe
# working directory, which is #499. The recipe smoke job covers them until then.
GENERATED = ("api/", "recipes/")
# BLOCK_TIMEOUT = 180: seconds one block may run. The slowest honest block is
# a seeded simulate() at budget=64 with repeats=4, which takes about 20 s on the
# CI runner; nine times that is a hang, not a slow machine.
BLOCK_TIMEOUT = 180


@dataclass
class Block:
    page: Path
    line: int  # 1-based line of the opening fence
    lang: str
    code: str
    skip_marker: str | None = None
    quoted_output: str | None = None
    needs_key: str | None = None


@dataclass
class Result:
    block: Block
    status: str  # ok | failed | skipped | mismatch
    detail: str = ""
    stdout: str = ""
    stderr: str = ""


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)


def pages(only: str | None) -> list[Path]:
    if only:
        p = (REPO / only).resolve()
        return [p] if p.exists() else []
    found: list[Path] = []
    for pattern in ("**/*.md", "**/*.mdx"):
        found += [
            p
            for p in DOCS.glob(pattern)
            if not any(p.relative_to(DOCS).as_posix().startswith(g) for g in GENERATED)
        ]
    return sorted(set(found))


def language(info: str) -> str:
    """First word of a fence info string: ```python title="x" -> python."""
    return info.strip().split()[0].lower() if info.strip() else ""


def parse(page: Path) -> list[Block]:
    """Pull every fenced block out of a page, in order."""
    lines = page.read_text(encoding="utf-8").splitlines()
    raw: list[tuple[int, str, str]] = []  # (line, lang, body)
    fenced = [False] * len(lines)  # a '#' in here is a comment, not a heading
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, lang = m.group("indent"), language(m.group("info"))
        fence = m.group("fence")
        body: list[str] = []
        i += 1
        while i < len(lines) and lines[i].strip() != fence:
            fenced[i] = True
            body.append(lines[i].removeprefix(indent))
            i += 1
        raw.append((i - len(body), lang, "\n".join(body)))
        i += 1

    blocks: list[Block] = []
    for idx, (line, lang, code) in enumerate(raw):
        if lang not in ("python", "bash"):
            continue
        b = Block(page=page, line=line, lang=lang, code=code)
        # An entry in skips.json opts the block out, by page and code substring.
        rel = page.relative_to(REPO).as_posix()
        for entry in SKIPS:
            if entry["page"] == rel and entry["contains"] in code:
                b.skip_marker = entry["reason"]
                break
        # A page quotes output in the unlabeled fence right after the code.
        if lang == "python" and idx + 1 < len(raw):
            nxt_line, nxt_lang, nxt_code = raw[idx + 1]
            if nxt_lang in ("", "text", "console") and nxt_line - line < len(code.splitlines()) + 4:
                b.quoted_output = nxt_code
        # Where the page declares a key, if it does. This does not decide
        # anything on its own: a block is only excused if it actually fails for
        # want of a key, so a declaration cannot silence a real bug.
        start = 0
        first_heading = len(lines)
        for n in range(line - 1, -1, -1):
            if lines[n].startswith("#") and not fenced[n]:
                start = n
                break
        for n, text in enumerate(lines):
            if text.startswith("#") and not fenced[n] and not text.startswith("---"):
                first_heading = n
                break
        # The page's own preamble speaks for the whole page: a reference page
        # whose examples all run on the platform says so once, at the top.
        declared = "\n".join(lines[:first_heading]) + "\n" + "\n".join(lines[start:line])
        for var in KEY_VARS:
            if var in declared or var in code:
                b.needs_key = var
                break
        blocks.append(b)
    return blocks


DRIVER = r"""
import io, json, os, sys, threading, time, traceback
from contextlib import redirect_stdout, redirect_stderr

blocks = json.loads(sys.argv[1])
fixture = sys.argv[2]
FIXTURES_ROOT = sys.argv[3]
BLOCK_TIMEOUT = float(sys.argv[4])
ns = {"__name__": "__main__"}
real_out = sys.stdout


def emit(rec):
    # One record per block, written the moment the block finishes, so a page
    # that dies or hangs later still reports every block that ran.
    real_out.write("\x00REC\x00" + json.dumps(rec) + "\n")
    real_out.flush()


if fixture:
    try:
        ns["__file__"] = fixture
        sys.path.insert(0, FIXTURES_ROOT)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            exec(compile(open(fixture, encoding="utf-8").read(), fixture, "exec"), ns)
        ns.pop("__file__", None)
    except BaseException:
        for b in blocks:
            emit({"line": b["line"], "status": "failed", "stdout": "", "stderr": "",
                  "detail": "the page fixture failed:\n" + traceback.format_exc(limit=4)[-900:]})
        raise SystemExit(0)

# The watchdog: one block may run for BLOCK_TIMEOUT seconds. Past that, the
# block is reported as timed out and the process stops, because the page's
# namespace after a hung block is not one worth continuing in.
current = {"line": None, "started": 0.0}


def watch():
    while True:
        time.sleep(1)
        line, started = current["line"], current["started"]
        if line is not None and time.monotonic() - started > BLOCK_TIMEOUT:
            emit({"line": line, "status": "failed", "stdout": "", "stderr": "",
                  "detail": "timed out after %d s" % BLOCK_TIMEOUT})
            real_out.flush()
            os._exit(0)


threading.Thread(target=watch, daemon=True).start()


def run_session(b, ns, rec):
    # A block written as a session (`>>> call` then what it printed or raised)
    # is checked as one: each line runs in the page's namespace, an exception
    # is rendered the way the page shows it (`ValueError: message`, no
    # traceback header), and whitespace is normalised because the page wraps
    # a long message at the margin and the interpreter does not.
    import doctest
    got, want = [], []
    for ex in doctest.DocTestParser().get_examples(b["code"]):
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                exec(compile(ex.source, "<%s:%s>" % (b["page"], b["line"]), "single"), ns)
        except Exception as e:  # what the page shows under the call
            out.write("%s: %s\n" % (type(e).__name__, e))
        got.append(out.getvalue())
        want.append(ex.want)
    if " ".join("".join(got).split()) != " ".join("".join(want).split()):
        rec["status"] = "mismatch"
        rec["detail"] = "page shows:\n    %s\n  session printed:\n    %s" % (
            "".join(want).strip() or "(nothing)", "".join(got).strip() or "(nothing)")


for b in blocks:
    buf, errbuf = io.StringIO(), io.StringIO()
    rec = {"line": b["line"], "status": "ok", "detail": "", "stdout": "", "stderr": ""}
    current["line"], current["started"] = b["line"], time.monotonic()
    try:
        # stdout and stderr apart: a page quotes what the block printed, and a
        # warning the package writes to stderr is not that. stderr is kept
        # rather than dropped, so the run can still name every block that
        # warned the reader.
        with redirect_stdout(buf), redirect_stderr(errbuf):
            if b["code"].lstrip().startswith(">>> "):
                run_session(b, ns, rec)
            else:
                exec(compile(b["code"], "<%s:%s>" % (b["page"], b["line"]), "exec"), ns)
    except BaseException:
        rec["status"] = "failed"
        rec["detail"] = traceback.format_exc(limit=6).strip()[-1200:]
    current["line"] = None
    rec["stdout"] = buf.getvalue()
    rec["stderr"] = errbuf.getvalue()
    emit(rec)
"""


def run_page(page: Path, blocks: list[Block], python: str, verbose: bool) -> list[Result]:
    results: list[Result] = []
    runnable: list[Block] = []
    for b in blocks:
        if b.lang != "python":
            continue
        if b.skip_marker:
            results.append(Result(b, "skipped", f"marked: {b.skip_marker}"))
        else:
            runnable.append(b)

    if runnable:
        env = clean_env()
        env["WHILEAI_DOCS_CHECK"] = "1"
        # The child reads and writes UTF-8 whatever the host's locale, so a page
        # with a non-ASCII character gives the same answer on Windows as on the
        # CI runner.
        env["PYTHONUTF8"] = "1"
        with tempfile.TemporaryDirectory(prefix="docsnip-") as tmp:
            env["HOME"] = tmp
            env["USERPROFILE"] = tmp  # what Path.home() reads on Windows
            payload = json.dumps(
                [{"line": b.line, "code": b.code, "page": page.name} for b in runnable]
            )
            fixture = FIXTURES / page.relative_to(DOCS).with_suffix(".py")
            argv = [
                python,
                "-c",
                DRIVER,
                payload,
                str(fixture) if fixture.exists() else "",
                str(FIXTURES),
                str(BLOCK_TIMEOUT),
            ]
            # The driver stops itself one block past BLOCK_TIMEOUT; this outer
            # limit is the backstop for a block that also wedged the watchdog.
            try:
                proc = subprocess.run(
                    argv,
                    cwd=tmp,
                    env=env,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=BLOCK_TIMEOUT * (len(runnable) + 1),
                )
                raw_out = proc.stdout or ""
            except subprocess.TimeoutExpired as e:
                raw_out = (
                    e.stdout.decode("utf-8", "replace")
                    if isinstance(e.stdout, bytes)
                    else (e.stdout or "")
                )
            recs = []
            for chunk in raw_out.split("\x00REC\x00")[1:]:
                try:
                    recs.append(json.loads(chunk.strip().splitlines()[0]))
                except (ValueError, IndexError):
                    continue
            by_line = {r["line"]: r for r in recs}
            stopped = any(r["status"] == "failed" for r in recs)
            for b in runnable:
                rec = by_line.get(b.line)
                if rec is None:
                    why = (
                        "did not run: the page stopped at an earlier block"
                        if stopped
                        else "did not run: the page's process ended without reporting"
                    )
                    results.append(Result(b, "failed", why))
                    continue
                if rec["status"] == "mismatch":
                    results.append(
                        Result(b, "mismatch", rec["detail"], rec["stdout"], rec.get("stderr", ""))
                    )
                    continue
                if rec["status"] == "failed":
                    if NEEDS_KEY.search(rec["detail"]):
                        if b.needs_key:
                            results.append(
                                Result(b, "skipped", f"needs a key; page declares {b.needs_key}")
                            )
                        else:
                            results.append(
                                Result(
                                    b,
                                    "failed",
                                    "this block needs a key, and nothing above it on the page "
                                    "says so. Name the variable in a sentence before it, or give "
                                    "the block an offline form.\n" + rec["detail"],
                                )
                            )
                        continue
                    results.append(
                        Result(b, "failed", rec["detail"], rec["stdout"], rec.get("stderr", ""))
                    )
                    continue
                if b.quoted_output is not None:
                    got, want = printed(rec["stdout"]), printed(b.quoted_output)
                    if got != want:
                        results.append(
                            Result(
                                b,
                                "mismatch",
                                f"page quotes:\n    {want}\n  block printed:\n    {got or '(nothing)'}",
                                rec["stdout"],
                                rec.get("stderr", ""),
                            )
                        )
                        continue
                results.append(Result(b, "ok", "", rec["stdout"], rec.get("stderr", "")))
                if verbose and rec["stdout"].strip():
                    print(f"  {page.name}:{b.line} printed: {rec['stdout'].strip()[:400]}")

    return sorted(results, key=lambda r: r.block.line)


def cli_subcommands(python: str) -> set[str]:
    try:
        proc = subprocess.run(
            [python, "-m", "whileai", "--help"], capture_output=True, text=True, timeout=120
        )
    except Exception:
        return set()
    text = proc.stdout + proc.stderr
    body = text.split("{", 1)
    names: set[str] = set()
    if len(body) > 1 and "}" in body[1]:
        names |= {n.strip() for n in body[1].split("}", 1)[0].split(",") if n.strip()}
    names |= set(re.findall(r"^\s{2,}([a-z][a-z0-9-]{2,})\s{2,}\S", text, re.M))
    return names


def check_bash(blocks: list[Block], known: set[str]) -> list[Result]:
    results = []
    for b in blocks:
        if b.lang != "bash" or b.skip_marker:
            continue
        used = re.findall(
            r"^\s*(?:\$\s*)?(?:uvx?\s+run\s+)?whileai\s+([a-z][a-z0-9-]*)", b.code, re.M
        )
        bad = [c for c in used if known and c not in known]
        if bad:
            results.append(
                Result(b, "failed", f"CLI has no subcommand {', '.join(sorted(set(bad)))}")
            )
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter with the package installed (default: this one)",
    )
    ap.add_argument("--page", help="check one page, repo-relative")
    ap.add_argument("--all", action="store_true", help="print the status of every block")
    ap.add_argument("-v", "--verbose", action="store_true", help="print what each block printed")
    args = ap.parse_args()

    known = cli_subcommands(args.python)
    report = Report()
    for page in pages(args.page):
        blocks = parse(page)
        if not blocks:
            continue
        report.results += run_page(page, blocks, args.python, args.verbose)
        report.results += check_bash(blocks, known)

    if args.all:
        for r in report.results:
            rel = r.block.page.relative_to(REPO).as_posix()
            print(f"{r.status:9s} {rel}:{r.block.line}  {r.detail[:90]}")

    bad = [r for r in report.results if r.status in ("failed", "mismatch")]
    for r in bad:
        rel = r.block.page.relative_to(REPO).as_posix()
        print(f"\n{rel}:{r.block.line}  {r.status.upper()}  ({r.block.lang})")
        for line in r.detail.splitlines():
            print(f"  {line}")
        if r.stderr.strip():
            print("  the block also wrote to stderr:")
            for line in r.stderr.strip().splitlines()[:6]:
                print(f"    {line}")

    # Warnings are not part of the quoted output, but a reader running the
    # block does see them, so they are named here rather than dropped.
    warned = [r for r in report.results if r.status == "ok" and "Warning:" in r.stderr]
    if warned:
        print(f"\n{len(warned)} block(s) ran clean but warned the reader:")
        for r in warned:
            rel = r.block.page.relative_to(REPO).as_posix()
            first = next(
                (ln for ln in r.stderr.splitlines() if "Warning:" in ln),
                "",
            )
            print(f"  {rel}:{r.block.line}  {first.split('Warning:', 1)[-1].strip()[:150]}")

    ran = report.count("ok")
    print(
        f"\n{ran} blocks ran clean, {report.count('skipped')} skipped, "
        f"{report.count('failed')} failed, {report.count('mismatch')} output mismatches"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
