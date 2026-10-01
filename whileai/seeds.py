"""Seeds from real material: a git repository, a database schema, recorded tool calls.

    import whileai as wai

    asks = wai.seeds.from_repo(".", n=12, seed=0)       # real paths, real months, one absent file
    data = wai.simulate(agent, tools=TOOLS, seeds=asks, simulator=False)

The writer only knows what the tool descriptions and ``seeds=`` tell it.
With neither, the offline writer (``simulator=False``) falls back to a
generic template, and an agent over a code history gets asked about
support tickets. An external case study on whileai 0.126 (Gently Ventures,
https://gentlyventures.com/casestudies/whileai) measured it: 94 of 115
offline cases carried a ticket id and none named a file in the repository.
Handing the same writer real examples through ``seeds=`` raised the cases
that fit the agent from 34 to 57 of 115.

These helpers write those examples from material you already have, so the
asks name things that exist and, on purpose, a few that do not:

* ``from_repo(path)``: tracked file paths, months with commits, a month in
  the history with none, and a file path that was never in the repository.
  It reads paths and dates only. Author names and emails are never read,
  so they cannot reach a seed.
* ``from_schema(source)``: table and column names from a SQLite file, a
  ``.sql`` file or DDL text, and a table that is not there. No row is read.
* ``from_traces(traces)``: identifiers your agent actually passed to its
  tools (or got back under an id-like key), and one of the same shape it
  never saw. Trace prompts are never copied, so a held-out trace stays out
  of the generated set (the leakage rule in ``ingest/traces.py``).

Each returns a :class:`Seeds`, a ``list[str]`` that ``simulate(seeds=)``
takes as is, and that prints what it read. The same material and the same
``seed`` give the same list.

There is no ``simulate(repo=)``: ``simulate`` is the widest call in the
package and the style ratchet refuses a wider one. The seeds go through the
argument the writer already reads.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

__all__ = ["Seeds", "from_repo", "from_schema", "from_traces"]

#: DEFAULT_N = 12: asks per call; enough that every kind of fact (file,
#: month, table, column, id) and the negative case appear at least once in
#: a default run, small enough to read in one screen (convention, untested).
DEFAULT_N = 12
#: MIN_N = 2: one real ask and one negative ask is the smallest set that
#: still tests both sides.
MIN_N = 2
#: NEGATIVE_EVERY = 6: one negative ask (absent file, empty month, absent
#: table, unseen id) per six asks, at least one per call (convention,
#: untested).
NEGATIVE_EVERY = 6
#: MAX_VALUE_CHARS = 64: an argument longer than this is free text, not an
#: identifier, and is not lifted into a seed.
MAX_VALUE_CHARS = 64

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_EMAIL = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
_ID_KEY = re.compile(r"(?:^|_)(?:id|path|file|filename|sku|code|number|ref|key|table)$", re.I)
_IDENT_VALUE = re.compile(r"^[\w./:#-]+$")
_CREATE_TABLE = re.compile(r"\bcreate\s+table\b", re.I)
_NUMERIC_TYPE = re.compile(r"INT|REAL|NUM|DEC|FLOA|DOUB", re.I)

FILE_ASKS = (
    "What changed in {path} over its history?",
    "When was {path} last modified, and what did that change do?",
    "Which commits touched {path}?",
    "When was {path} first added to the repository?",
    "Summarize how {path} has evolved.",
)
MONTH_ASKS = (
    "What was committed in {month}?",
    "Summarize the changes made to the repository in {month}.",
    "Which files changed in {month}?",
)
TABLE_ASKS = (
    "How many rows are in the {table} table?",
    "Show the five most recent rows of {table}.",
)
COLUMN_ASKS = (
    "List the distinct values of {column} in {table}.",
    "Which rows in {table} have no {column}?",
)
NUMERIC_ASKS = (
    "What is the average {column} in {table}?",
    "What is the total {column} across {table}?",
)
JOIN_ASK = "For each row in {table}, show the matching {other} row through {column}."
VALUE_ASKS = (
    "Can you look up {label} {value}?",
    "What is the status of {label} {value}?",
    "Pull up everything on {label} {value}.",
)
ABSENT_TABLES = (
    "invoices",
    "customers",
    "orders",
    "payments",
    "events",
    "shipments",
    "accounts",
    "refunds",
)


class Seeds(list[str]):
    """The asks, plus what they were read from.

    A ``list[str]``: pass it to ``simulate(seeds=)`` unchanged. ``facts``
    holds the material the asks name (``files``, ``months``, ``tables``,
    ``values``, ...), ``negatives`` the asks whose subject does not exist,
    so a judge can expect "not found" on exactly those. ``print(seeds)``
    shows both.
    """

    def __init__(
        self,
        asks: Iterable[str] = (),
        *,
        source: str = "",
        facts: dict[str, list[str]] | None = None,
        negatives: Sequence[str] = (),
    ) -> None:
        super().__init__(asks)
        self.source = source
        self.facts: dict[str, list[str]] = dict(facts or {})
        self.negatives: list[str] = list(negatives)

    def __str__(self) -> str:
        counts = ", ".join(f"{len(v)} {k.replace('_', ' ')}" for k, v in self.facts.items() if v)
        head = f"{len(self)} seeds from {self.source}" + (f": {counts}" if counts else "")
        lines = [head]
        negatives = set(self.negatives)
        for ask in self:
            lines.append(f"  {'-' if ask in negatives else '+'} {ask}")
        lines.append("  (+ names something real, - names something absent: expect 'not found')")
        return "\n".join(lines)


def _check_n(n: int, call: str) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < MIN_N:
        raise ValueError(
            f"{call}(n={n!r}): n is the number of asks, an int >= {MIN_N} "
            "(one real ask and one negative)"
        )
    return n


def _n_negative(n: int) -> int:
    return max(1, n // NEGATIVE_EVERY)


def _fill(rng: random.Random, pools: list[list[str]], k: int) -> list[str]:
    """Round-robin ``k`` distinct asks across shuffled pools, so every kind of
    fact appears before any kind repeats."""
    pools = [list(p) for p in pools if p]
    for pool in pools:
        rng.shuffle(pool)
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < k and any(pools):
        for pool in pools:
            while pool:
                ask = pool.pop()
                if ask not in seen:
                    seen.add(ask)
                    out.append(ask)
                    break
            if len(out) >= k:
                break
    return out


def _git(root: Path, *args: str) -> str:
    cmd = ["git", "-C", str(root), "-c", "core.quotepath=false", *args]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
    except FileNotFoundError:
        raise ValueError("from_repo needs git on PATH; install git and call again") from None
    if done.returncode != 0:
        raise ValueError(
            f"from_repo(path={str(root)!r}) is not a git repository with commits "
            f"({done.stderr.strip() or 'git failed'}); pass the repository root"
        )
    return done.stdout


def _month_label(key: str) -> str:
    year, month = key.split("-")
    return f"{_MONTHS[int(month) - 1]} {year}"


def _shift_month(key: str, delta: int) -> str:
    year, month = (int(x) for x in key.split("-"))
    index = year * 12 + (month - 1) + delta
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _absent_path(rng: random.Random, files: list[str], ever: set[str]) -> str:
    base = rng.choice(files)
    folder, _, name = base.rpartition("/")
    stem, dot, ext = name.partition(".")
    for suffix in ("_legacy", "_old", "_v2", "_backup", "_draft"):
        candidate = (f"{folder}/" if folder else "") + f"{stem}{suffix}{dot}{ext}"
        if candidate not in ever:
            return candidate
    return (f"{folder}/" if folder else "") + f"never_committed_{rng.randrange(10**6)}{dot}{ext}"


def from_repo(path: str | Path = ".", *, n: int = DEFAULT_N, seed: int = 0) -> Seeds:
    """Asks about a git repository's real files and history, plus ones that must come back empty.

    Reads ``git ls-files`` for the tracked paths and ``git log`` for the
    month of each commit and the paths it touched. It never asks git for an
    author, committer or message, so no name or email can reach a seed.

    * ``path``: any directory inside the repository.
    * ``n`` (``DEFAULT_N``, 12): how many asks, fewer only when the
      material runs out. One in ``NEGATIVE_EVERY``
      (6), at least one, is negative: a file path that was never in the
      history, then a month inside the history with no commit (or the month
      before the first commit). The rest alternate between tracked files and
      months with commits.
    * ``seed`` (0): which files, months and phrasings are drawn. Same
      repository, same ``seed``, same list.

    Returns :class:`Seeds`; ``facts`` has ``files``, ``months`` (with
    commits), ``empty_months`` and ``absent_files``.

    ```python
    asks = wai.seeds.from_repo(".", n=12, seed=0)
    print(asks)  # + What changed in src/app.py over its history? ...
    data = wai.simulate(agent, tools=TOOLS, seeds=asks, simulator=False)
    ```

    Reference: Gently Ventures case study on whileai 0.126 (seeds raised
    on-topic offline cases from 34 to 57 of 115); the ask templates are a
    convention, untested.
    """
    n = _check_n(n, "from_repo")
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"from_repo(path={str(path)!r}): not a directory; pass the repo root")
    files = sorted(
        f for f in _git(root, "ls-files", "-z").split("\0") if f and not _EMAIL.search(f)
    )
    if not files:
        raise ValueError(f"from_repo(path={str(path)!r}): no tracked files; commit some first")
    log = _git(
        root, "log", "--no-color", "--no-renames", "--format=%x1e%ad", "--date=format:%Y-%m",
        "--name-only",
    )  # fmt: skip
    months: set[str] = set()
    ever: set[str] = set(files)
    for block in log.split("\x1e"):
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        months.add(lines[0])
        ever.update(lines[1:])
    ordered = sorted(months)
    rng = random.Random(seed)

    empty: list[str] = []
    if ordered:
        key = ordered[0]
        while key < ordered[-1]:
            if key not in months:
                empty.append(key)
            key = _shift_month(key, 1)
        if not empty:
            empty.append(_shift_month(ordered[0], -1))
    absent = _absent_path(rng, files, ever)

    negatives: list[str] = []
    k_neg = _n_negative(n)
    neg_pool = [rng.choice(FILE_ASKS).format(path=absent)]
    if empty:
        neg_pool.append(rng.choice(MONTH_ASKS).format(month=_month_label(rng.choice(empty))))
    negatives = neg_pool[:k_neg]

    picked_files = rng.sample(files, min(len(files), n))
    # one pool per template, so a small repo still fills n with a new
    # phrasing before any ask repeats; four file pools to one month pool,
    # because files are what the case study found missing
    file_pools = [[t.format(path=f) for f in picked_files] for t in FILE_ASKS]
    month_pool = [t.format(month=_month_label(m)) for m in ordered for t in MONTH_ASKS]
    real = _fill(rng, [*file_pools[:2], month_pool, *file_pools[2:]], n - len(negatives))
    asks = real + negatives
    rng.shuffle(asks)
    named_files = [f for f in picked_files if any(f in a for a in real)]
    named_months = [m for m in ordered if any(_month_label(m) in a for a in real)]
    return Seeds(
        asks,
        source=f"repo {root.resolve().name}",
        facts={
            "files": named_files,
            "months": named_months,
            "empty_months": [m for m in empty if any(_month_label(m) in a for a in negatives)],
            "absent_files": [absent] if any(absent in a for a in negatives) else [],
        },
        negatives=negatives,
    )


def _schema_from_connection(con: sqlite3.Connection) -> dict[str, list[tuple[str, str]]]:
    tables = [
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    out: dict[str, list[tuple[str, str]]] = {}
    for table in tables:
        quoted = '"' + table.replace('"', '""') + '"'
        out[table] = [
            (str(r[1]), str(r[2] or "")) for r in con.execute(f"PRAGMA table_info({quoted})")
        ]
    return out


def _foreign_keys(con: sqlite3.Connection, tables: Iterable[str]) -> list[tuple[str, str, str]]:
    keys: list[tuple[str, str, str]] = []
    for table in tables:
        quoted = '"' + table.replace('"', '""') + '"'
        for r in con.execute(f"PRAGMA foreign_key_list({quoted})"):
            keys.append((table, str(r[3]), str(r[2])))
    return sorted(keys)


def _open_schema(source: str | Path) -> sqlite3.Connection:
    text = str(source)
    is_ddl = isinstance(source, str) and _CREATE_TABLE.search(text) is not None
    candidate = None if is_ddl else Path(text)
    if candidate is not None and candidate.is_file():
        with candidate.open("rb") as fh:
            header = fh.read(16)
        if header.startswith(b"SQLite format 3"):
            return sqlite3.connect(candidate.resolve().as_uri() + "?mode=ro", uri=True)
        text = candidate.read_text(encoding="utf-8")
    elif not is_ddl:
        raise ValueError(
            f"from_schema(source={text[:60]!r}): not a SQLite file, a .sql file or "
            "CREATE TABLE text; pass one of those"
        )
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(text)
    except sqlite3.Error as exc:
        con.close()
        raise ValueError(
            f"from_schema: the DDL did not load into SQLite ({exc}); pass a SQLite "
            "file, or DDL with the dialect-only clauses removed"
        ) from None
    return con


def from_schema(source: str | Path, *, n: int = DEFAULT_N, seed: int = 0) -> Seeds:
    """Asks about a database's real tables and columns, plus one about a table that is not there.

    Reads names only: ``sqlite_master`` and ``PRAGMA table_info`` /
    ``foreign_key_list``. No row is selected, so no value in the data can
    reach a seed.

    * ``source``: a SQLite database file (opened read-only), a ``.sql``
      file, or ``CREATE TABLE`` text; DDL is loaded into an in-memory SQLite
      database to read it.
    * ``n`` (``DEFAULT_N``, 12): how many asks, fewer only when the
      material runs out. One in ``NEGATIVE_EVERY``
      (6), at least one, names a table the schema does not have.
    * ``seed`` (0): which tables, columns and phrasings are drawn.

    Returns :class:`Seeds`; ``facts`` has ``tables``, ``columns``
    (``table.column``) and ``absent_tables``.

    ```python
    asks = wai.seeds.from_schema("shop.db", n=12, seed=0)
    data = wai.simulate(agent, tools=TOOLS, seeds=asks, simulator=False)
    ```

    Reference: convention, untested (the ask templates).
    """
    n = _check_n(n, "from_schema")
    con = _open_schema(source)
    try:
        schema = _schema_from_connection(con)
        fks = _foreign_keys(con, schema)
    finally:
        con.close()
    if not schema:
        raise ValueError("from_schema: no tables found; pass the database or the CREATE TABLE text")
    rng = random.Random(seed)
    tables = sorted(schema)
    table_asks = [tpl.format(table=t) for t in tables for tpl in TABLE_ASKS]
    column_asks: list[str] = []
    for t in tables:
        for col, kind in schema[t]:
            numeric = _NUMERIC_TYPE.search(kind) and not _ID_KEY.search(col)
            pool = NUMERIC_ASKS if numeric else COLUMN_ASKS
            column_asks.extend(tpl.format(table=t, column=col) for tpl in pool)
    join_asks = [JOIN_ASK.format(table=t, other=o, column=c) for t, c, o in fks]

    lowered = {t.lower() for t in tables}
    candidates = [name for name in ABSENT_TABLES if name not in lowered]
    absent_names = rng.sample(candidates, len(candidates)) or [f"{tables[0]}_archive"]
    negatives = [
        rng.choice(TABLE_ASKS).format(table=name) for name in absent_names[: _n_negative(n)]
    ]
    real = _fill(rng, [table_asks, column_asks, join_asks], n - len(negatives))
    asks = real + negatives
    rng.shuffle(asks)
    return Seeds(
        asks,
        source=f"schema, {len(tables)} tables",
        facts={
            "tables": [t for t in tables if any(f" {t}" in a for a in real)],
            "columns": [
                f"{t}.{c}"
                for t in tables
                for c, _ in schema[t]
                if any(f"{c} in {t}" in a or f"{c} across {t}" in a for a in real)
            ],
            "absent_tables": absent_names[: _n_negative(n)],
        },
        negatives=negatives,
    )


def _label(key: str) -> str:
    key = re.sub(r"(?i)_?id$", "", key) or key
    return key.replace("_", " ").strip().lower() or "id"


def _usable(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = str(value).strip()
    if not text or len(text) > MAX_VALUE_CHARS or _EMAIL.search(text):
        return None
    return text if _IDENT_VALUE.match(text) else None


def _result_ids(result: Any) -> Iterable[tuple[str, str]]:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return
    items = result if isinstance(result, list) else [result]
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if _ID_KEY.search(str(key)):
                text = _usable(value)
                if text is not None:
                    yield str(key), text


def _unseen_like(rng: random.Random, value: str, seen: set[str]) -> str:
    match = re.search(r"\d+(?=\D*$)", value)
    if match:
        digits = match.group(0)
        width = len(digits)
        for _ in range(100):
            number = (int(digits) + rng.randrange(1, 10**width)) % 10**width
            candidate = value[: match.start()] + f"{number:0{width}d}" + value[match.end() :]
            if candidate not in seen:
                return candidate
    return f"{value}-missing"


def from_traces(traces: Any, *, n: int = DEFAULT_N, seed: int = 0) -> Seeds:
    """Asks about identifiers the agent really used, plus one it never saw.

    Normalizes ``traces`` with ``load_traces`` (any shape it reads: the
    canonical ``steps``, ``tool_trace``, OpenAI or Anthropic ``messages``,
    or a JSONL path), then lifts identifier-shaped values from tool-call
    arguments and from results under id-like keys (``*_id``, ``path``,
    ``sku``, ...). Free text, anything longer than ``MAX_VALUE_CHARS`` and
    anything shaped like an email address is skipped. Trace prompts are
    never copied, so traces held out for evaluation stay out of the
    generated set.

    * ``traces``: rows, a JSONL path, or anything ``load_traces`` takes.
    * ``n`` (``DEFAULT_N``, 12): how many asks, fewer only when the
      material runs out. One in ``NEGATIVE_EVERY``
      (6), at least one, names a value of the same shape that never
      appeared (``A1001`` -> ``A1437``), where the right answer is "not
      found".
    * ``seed`` (0): which values and phrasings are drawn.

    Returns :class:`Seeds`; ``facts`` has ``tools``, ``values``
    (``label value``) and ``absent_values``.

    ```python
    asks = wai.seeds.from_traces("prod_traces.jsonl", n=12, seed=0)
    data = wai.simulate(agent, tools=TOOLS, seeds=asks, simulator=False)
    ```

    Reference: convention, untested (the ask templates).
    """
    from .simulations.ingest.traces import load_traces

    n = _check_n(n, "from_traces")
    rows = load_traces(traces)
    found: set[tuple[str, str, str]] = set()
    for row in rows:
        for step in row.get("steps") or []:
            if not isinstance(step, dict) or "tool" not in step:
                continue
            tool = str(step.get("tool") or "")
            args = step.get("arguments")
            for key, value in args.items() if isinstance(args, dict) else []:
                text = _usable(value)
                if text is not None:
                    found.add((tool, str(key), text))
            for key, text in _result_ids(step.get("result")):
                found.add((tool, key, text))
    if not found:
        raise ValueError(
            "from_traces: no tool call with an identifier-shaped argument or result; "
            "pass traces whose steps carry {'tool', 'arguments', 'result'}"
        )
    rng = random.Random(seed)
    ordered = sorted(found)
    by_tool: dict[str, list[str]] = {}
    seen_values = {value for _, _, value in ordered}
    for tool, key, value in ordered:
        by_tool.setdefault(tool, []).extend(
            tpl.format(label=_label(key), value=value) for tpl in VALUE_ASKS
        )
    negatives: list[str] = []
    absent: list[str] = []
    for _ in range(_n_negative(n)):
        tool, key, value = rng.choice(ordered)
        missing = _unseen_like(rng, value, seen_values | set(absent))
        absent.append(missing)
        negatives.append(rng.choice(VALUE_ASKS).format(label=_label(key), value=missing))
    real = _fill(rng, [by_tool[t] for t in sorted(by_tool)], n - len(negatives))
    asks = real + negatives
    rng.shuffle(asks)
    return Seeds(
        asks,
        source=f"traces, {len(rows)} rows",
        facts={
            "tools": sorted(by_tool),
            "values": [
                f"{_label(k)} {v}" for _, k, v in ordered if any(f" {v}" in a for a in real)
            ],
            "absent_values": absent,
        },
        negatives=negatives,
    )
