"""The verifier: run the candidate SQL on the store database, match the gold
query's result set. Plus the small shared pieces every script needs (tasks,
split, rows) and, for the Modal trainer, a Postgres that starts inside the
container.

`SQLExec` is a `whileai.simulations.verify.Verifier`, so it is the judge
for `data.grade(judge=SQLExec())`, `evaluate`, `optimize` and a gated push.
The gold is read from `privileged.reference`, which the training export never
projects. The match is Spider-style execution accuracy: same multiset of rows
(same sequence when the gold has ORDER BY), floats rounded to 2 places, text
case-folded, column names ignored, columns in any order.
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import re
import subprocess
import threading
import time
from collections import Counter
from datetime import date, datetime, timedelta
from datetime import time as dtime
from decimal import Decimal
from pathlib import Path
from typing import Any

from schema_prompt import NOTES, ddl, system_prompt  # noqa: F401  (re-exported)

from whileai.simulations.verify.base import Verifier

HERE = Path(__file__).resolve().parent
DSN = os.environ.get("T2S_PG_DSN") or "postgresql://postgres@127.0.0.1:5499/shop"
TASKS = HERE / "tasks.jsonl"
RAW = HERE / "raw"
OUT = HERE / "out"
AGENT = "text-to-sql-shop"
HOLDOUT = 0.2
MAX_ROWS = 200
STATEMENT_TIMEOUT_MS = int(os.environ.get("T2S_STATEMENT_TIMEOUT_MS") or 5000)

TABLES = [
    "categories",
    "products",
    "customers",
    "employees",
    "orders",
    "order_items",
    "payments",
    "reviews",
]

# What author.py asks the task writer for: one question per (archetype,
# difficulty) cell, phrased in a rotating style.
ARCHETYPES = [
    "single-table aggregation (COUNT/SUM/AVG/MIN/MAX with WHERE filters)",
    "top-N / ranking with ORDER BY and LIMIT",
    "two-table JOIN with a filter or aggregate",
    "multi-table JOIN (three or more tables)",
    "GROUP BY breakdown with HAVING or a per-group aggregate",
    "NULL semantics (shipped_at, sales_rep_id, discount_pct, referred_by, review title)",
    "self-join / hierarchy (category parent, employee manager, customer referrer)",
    "date and time (DATE_TRUNC, EXTRACT, month or year buckets, intervals, first/last event)",
    "anti-join / existence (customers with no orders, products never reviewed, orders without a captured payment, NOT EXISTS / NOT IN / LEFT JOIN IS NULL)",
    "derived metric (order total with discount and shipping, margin, average rating, share of total, days to ship)",
]
DIFFICULTIES = ["easy", "medium", "hard"]
STYLES = [
    "casual business user asking a quick question",
    "formal reporting request from a manager",
    "terse power-user shorthand",
    "precise analyst specification that may name columns",
]

_SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)
_SELECT_START = re.compile(r"\b(select|with)\b", re.I)
_tls = threading.local()


# ----------------------------------------------------------------- execution


def extract_sql(text: str) -> str | None:
    """The last fenced query in a reply, else the text from its first SELECT/WITH."""
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = _SQL_BLOCK.findall(text)
    sql = blocks[-1] if blocks else None
    if sql is None:
        m = _SELECT_START.search(text)
        if not m:
            return None
        sql = text[m.start() :]
    sql = sql.strip().rstrip(";").strip()
    if not _SELECT_START.match(sql):
        return None
    return sql


def _conn():
    import psycopg

    conn = getattr(_tls, "conn", None)
    if conn is None or conn.closed:
        conn = psycopg.connect(DSN, autocommit=True)
        conn.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
        conn.execute("SET default_transaction_read_only = on")
        _tls.conn = conn
    return conn


def run_sql(sql: str, limit: int = MAX_ROWS + 1) -> list[tuple]:
    """Execute one read-only SELECT. Raises on error."""
    if not _SELECT_START.match(sql.strip()):
        raise ValueError("not a SELECT")
    if ";" in sql.rstrip(";"):
        raise ValueError("one statement only")
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)  # type: ignore[arg-type]
            if cur.description is None:
                return []
            return cur.fetchmany(limit)
    except Exception:
        if conn.closed or getattr(conn, "broken", False):
            _tls.conn = None
        raise


def _norm(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, Decimal):
        v = float(v)
    if isinstance(v, (int, float)):
        r = round(float(v), 2)
        return 0.0 if r == 0 else r
    if isinstance(v, datetime):
        return v.isoformat(sep=" ")
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, dtime):
        return v.isoformat()
    if isinstance(v, timedelta):
        return round(v.total_seconds() / 86400, 2)
    if isinstance(v, str):
        s = v.strip().lower()
        try:
            return _norm(float(s.replace(",", "").replace("$", "")))
        except ValueError:
            return s
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    return str(v)


def norm_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_norm(v) for v in r) for r in rows]


def has_order_by(sql: str) -> bool:
    return re.search(r"\border\s+by\b", sql, re.I) is not None


def _key(t: tuple) -> str:
    return json.dumps(t, default=str)


def equivalent(cand: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    if len(cand) != len(gold):
        return False
    if not gold:
        return True
    ncols = len(gold[0])
    if any(len(r) != ncols for r in cand):
        return False
    perms = [tuple(range(ncols))]
    if ncols <= 5:
        perms += [p for p in itertools.permutations(range(ncols)) if p != perms[0]]
    for perm in perms:
        c = [tuple(r[i] for i in perm) for r in cand]
        if ordered:
            if c == gold:
                return True
        elif Counter(map(_key, c)) == Counter(map(_key, gold)):
            return True
    return False


_gold_cache: dict[str, list[tuple]] = {}
_gold_lock = threading.Lock()


def gold_rows(sql: str) -> list[tuple]:
    h = hashlib.sha256(sql.encode()).hexdigest()
    with _gold_lock:
        if h in _gold_cache:
            return _gold_cache[h]
    rows = norm_rows(run_sql(sql))
    with _gold_lock:
        _gold_cache[h] = rows
    return rows


def verdict(text: str, gold_sql: str) -> tuple[int, int, str]:
    """(correct 0/1, executes 0/1, reason)."""
    sql = extract_sql(text)
    if not sql:
        return 0, 0, "no sql query in reply"
    try:
        got = norm_rows(run_sql(sql))
    except Exception as exc:
        return 0, 0, f"sql error: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
    if len(got) > MAX_ROWS:
        return 0, 1, f"result too large (>{MAX_ROWS} rows)"
    want = gold_rows(gold_sql)
    ok = equivalent(got, want, ordered=has_order_by(gold_sql))
    if ok:
        return 1, 1, f"result matches ({len(want)} rows)"
    got_cols = len(got[0]) if got else 0
    want_cols = len(want[0]) if want else 0
    return (
        0,
        1,
        f"result differs: got {len(got)} rows x {got_cols} cols, want {len(want)} x {want_cols}",
    )


class SQLExec(Verifier):
    """Reward 1 when the candidate's result matches the gold's result."""

    kind = "rule"

    def __init__(self):
        super().__init__(name="sql_exec")

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        gold_sql = reference if isinstance(reference, str) else (reference or {}).get("sql")
        if not gold_sql:
            return None, "no gold sql"
        correct, _executes, reason = verdict(candidate, gold_sql)
        return correct, reason


def shaped_reward(text: str, gold_sql: str) -> float:
    """Training reward: 1.0 correct, 0.1 runs but wrong, 0 no query or error."""
    correct, executes, _ = verdict(text, gold_sql)
    if correct:
        return 1.0
    return 0.1 if executes else 0.0


# ----------------------------------------------------------------- tasks, split, rows


def task_id(question: str) -> str:
    return "t2s_" + hashlib.sha256(question.strip().lower().encode()).hexdigest()[:12]


def load_tasks(path: Path = TASKS) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def bucket(scenario_id: str) -> float:
    h = hashlib.sha256(str(scenario_id).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_of(scenario_id: str) -> str:
    return "holdout" if bucket(scenario_id) < HOLDOUT else "train"


def make_row(
    task: dict,
    rollout_index: int,
    model: str,
    reply: str,
    *,
    finish_reason: str | None = None,
    usage: dict | None = None,
    latency_s: float | None = None,
) -> dict:
    sys_p = system_prompt()
    return {
        "scenario_id": task["id"],
        "rollout_index": rollout_index,
        "prompt": task["question"],
        "system_prompt": sys_p,
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": task["question"]},
            {"role": "assistant", "content": reply},
        ],
        "final_text": reply,
        "steps": [],
        "privileged": {"reference": task["sql"]},
        "model_version": model,
        "category": task["archetype"],
        "difficulty": task["difficulty"],
        "style": task.get("style"),
        "split": split_of(task["id"]),
        "agent": AGENT,
        "truncated": finish_reason == "length",
        "finish_reason": finish_reason,
        "usage": usage,
        "latency_s": latency_s,
    }


def teacher_row(task: dict) -> dict:
    """The gold SQL as a demonstration (a Rollout by the teacher, keyed by task)."""
    reply = f"```sql\n{task['sql']}\n```"
    row = make_row(task, 0, "teacher:gold", reply, finish_reason="stop")
    row["reward"] = 1
    row["reason"] = "gold sql"
    row["judge_status"] = "ok"
    row["judge_name"] = "sql_exec"
    return row


def reward_rows(tasks: list[dict], replies: list[list[str]], model: str) -> list[dict]:
    """Graded SDK rows from k replies per task (the trainer's in-container eval)."""
    rows = []
    for task, group in zip(tasks, replies):
        for i, text in enumerate(group):
            correct, executes, reason = verdict(text, task["sql"])
            row = make_row(task, i, model, text)
            row.update(
                {
                    "reward": correct,
                    "reason": reason,
                    "judge_status": "ok",
                    "judge_name": "sql_exec",
                    "markers": {
                        "executes": executes,
                        "has_sql": int(extract_sql(text) is not None),
                    },
                }
            )
            rows.append(row)
    return rows


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


# ----------------------------------------------------------------- postgres in a container


def start_postgres(
    schema_sql: str, seed_sql: str, data_dir: str = "/tmp/pgdata", port: int = 5499
) -> None:
    """initdb + start a trust-auth cluster as the postgres user, load the store.
    For the Modal trainer (Debian image with the postgresql package)."""
    bins = sorted(glob.glob("/usr/lib/postgresql/*/bin"))
    if not bins:
        raise RuntimeError("no postgresql binaries in the image")
    pgbin = bins[-1]
    subprocess.run(["chown", "-R", "postgres:postgres", os.path.dirname(data_dir)], check=False)
    if not os.path.exists(os.path.join(data_dir, "PG_VERSION")):
        subprocess.run(
            [
                "su",
                "postgres",
                "-c",
                f"{pgbin}/initdb -D {data_dir} -A trust -E UTF8 >/tmp/initdb.log 2>&1",
            ],
            check=True,
        )
    subprocess.run(
        [
            "su",
            "postgres",
            "-c",
            f"{pgbin}/pg_ctl -D {data_dir} -o '-p {port} -c listen_addresses=127.0.0.1' -l /tmp/pg.log -w start",
        ],
        check=True,
    )
    for _ in range(30):
        ready = subprocess.run(
            ["su", "postgres", "-c", f"{pgbin}/pg_isready -h 127.0.0.1 -p {port}"],
            capture_output=True,
        )
        if ready.returncode == 0:
            break
        time.sleep(1)
    subprocess.run(
        ["su", "postgres", "-c", f"{pgbin}/createdb -h 127.0.0.1 -p {port} shop"], check=False
    )
    Path("/tmp/schema.sql").write_text(schema_sql, encoding="utf-8")
    Path("/tmp/seed.sql").write_text(seed_sql, encoding="utf-8")
    subprocess.run(
        [
            "su",
            "postgres",
            "-c",
            f"{pgbin}/psql -q -v ON_ERROR_STOP=1 -h 127.0.0.1 -p {port} -d shop -f /tmp/schema.sql -f /tmp/seed.sql",
        ],
        check=True,
    )
    n = run_sql("select count(*) from orders")[0][0]
    print(f"postgres up on {port}: {n} orders")


# ----------------------------------------------------------------- selftest

MIN_TASKS = 400  # the shipped set is 741 tasks; fewer means a truncated file
HOLDOUT_BAND = (0.15, 0.25)  # the hash split lands within a quarter of HOLDOUT either way


def selftest() -> None:
    """The matching rule on rows with known answers, the shipped task set and
    the prompt file the trainer mounts. No database, no key, no GPU."""
    assert extract_sql("<think>plan</think>\n```sql\nSELECT 1\n```\n```sql\nSELECT 2;\n```") == (
        "SELECT 2"
    )
    assert extract_sql("no query here") is None
    assert extract_sql("```sql\nDELETE FROM orders\n```") is None
    gold = norm_rows([("US", 40), ("CA", 15)])
    assert equivalent(norm_rows([(15, "ca"), (40, "US")]), gold, ordered=False)
    assert not equivalent(norm_rows([("US", 40)]), gold, ordered=False)
    assert not equivalent(norm_rows([("CA", 15), ("US", 40)]), gold, ordered=True)
    assert equivalent(norm_rows([("US", 40.004), ("CA", 15)]), gold, ordered=True)
    assert _norm(Decimal("12.345")) == 12.35 and _norm("  Chase ") == "chase"
    print("matching rule: ok")

    tasks = load_tasks()
    assert len(tasks) >= MIN_TASKS, f"{len(tasks)} tasks in {TASKS.name}"
    assert len({t["id"] for t in tasks}) == len(tasks), "duplicate task id"
    for t in tasks:
        assert {"id", "question", "sql", "archetype", "difficulty"} <= set(t), t.get("id")
        assert _SELECT_START.match(t["sql"].strip()), t["id"]
    share = sum(1 for t in tasks if split_of(t["id"]) == "holdout") / len(tasks)
    assert HOLDOUT_BAND[0] <= share <= HOLDOUT_BAND[1], f"holdout share {share:.2f}"
    print(f"tasks: {len(tasks)} well formed, holdout share {share:.2f}")

    assert (HERE / "prompt.txt").read_text(encoding="utf-8") == system_prompt()
    assert "CREATE TABLE orders" in system_prompt()
    print("prompt.txt matches schema_prompt.system_prompt()")


if __name__ == "__main__":
    import argparse
    import sys

    from whileai.config import provenance

    print(provenance(), file=sys.stderr)
    ap = argparse.ArgumentParser(description=selftest.__doc__)
    ap.add_argument("--selftest", action="store_true", help="what smoke.sh runs")
    if ap.parse_args().selftest:
        selftest()
        print("selftest ok")
    else:
        ap.print_help()
