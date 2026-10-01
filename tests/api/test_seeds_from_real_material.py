"""wai.seeds: asks from a git repo, a schema or recorded tool calls.

An external case study on whileai 0.126 (Gently Ventures) ran the offline
writer against a code-history agent: 94 of 115 cases carried a ticket id
and none named a file in the repository. Every fixture here is built in
tmp_path with fixed dates, so the numbers below are the same on any
machine.
"""

from __future__ import annotations

import os
import re
import sqlite3
import subprocess
from pathlib import Path

import pytest

import whileai as wai
from tests.helpers import simulate_offline

AUTHOR_NAME = "Ada Lovelace"
AUTHOR_EMAIL = "ada.lovelace@example.com"
COMMITTER_NAME = "Charles Babbage"
COMMITTER_EMAIL = "charles@analytical.example.org"
PRIVATE = (AUTHOR_NAME, AUTHOR_EMAIL, COMMITTER_NAME, COMMITTER_EMAIL, "Ada", "Babbage", "@")

# (month, files written that month). 2026-03 and 2026-04 have no commit.
HISTORY = [
    ("2026-01-15T10:00:00", {"src/app.py": "print('v1')\n", "README.md": "# demo\n"}),
    ("2026-02-03T09:30:00", {"src/app.py": "print('v2')\n", "src/db/models.py": "X = 1\n"}),
    ("2026-02-20T16:45:00", {"docs/guide.md": "guide\n"}),
    ("2026-05-11T12:00:00", {"src/db/models.py": "X = 2\n", "tests/test_app.py": "pass\n"}),
]
FILES = {"src/app.py", "README.md", "src/db/models.py", "docs/guide.md", "tests/test_app.py"}


def _git(repo: Path, *args: str, date: str | None = None) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": AUTHOR_NAME,
        "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL,
        "GIT_COMMITTER_NAME": COMMITTER_NAME,
        "GIT_COMMITTER_EMAIL": COMMITTER_EMAIL,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date + "+00:00"
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=env)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "history-demo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    for i, (date, files) in enumerate(HISTORY):
        for rel, body in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
            _git(root, "add", rel)
        # the message names the author too: it must not leak either
        _git(root, "commit", "-q", "-m", f"change {i} by {AUTHOR_NAME}", date=date)
    return root


@pytest.fixture
def shop_db(tmp_path: Path) -> Path:
    path = tmp_path / "shop.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, email TEXT, region TEXT);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            total_usd REAL,
            status TEXT
        );
        INSERT INTO customers VALUES (1, 'grace.hopper@example.com', 'east');
        """
    )
    con.commit()
    con.close()
    return path


TRACES = [
    {
        "prompt": "Where is my order A1001? Grace Hopper, grace@example.com",
        "steps": [
            {
                "tool": "lookup_order",
                "arguments": {"order_id": "A1001"},
                "result": {"status": "shipped"},
            },
            {
                "tool": "get_refund_status",
                "arguments": {"refund_id": "re_204"},
                "result": {"refund_id": "re_204", "owner_email": "grace@example.com"},
            },
        ],
        "final_text": "Shipped.",
    },
    {
        "messages": [
            {"role": "user", "content": "Order A1002 please"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "lookup_order",
                            "arguments": '{"order_id": "A1002", "note": "customer is upset about the delay"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": '{"order_id": "A1002", "status": "open"}',
            },
        ]
    },
]


def _text(seeds: wai.seeds.Seeds) -> str:
    return "\n".join([str(seeds), repr(list(seeds)), repr(seeds.facts), repr(seeds.negatives)])


# ------------------------------------------------------------------ repo


def test_repo_seeds_name_real_files_and_months(repo: Path):
    seeds = wai.seeds.from_repo(repo, n=12, seed=0)
    assert isinstance(seeds, list) and len(seeds) == 12
    assert seeds.facts["files"] and set(seeds.facts["files"]) <= FILES
    assert set(seeds.facts["months"]) <= {"2026-01", "2026-02", "2026-05"}
    real = [a for a in seeds if a not in seeds.negatives]
    assert sum(any(f in a for f in FILES) for a in real) >= len(real) // 2


def test_repo_seeds_carry_a_file_that_never_existed_and_an_empty_month(repo: Path):
    seeds = wai.seeds.from_repo(repo, n=12, seed=0)
    assert len(seeds.negatives) == 2 and set(seeds.negatives) <= set(seeds)
    (absent,) = seeds.facts["absent_files"]
    assert absent not in FILES and not (repo / absent).exists()
    assert any(absent in a for a in seeds.negatives)
    assert seeds.facts["empty_months"][0] in {"2026-03", "2026-04"}


def test_repo_seeds_never_carry_an_author_name_or_email(repo: Path):
    for seed in range(5):
        text = _text(wai.seeds.from_repo(repo, n=20, seed=seed))
        for needle in PRIVATE:
            assert needle not in text, needle


def test_repo_seeds_are_identical_for_the_same_seed(repo: Path):
    a = wai.seeds.from_repo(repo, n=12, seed=7)
    b = wai.seeds.from_repo(repo, n=12, seed=7)
    assert list(a) == list(b) and a.facts == b.facts and a.negatives == b.negatives
    assert list(wai.seeds.from_repo(repo, n=12, seed=8)) != list(a)


def test_from_repo_names_the_fix(tmp_path: Path):
    with pytest.raises(ValueError, match="git repository"):
        wai.seeds.from_repo(tmp_path)
    with pytest.raises(ValueError, match="n is the number of asks"):
        wai.seeds.from_repo(tmp_path, n=1)


# ---------------------------------------------------------------- schema


def test_schema_seeds_name_real_tables_and_an_absent_one(shop_db: Path):
    seeds = wai.seeds.from_schema(shop_db, n=12, seed=0)
    assert len(seeds) == 12
    assert set(seeds.facts["tables"]) == {"customers", "orders"}
    assert seeds.facts["columns"]
    assert len(seeds.facts["absent_tables"]) == len(seeds.negatives) == 2
    for absent in seeds.facts["absent_tables"]:
        assert absent not in {"customers", "orders"}
        assert any(absent in a for a in seeds.negatives)
    # names only: no row value is read
    assert "grace" not in _text(seeds)
    assert any("orders" in a and "customers" in a for a in seeds)  # the foreign key


def test_schema_from_ddl_text_matches_the_file(shop_db: Path, tmp_path: Path):
    ddl = "CREATE TABLE customers (id INTEGER PRIMARY KEY, email TEXT, region TEXT);\n"
    ddl += "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), total_usd REAL, status TEXT);"
    (tmp_path / "schema.sql").write_text(ddl, encoding="utf-8")
    from_file = wai.seeds.from_schema(shop_db, n=10, seed=3)
    assert list(wai.seeds.from_schema(ddl, n=10, seed=3)) == list(from_file)
    assert list(wai.seeds.from_schema(tmp_path / "schema.sql", n=10, seed=3)) == list(from_file)


def test_schema_seeds_are_identical_for_the_same_seed(shop_db: Path):
    a, b = (wai.seeds.from_schema(shop_db, n=12, seed=1) for _ in range(2))
    assert list(a) == list(b) and a.facts == b.facts


# ---------------------------------------------------------------- traces


def test_trace_seeds_name_ids_the_agent_used_and_one_it_never_saw():
    seeds = wai.seeds.from_traces(TRACES, n=6, seed=0)
    assert len(seeds) == 6
    assert seeds.facts["tools"] == ["get_refund_status", "lookup_order"]
    text = _text(seeds)
    assert "A1001" in text and "A1002" in text and "re_204" in text
    (absent,) = seeds.facts["absent_values"]
    assert absent not in {"A1001", "A1002", "re_204"}
    assert any(absent in a for a in seeds.negatives)
    # no email, no free text, no prompt copied from a trace
    for needle in ("@", "Grace", "upset", "Where is my order"):
        assert needle not in text, needle


def test_trace_seeds_are_identical_for_the_same_seed():
    a, b = (wai.seeds.from_traces(TRACES, n=6, seed=4) for _ in range(2))
    assert list(a) == list(b) and a.facts == b.facts


# ------------------------------------------------- the writer takes them


HISTORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_history",
            "description": "Commits that touched a file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commits_in_month",
            "description": "Commits made in one calendar month.",
            "parameters": {
                "type": "object",
                "properties": {"month": {"type": "string"}},
                "required": ["month"],
            },
        },
    },
]
HISTORY_POLICY = "Answer questions about this repository's files and commit history."


def _history_agent(message: str) -> dict:
    hit = next((f for f in sorted(FILES) if f in message), None)
    steps = [
        {
            "tool": "file_history",
            "arguments": {"path": hit or ""},
            "result": {"commits": 1 if hit else 0},
        }
    ]
    return {"steps": steps, "final_text": "found" if hit else "not found"}


TICKET_ID = re.compile(r"\b[A-Z]{2,5}-\d{3,6}\b")


def _cases(seeds: list[str] | None) -> tuple[int, int, int]:
    """(cases naming a tracked file, cases carrying a ticket-style id, cases)."""
    data = simulate_offline(
        _history_agent,
        tools=HISTORY_TOOLS,
        policy=HISTORY_POLICY,
        seeds=seeds,
        budget=24,
        repeats=1,
        mode="sft",
    )
    asks = sorted({r["prompt"] for r in data.trajectories})
    real = sum(any(f in a for f in FILES) for a in asks)
    tickets = sum(bool(TICKET_ID.search(a)) for a in asks)
    return real, tickets, len(asks)


def test_repo_seeds_raise_the_share_of_cases_naming_a_real_file(repo: Path):
    """The case study's measurement on a fixed repo and the same budget:
    without seeds the offline writer names no file and hands a code-history
    agent ticket ids; with repo seeds a third of the cases name a tracked
    file and fewer carry a ticket id."""
    before = _cases(None)
    after = _cases(wai.seeds.from_repo(repo, n=12, seed=0))
    for label, (real, tickets, total) in (("without seeds", before), ("with from_repo", after)):
        print(f"\n{label}: {real}/{total} name a real file, {tickets}/{total} carry a ticket id")
    assert before[0] == 0 and before[2] == after[2]
    # the ten real seeds, eight about a file, all run, and each replaces a
    # template case
    assert after[0] == 8
    assert after[1] < before[1]


def test_seeds_print_what_they_read(repo: Path):
    out = str(wai.seeds.from_repo(repo, n=6, seed=0))
    assert out.startswith("6 seeds from repo history-demo")
    assert re.search(r"^  - ", out, re.M) and re.search(r"^  \+ ", out, re.M)
