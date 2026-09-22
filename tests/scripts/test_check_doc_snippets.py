"""scripts/check_doc_snippets.py: a recipe page runs in its recipe directory, offline.

A page under docs/recipes/ is generated from recipes/<step>/<name>/README.md
and excerpts a script that runs inside that directory, so the checker maps
the page back to the directory, copies it, and runs the page's blocks with
the copy as the working directory and first on sys.path. Every page, recipe
or not, runs with network egress refused at the socket, so a public endpoint
that needs no key cannot slip through.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_doc_snippets.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_doc_snippets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered first: the script's dataclasses resolve their (string)
    # annotations through sys.modules[module.__name__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------- page -> directory


def test_recipe_page_maps_to_its_recipe_directory(checker):
    page = checker.DOCS / "recipes" / "02-measure" / "eval-your-agent.mdx"
    assert checker.recipe_dir(page) == checker.RECIPES / "02-measure" / "eval-your-agent"


@pytest.mark.parametrize(
    "rel",
    [
        "recipes/index.mdx",  # the section index: no directory of its own
        "recipes/community/index.mdx",
        "evals.md",  # not a recipe page
        "reference/platform.md",
    ],
)
def test_pages_without_a_recipe_directory_get_none(checker, rel):
    assert checker.recipe_dir(checker.DOCS / rel) is None


def test_recipes_tree_is_checked_and_api_tree_is_not(checker):
    rels = {p.relative_to(checker.DOCS).as_posix() for p in checker.pages(None)}
    assert any(r.startswith("recipes/") for r in rels)
    assert not any(r.startswith("api/") for r in rels)


# ------------------------------------------------- a page run in its recipe


@pytest.fixture
def fake_repo(checker, tmp_path, monkeypatch):
    """A docs/ and recipes/ tree of one recipe, with a sibling module and a data file."""
    recipe = tmp_path / "recipes" / "01-simulate" / "demo"
    recipe.mkdir(parents=True)
    (recipe / "README.md").write_text("# Demo\n", encoding="utf-8")
    (recipe / "sibling.py").write_text("NAME = 'from the recipe'\n", encoding="utf-8")
    (recipe / "rows.jsonl").write_text('{"prompt": "hi"}\n', encoding="utf-8")
    docs = tmp_path / "docs" / "recipes" / "01-simulate"
    docs.mkdir(parents=True)
    monkeypatch.setattr(checker, "REPO", tmp_path)
    monkeypatch.setattr(checker, "DOCS", tmp_path / "docs")
    monkeypatch.setattr(checker, "RECIPES", tmp_path / "recipes")
    return tmp_path


def _run(checker, page: Path):
    blocks = checker.parse(page)
    return checker.run_page(page, blocks, sys.executable, verbose=False)


def test_recipe_blocks_run_in_a_copy_of_the_recipe_with_it_first_on_sys_path(checker, fake_repo):
    page = fake_repo / "docs" / "recipes" / "01-simulate" / "demo.mdx"
    page.write_text(
        "---\ntitle: Demo\n---\n\n"
        "```python\n"
        "import os, sys\n"
        "from sibling import NAME\n"
        "print(NAME)\n"
        "print(open('rows.jsonl').read().strip())\n"
        "print(os.path.basename(os.getcwd()))\n"
        "assert sys.path[0] == os.getcwd(), sys.path[:2]\n"
        "open('written.txt', 'w').write('x')\n"
        "```\n",
        encoding="utf-8",
    )
    (result,) = _run(checker, page)
    assert result.status == "ok", result.detail
    assert result.stdout.splitlines() == ["from the recipe", '{"prompt": "hi"}', "demo"]
    # It ran in a copy: what the block wrote never landed in the tree.
    assert not (fake_repo / "recipes" / "01-simulate" / "demo" / "written.txt").exists()


def test_a_plain_page_still_runs_in_a_scratch_directory(checker, fake_repo):
    page = fake_repo / "docs" / "plain.md"
    page.write_text(
        "# Plain\n\n```python\nimport os\nprint(os.path.basename(os.getcwd()))\n```\n",
        encoding="utf-8",
    )
    (result,) = _run(checker, page)
    assert result.status == "ok", result.detail
    assert result.stdout.strip().startswith("docsnip-")


# ------------------------------------------------------------- no egress


def test_a_block_that_reaches_the_network_is_refused_before_it_leaves(checker, fake_repo):
    page = fake_repo / "docs" / "net.md"
    page.write_text(
        "# Net\n\n```python\n"
        "import urllib.request\n"
        "urllib.request.urlopen('https://example.com/', timeout=10)\n"
        "```\n",
        encoding="utf-8",
    )
    (result,) = _run(checker, page)
    assert result.status == "failed"
    assert result.detail.startswith("this block reaches the network, and nothing above it")
    assert "blocks network egress" in result.detail
    assert "example.com" in result.detail


def test_a_network_block_is_excused_only_where_the_page_names_the_key(checker, fake_repo):
    page = fake_repo / "docs" / "declared.md"
    page.write_text(
        "# Declared\n\nThis block reaches the platform, so set `WHILEAI_API_KEY` first.\n\n"
        "```python\n"
        "import urllib.request\n"
        "urllib.request.urlopen('https://api.while.ai/', timeout=10)\n"
        "```\n",
        encoding="utf-8",
    )
    (result,) = _run(checker, page)
    assert result.status == "skipped"
    assert result.detail == "reaches the network; page declares WHILEAI_API_KEY"


def test_loopback_is_not_egress(checker, fake_repo):
    page = fake_repo / "docs" / "local.md"
    page.write_text(
        "# Local\n\n```python\n"
        "import socket\n"
        "assert socket.getaddrinfo('localhost', 80)\n"
        "assert socket.getaddrinfo('127.0.0.1', 80)\n"
        "try:\n"
        "    socket.getaddrinfo('example.com', 443)\n"
        "except BaseException as e:  # the refusal is not an OSError, on purpose\n"
        "    assert 'egress' in str(e), e\n"
        "else:\n"
        "    raise AssertionError('a remote name resolved')\n"
        "print('ok')\n"
        "```\n",
        encoding="utf-8",
    )
    (result,) = _run(checker, page)
    assert result.status == "ok", result.detail
    assert result.stdout.strip() == "ok"
