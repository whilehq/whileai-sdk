"""``whileai.config.provenance()``: which ``whileai`` a run imported (#443).

A clone's ``whileai/`` folder shadows the installed wheel when a recipe
starts from the repository root, with no message. The line names the
directory and, by path alone, says whether it is an install, an editable
install of the tree, or a checkout shadowing the wheel.
"""

from __future__ import annotations

import re
from pathlib import Path

import whileai
from whileai.config import _provenance_line, provenance

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / "recipes"

TREE = Path("/home/me/whileai-sdk/whileai")
WHEEL = Path("/home/me/.venv/lib/python3.12/site-packages/whileai")
DEBIAN = Path("/usr/lib/python3/dist-packages/whileai")


def test_installed_wheel_has_no_suffix() -> None:
    assert _provenance_line("0.110", WHEEL, None) == f"whileai 0.110 from {WHEEL}"
    assert _provenance_line("0.110", DEBIAN, None) == f"whileai 0.110 from {DEBIAN}"
    # an editable root elsewhere does not change what a site-packages path is
    assert _provenance_line("0.110", WHEEL, TREE.parent) == f"whileai 0.110 from {WHEEL}"


def test_checkout_shadowing_the_wheel_says_so() -> None:
    assert _provenance_line("0.110", TREE, None) == (
        f"whileai 0.110 from {TREE} (source tree, not the installed wheel)"
    )
    # editable, but of another clone: this tree is still not the installed one
    other = Path("/home/me/other-clone")
    assert _provenance_line("0.110", TREE, other).endswith("(source tree, not the installed wheel)")


def test_editable_install_of_this_tree() -> None:
    assert _provenance_line("0.110", TREE, TREE.parent) == (
        f"whileai 0.110 from {TREE} (source tree, installed editable)"
    )


def test_provenance_names_the_imported_package() -> None:
    line = provenance()
    where = Path(whileai.__file__).resolve().parent
    assert line.startswith(f"whileai {whileai.__version__} from {where}")
    assert line.count("\n") == 0


def _entrypoints() -> list[Path]:
    """Recipe scripts a reader runs that use the SDK on the local side: a
    top-level ``whileai`` import, or a Modal image that mounts the local
    ``whileai`` (``add_local_python_source``). ``papers/check.py`` counts
    too: since #616 it imports the package's ``noise_band`` from the
    checkout instead of carrying a copy, so it says which one it used."""
    out: list[Path] = []
    for path in sorted(RECIPES.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if '__name__ == "__main__"' not in text and "local_entrypoint" not in text:
            continue
        uses_sdk = re.search(r"^(import|from) whileai\b", text, re.M) or (
            'add_local_python_source("whileai")' in text
        )
        if uses_sdk:
            out.append(path)
    return out


def test_every_recipe_entrypoint_prints_it() -> None:
    """The line is the fix for #443 only if every entrypoint prints it, first
    and on stderr, so stdout stays the result a test or a pipe reads."""
    paths = _entrypoints()
    assert len(paths) >= 30, paths
    missing = [
        p.relative_to(ROOT).as_posix()
        for p in paths
        if "print(provenance(), file=sys.stderr)" not in p.read_text(encoding="utf-8")
    ]
    assert missing == [], missing
