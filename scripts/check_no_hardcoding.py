#!/usr/bin/env python3
"""Fail when a number that steers behaviour is written inline.

Every threshold, share, cap and budget in ``whileai/simulations`` has one
home: ``defaults.py`` (values more than one module reads, or that a
``simulate()`` key moves) or a named module constant with a comment that
says why. This check reads every module under ``whileai/simulations``
except ``defaults.py`` and reports each numeric literal that is

* an operand of a comparison (``x < 0.35``), or
* the value of an assignment (``rate = 0.35``, ``x: float = 0.35``,
  ``n += 4``),

other than ``0``, ``1``, ``2`` and ``-1`` (loop seeds, counts, the
structural minimums) and other than indices and slices, which are
positions, not thresholds. Ruff's ``PLR2004`` (enabled in pyproject)
covers the shapes this script does not read: a default argument, a call
argument, a return expression, a ``min``/``max`` operand, a dict value.

A module-level ``NAME = value`` is a named constant and passes only when
a comment documents it in one of two forms:

* ``# NAME = value: why (source)`` anywhere in the module, the form
  ``defaults.py`` uses (one comment may name several constants, each
  with its own ``NAME = value`` and one shared ``: why``), or
* a ``#:`` attribute-doc line directly above it (a contiguous block of
  ``#:`` lines may document a group of constants), or trailing on its
  line.

A plain ``#`` comment above or beside a constant does not count: the
old rule let any remark pass as a reason. ``defaults.py`` itself is held
to the strict ``# NAME = value: why`` form by
``tests/grade/test_no_hardcoding_score.py``.

A line that must keep a literal ends with ``# literal: <reason>`` and
passes when the reason is a phrase: at least 12 characters and more than
one word. ``# literal: x`` or ``# literal: heuristic`` does not pass.

``defaults.py`` is read for one thing: a default with no source says the
exact words ``(convention, untested`` (CONSTITUTION.md, belief 3 and rule
12), so one grep finds every unsourced number. A comment that opens
``(convention`` any other way, ``(convention)`` or ``(convention inside
the band, ...)``, is a finding. Wrapped comment lines are joined before
the check, so the phrase may break across a line.

    uv run python scripts/check_no_hardcoding.py          # exit 1 on a finding
    uv run python scripts/check_no_hardcoding.py --list   # print, exit 0

Tests, recipes and scripts are not scanned: a test writes the number it
checks, and a recipe is a worked example.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "whileai" / "simulations"
SKIP_FILES = {"defaults.py"}

# The values that are structure, not tuning: an empty count, a unit step,
# a pair, and the last index.
FREE_VALUES = {0, 1, 2, -1}

#: A ``# literal:`` reason has to be a phrase this long, in characters.
LITERAL_REASON_MIN_CHARS = 12

ALLOW_MARK = re.compile(r"#\s*literal:\s*(?P<reason>.+?)\s*$")
#: The words a default with no source says, verbatim, from the open paren.
CONVENTION_PHRASE = "(convention, untested"
CONVENTION_MARK = re.compile(r"\(convention\b")
CONSTANT_LINE = re.compile(r"^_?[A-Z][A-Z0-9_]*(?:, _?[A-Z][A-Z0-9_]*)*(?::[^=]+)? = ")
ATTR_DOC_LINE = re.compile(r"^#:\s*\S")


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    text: str

    def __str__(self) -> str:
        rel = self.path.relative_to(REPO_ROOT).as_posix()
        return f"{rel}:{self.line}: {self.text}"


def _literal_value(node: ast.AST) -> int | float | None:
    """The numeric value of a plain literal or ``-literal``; None otherwise."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal_value(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            return None
        return node.value
    return None


def literal_reason_ok(reason: str) -> bool:
    """A ``# literal:`` reason counts when it is a phrase: long enough and
    more than one word."""
    text = reason.strip()
    return len(text) >= LITERAL_REASON_MIN_CHARS and len(text.split()) > 1


def _line_allows(line: str) -> bool:
    m = ALLOW_MARK.search(line)
    return bool(m) and literal_reason_ok(m.group("reason"))


def _explained_names(source_lines: list[str]) -> set[str]:
    """Names a comment documents in the ``# NAME = value: why`` form.

    The colon and the why are required; ``# NAME = value`` alone is a
    restatement, not a reason. One comment line may carry several names
    (``# A = 1 / B = 2: why``); each name before the colon is explained.
    """
    names: set[str] = set()
    for line in source_lines:
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        head, colon, why = stripped.partition(":")
        if not colon or not why.strip():
            continue
        for m in re.finditer(r"(?<![A-Za-z0-9_])([A-Z_][A-Z0-9_]*) = ", head):
            names.add(m.group(1))
    return names


def _target_names(node: ast.stmt) -> list[str]:
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Tuple):
            names.extend(e.id for e in target.elts if isinstance(e, ast.Name))
    return names


def _assigned_literals(value: ast.expr) -> list[ast.expr]:
    """The literal nodes an assignment's value is made of: the value
    itself, or each element of a tuple/list of literals."""
    if _literal_value(value) is not None:
        return [value]
    if isinstance(value, (ast.Tuple, ast.List)):
        return [e for e in value.elts if _literal_value(e) is not None]
    return []


def check_file(path: Path) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    explained = _explained_names(lines)
    findings: list[Finding] = []
    module_level = {id(stmt) for stmt in tree.body}

    # statement spans, innermost first, so a ``# literal:`` mark counts
    # anywhere in the statement the literal sits in (the formatter may
    # wrap a marked line and leave the comment on its last line)
    spans = sorted(
        (
            (node.lineno, node.end_lineno or node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.stmt)
        ),
        key=lambda span: span[1] - span[0],
    )

    def allowed_line(lineno: int) -> bool:
        if _line_allows(lines[lineno - 1]):
            return True
        for start, end in spans:
            if start <= lineno <= end:
                return any(_line_allows(lines[i - 1]) for i in range(start, end + 1))
        return False

    def _documented(name: str, lineno: int) -> bool:
        """A module constant is documented by a ``# NAME = value: why``
        comment anywhere in the module, or by a ``#:`` attribute doc:
        a contiguous block of ``#:`` lines directly above it (sibling
        ``NAME = value`` lines in between are allowed, so one block can
        explain a group) or trailing on its own line."""
        if name.lstrip("_") in explained or name in explained:
            return True
        trailing = lines[lineno - 1].partition("#")[2]
        if trailing.startswith(":") and trailing[1:].strip():
            return True
        i = lineno - 2
        while i >= 0:
            stripped = lines[i].strip()
            if ATTR_DOC_LINE.match(stripped):
                return True
            if stripped.startswith("#") or CONSTANT_LINE.match(lines[i]):
                i -= 1
                continue
            break
        return False

    def report(node: ast.AST) -> None:
        if allowed_line(node.lineno):
            return
        findings.append(Finding(path, node.lineno, lines[node.lineno - 1].strip()))

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for operand in [node.left, *node.comparators]:
                value = _literal_value(operand)
                if value is not None and value not in FREE_VALUES:
                    report(operand)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if node.value is None:
                continue
            literals = _assigned_literals(node.value)
            if not literals:
                continue
            if any(_literal_value(lit) in FREE_VALUES for lit in literals) and len(literals) == 1:
                continue
            names = _target_names(node)
            is_module_constant = id(node) in module_level and all(
                n.lstrip("_").isupper() for n in names
            )
            if is_module_constant and all(_documented(n, node.lineno) for n in names):
                continue
            for lit in literals:
                value = _literal_value(lit)
                if value is not None and value not in FREE_VALUES:
                    report(node)
                    break
    return findings


def check_convention_phrase(path: Path) -> list[Finding]:
    """Every ``(convention`` in a comment of ``path`` opens the exact
    phrase ``(convention, untested``. Comment lines are joined into
    paragraphs first, so a phrase wrapped over two lines is read whole;
    a finding names the line the marker starts on."""
    lines = path.read_text(encoding="utf-8").splitlines()
    findings: list[Finding] = []
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("#"):
            i += 1
            continue
        start = i
        parts: list[str] = []
        while i < len(lines) and lines[i].lstrip().startswith("#"):
            parts.append(lines[i].lstrip().lstrip("#").strip())
            i += 1
        paragraph = " ".join(parts)
        for m in CONVENTION_MARK.finditer(paragraph):
            if paragraph.startswith(CONVENTION_PHRASE, m.start()):
                continue
            lineno, seen = start + 1, 0
            for offset, part in enumerate(parts):
                if seen + len(part) >= m.start():
                    lineno = start + offset + 1
                    break
                seen += len(part) + 1
            snippet = paragraph[m.start() : m.start() + 40]
            findings.append(Finding(path, lineno, f'"{snippet}" is not "{CONVENTION_PHRASE}"'))
    return findings


def main(argv: list[str]) -> int:
    list_only = "--list" in argv
    findings: list[Finding] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name in SKIP_FILES:
            continue
        findings.extend(check_file(path))
    for finding in findings:
        print(finding)
    if findings:
        print(
            f"\n{len(findings)} inline number(s). Move each to defaults.py or a named "
            "module constant with a `# NAME = value: why (source)` comment (or a `#:` "
            "attribute doc above it), or end the line with `# literal: <reason>` where "
            "the reason is a phrase."
        )
    phrase = check_convention_phrase(PACKAGE / "defaults.py")
    for finding in phrase:
        print(finding)
    if phrase:
        print(
            f"\n{len(phrase)} default(s) in defaults.py say convention without the exact "
            f'words. A default with no source says "{CONVENTION_PHRASE}" (CONSTITUTION.md, '
            "rule 12); qualify it after those words, not instead of them."
        )
    if findings or phrase:
        return 0 if list_only else 1
    print("no inline thresholds in whileai/simulations; every convention default says untested")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
