"""Shared helpers for the example tests: load a script by path, stand in for
the ``modal`` client, and read a README's flags and defaults.

The GPU scripts (``train_modal.py`` and friends) cannot run here, but they
can import: with a stub ``modal`` in ``sys.modules`` the module body runs
up to the ``local_entrypoint``, which is where an import rots. The
entrypoint's signature is then the source of truth for every ``--flag`` a
README names and every default a README's knobs table states.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "recipes"


def load_script(name: str, path: Path) -> types.ModuleType:
    """Import ``path`` as module ``name`` (registered, so bare-name imports
    between example files resolve the way they do on Modal)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Chain:
    """Anything: ``Image.debian_slim().pip_install().env()...`` returns itself."""

    def __getattr__(self, _name: str) -> Any:
        return lambda *a, **k: self

    def __call__(self, *a: Any, **k: Any) -> _Chain:
        return self


def install_modal_stub() -> types.ModuleType:
    """A ``modal`` module whose decorators return the function unchanged."""
    existing = sys.modules.get("modal")
    if existing is not None and getattr(existing, "_whileai_stub", False):
        return existing
    modal = types.ModuleType("modal")
    modal._whileai_stub = True  # type: ignore[attr-defined]

    class App:
        def __init__(self, *a: Any, **k: Any) -> None:
            self.name = a[0] if a else k.get("name")

        def function(self, *a: Any, **k: Any):
            return lambda fn: fn

        def local_entrypoint(self, *a: Any, **k: Any):
            return lambda fn: fn

    class Image:
        debian_slim = staticmethod(lambda *a, **k: _Chain())

    class Volume:
        from_name = staticmethod(lambda *a, **k: _Chain())

    class Secret:
        from_dict = staticmethod(lambda *a, **k: _Chain())

    modal.App = App  # type: ignore[attr-defined]
    modal.Image = Image  # type: ignore[attr-defined]
    modal.Volume = Volume  # type: ignore[attr-defined]
    modal.Secret = Secret  # type: ignore[attr-defined]
    sys.modules["modal"] = modal
    return modal


def load_modal_script(name: str, path: Path) -> types.ModuleType:
    install_modal_stub()
    return load_script(name, path)


_FLAG = re.compile(r"--([a-z][a-z0-9-]*)")
_ROW = re.compile(r"^\| `(--[a-z][a-z0-9-]*)` \| ([^|]*) \|", re.M)


def _param(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def readme_flags(readme: Path, script: str) -> set[str]:
    """Every ``--flag`` on a README line that invokes ``script``, as the
    parameter name it maps to."""
    flags: set[str] = set()
    for line in readme.read_text(encoding="utf-8").splitlines():
        if script in line:
            # Only the script's own flags: ``uv run --with modal modal run
            # <script> --steps 80`` names ``--with`` before the script.
            flags.update(_FLAG.findall(line.split(script, 1)[1]))
    return {_param(f) for f in flags}


def readme_defaults(readme: Path) -> dict[str, str]:
    """``{param: default}`` from the knobs table rows ``| `--flag` | default |``."""
    text = readme.read_text(encoding="utf-8")
    return {_param(flag): default.strip() for flag, default in _ROW.findall(text)}


def assert_readme_matches_entrypoints(readme: Path, entrypoints: Mapping[str, Any]) -> None:
    """``entrypoints`` maps a script path as the README writes it to its
    ``local_entrypoint``. Every flag on a command line for that script is
    one of its parameters; every knobs-table flag belongs to one of the
    scripts; every documented default (a number, ``off``, ``required``, or
    a literal) is what the parameter's default resolves to."""
    params = {script: inspect.signature(main).parameters for script, main in entrypoints.items()}
    for script, names in params.items():
        missing = sorted(readme_flags(readme, script) - set(names))
        assert not missing, f"{readme.name} names flags {script} does not have: {missing}"
    for name, documented in readme_defaults(readme).items():
        owners = [p for p in params.values() if name in p]
        assert owners, f"{readme.name} documents --{name.replace('_', '-')}, no script has it"
        documented = documented.strip("`")
        for owner in owners:
            actual = owner[name].default
            if documented == "required":
                assert actual is inspect.Parameter.empty, f"{name}: README says required"
            elif documented == "":
                assert actual in ("", None, inspect.Parameter.empty), f"{name}: {actual!r}"
            elif documented == "off":
                assert actual is False, f"{name}: README says off, code says {actual!r}"
            else:
                try:
                    assert float(documented) == float(actual), (name, documented, actual)
                except (TypeError, ValueError):
                    assert str(actual) == documented, (
                        f"{name}: README {documented!r}, code {actual!r}"
                    )
