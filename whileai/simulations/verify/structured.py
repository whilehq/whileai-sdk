"""Structured-output verifiers: valid JSON, JSON Schema, a field's value.

Many agent rewards are "did it emit the right structured call" (Lambert
2025, chapter Tool Use).
``JSONSchema`` uses the ``jsonschema`` package when installed and falls back
to a minimal type/required check so it still runs without it.
"""

from __future__ import annotations

from typing import Any

from .base import Verifier, extract_json


class JSONValid(Verifier):
    """The candidate parses as JSON (optionally as a specific top-level type)."""

    def __init__(
        self, *, top_type: type | None = None, field: str | None = None, name: str | None = None
    ):
        super().__init__(field=field, name=name)
        self.top_type = top_type

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        obj = extract_json(candidate)
        if obj is None:
            return 0, "not valid JSON"
        if self.top_type is not None and not isinstance(obj, self.top_type):
            return 0, f"JSON is {type(obj).__name__}, want {self.top_type.__name__}"
        return 1, "valid JSON"


def _minimal_validate(obj: Any, schema: dict) -> str | None:
    """Tiny subset of JSON Schema: type, required, properties.type. Returns an
    error string or None. Used only when jsonschema is not installed."""
    t = schema.get("type")
    py: dict[str, Any] = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "null": type(None),
    }
    if t and t in py and not isinstance(obj, py[t]):
        return f"top-level type {type(obj).__name__}, want {t}"
    if t == "object" and isinstance(obj, dict):
        for req in schema.get("required", []) or []:
            if req not in obj:
                return f"missing required field {req!r}"
        for key, sub in (schema.get("properties") or {}).items():
            if (
                key in obj
                and "type" in sub
                and sub["type"] in py
                and not isinstance(obj[key], py[sub["type"]])
            ):
                return f"field {key!r} is {type(obj[key]).__name__}, want {sub['type']}"
    return None


class JSONSchema(Verifier):
    """The candidate is JSON that validates against ``schema``."""

    def __init__(self, schema: dict, *, field: str | None = None, name: str | None = None):
        super().__init__(field=field, name=name)
        self.schema = schema

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        obj = extract_json(candidate)
        if obj is None:
            return 0, "not valid JSON"
        try:
            import jsonschema

            jsonschema.validate(obj, self.schema)
            return 1, "schema valid"
        except ImportError:
            err = _minimal_validate(obj, self.schema)
            return (1 if err is None else 0), (err or "schema valid (minimal check)")
        except Exception as exc:  # jsonschema.ValidationError
            msg = getattr(exc, "message", str(exc))
            return 0, f"schema: {msg}"[:200]


class JSONField(Verifier):
    """Extract a dotted path from the candidate JSON and compare it to the
    reference (or to a fixed ``equals`` value)."""

    def __init__(
        self, path: str, *, equals: Any = None, field: str | None = None, name: str | None = None
    ):
        super().__init__(field=field, name=name)
        self.path, self.equals = path, equals

    def _dig(self, obj: Any) -> Any:
        for part in self.path.split("."):
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            elif isinstance(obj, list) and part.isdigit() and int(part) < len(obj):
                obj = obj[int(part)]
            else:
                return None
        return obj

    def check(self, candidate: str, reference: Any, row: dict) -> Any:
        obj = extract_json(candidate)
        if obj is None:
            return 0, "not valid JSON"
        got = self._dig(obj)
        want = self.equals if self.equals is not None else reference
        if want is None:
            return (
                1 if got is not None else 0
            ), f"{self.path} {'present' if got is not None else 'missing'}"
        ok = str(got).strip().lower() == str(want).strip().lower()
        return (1 if ok else 0), f"{self.path}={got!r}, want {want!r}"
