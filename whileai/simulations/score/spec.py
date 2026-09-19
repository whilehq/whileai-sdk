"""Model spec as a versioned object (Lambert 2025, chapter Model Character and
Products).

A model spec (or constitution) is a living document: a set of named traits,
each with a principle the model should follow. Lambert 2025 makes the point
that it is versioned and measured — you track adherence to *this* version
across model releases, and you notice when an edit to the spec, not the model,
moved a number. Character training already grades replies against a
constitution; what was missing is the constitution as an object with an
identity.

``Spec`` wraps the ``{source, traits: [{id, name, authority, principle,
examples}]}`` shape the character example already writes. Its ``version`` is
a content hash, so any edit to a principle changes it. ``stamp_spec(rows,
spec)`` records which spec version a run targeted; ``spec.behaviors()`` are
the trait ids, ready to hand to ``delta_report(must_not_regress=...)``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Trait:
    """One named expectation. ``authority`` is the strength (``must`` /
    ``should`` / ``may``, following the model-spec convention)."""

    id: str
    name: str = ""
    principle: str = ""
    authority: str = "should"
    examples: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Spec:
    """A versioned model spec. ``version`` is derived from the content when
    left empty, so it is stable across processes and changes on any edit."""

    id: str
    traits: tuple[Trait, ...] = ()
    source: dict = field(default_factory=dict)
    version: str = ""

    def __post_init__(self) -> None:
        if not self.version:
            object.__setattr__(self, "version", self._content_hash())

    def _content_hash(self) -> str:
        payload = [
            {"id": t.id, "name": t.name, "principle": t.principle, "authority": t.authority}
            for t in self.traits
        ]
        blob = json.dumps({"id": self.id, "traits": payload}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def behaviors(self) -> list[str]:
        """Trait ids, in order. Hand to ``delta_report(must_not_regress=...)``."""
        return [t.id for t in self.traits]

    def principle(self, trait_id: str) -> str | None:
        for t in self.traits:
            if t.id == trait_id:
                return t.principle
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "source": self.source,
            "traits": [
                {
                    "id": t.id,
                    "name": t.name,
                    "principle": t.principle,
                    "authority": t.authority,
                    "examples": list(t.examples),
                }
                for t in self.traits
            ],
        }


def load_spec(obj: Any, *, spec_id: str | None = None) -> Spec:
    """Build a ``Spec`` from a constitution dict, a list of traits, or a path
    to a JSON file with either shape. A trait may be a full dict or a bare
    principle string."""
    if isinstance(obj, (str, Path)):
        data = json.loads(Path(obj).read_text(encoding="utf-8"))
    else:
        data = obj
    source: dict = {}
    if isinstance(data, dict):
        raw_traits = data.get("traits") or data.get("principles") or []
        source = data.get("source") or {}
        sid = spec_id or str(data.get("id") or source.get("file") or "spec")
    elif isinstance(data, list):
        raw_traits = data
        sid = spec_id or "spec"
    else:
        raise TypeError("load_spec needs a dict, a list of traits, or a path")

    traits: list[Trait] = []
    for i, t in enumerate(raw_traits):
        if isinstance(t, str):
            traits.append(Trait(id=f"trait_{i}", principle=t))
        elif isinstance(t, dict):
            traits.append(
                Trait(
                    id=str(t.get("id") or t.get("name") or f"trait_{i}"),
                    name=str(t.get("name") or ""),
                    principle=str(t.get("principle") or t.get("body") or ""),
                    authority=str(t.get("authority") or "should"),
                    examples=tuple(t.get("examples") or ()),
                )
            )
    return Spec(
        id=sid,
        traits=tuple(traits),
        source=source,
        version=str((data or {}).get("version") or "") if isinstance(data, dict) else "",
    )


def spec_version(spec: Spec | dict | Any) -> str:
    """The content version of a Spec (or anything ``load_spec`` accepts)."""
    return spec.version if isinstance(spec, Spec) else load_spec(spec).version


def stamp_spec(rows: Sequence[dict], spec: Spec) -> list[dict]:
    """Return copies of ``rows`` tagged with the spec they were produced or
    graded against: ``spec_id`` and ``spec_version``. Provenance for the ch.
    17 retention question — did adherence hold from one spec version, or model
    version, to the next."""
    return [
        {**row, "spec_id": spec.id, "spec_version": spec.version} if isinstance(row, dict) else row
        for row in rows
    ]
