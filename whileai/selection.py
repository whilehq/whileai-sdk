"""``Selection``: the rows worth training on, as an object that prints its report.

    rows = scored.select(mode="rl")     # or wai.select(scored, mode="rl")
    print(rows)                          # what each gate dropped and why
    rows.export("train.jsonl")           # trainer-ready JSONL
    rows.push("my-agent-rl-v1")          # to the platform, gated

A ``Selection`` is a list of rows (it feeds anything a row list feeds) that
also carries ``report`` (the dict ``optimize`` computes), ``mode`` and the
system prompt and tools the rows were generated under, so export needs no
re-typing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .simulations.data import RowList


class Selection(RowList):
    """Rows kept by ``optimize``, with the report that says why."""

    def __init__(
        self,
        rows: Sequence[dict],
        *,
        report: dict[str, Any] | None = None,
        mode: str = "rl",
        system_prompt: str = "",
        tools: Sequence[dict] | None = None,
    ):
        super().__init__(rows)
        self.report: dict[str, Any] = report if report is not None else {}
        self.mode = mode
        self.system_prompt = system_prompt
        self.tools = list(tools or [])

    # -- printing ---------------------------------------------------------

    def __str__(self) -> str:
        r = self.report
        n_in = r.get("n", len(self))
        lines = [f"{self.mode} selection: kept {len(self)} of {n_in} rows"]
        if self.mode == "rl":
            lo, hi = r.get("band", (None, None))
            if lo is not None:
                lines.append(
                    f"  band {lo:.0%}..{hi:.0%} pass rate: "
                    f"{r.get('band_groups_dropped', 0)} asks dropped "
                    f"({(r.get('band_dropped') or {}).get('too_easy', 0)} too easy, "
                    f"{(r.get('band_dropped') or {}).get('too_hard', 0)} too hard)"
                )
            lines.append(
                f"  unanimous groups dropped: {r.get('unanimous_groups_dropped', 0)}; "
                f"duplicates dropped: {(r.get('duplicates') or {}).get('n_dropped', 0)}; "
                f"truncated {r.get('truncated_policy', 'drop')}: {r.get('truncated_dropped', 0)}"
            )
            lines.append(f"  privileged leaks dropped: {r.get('privileged_leaks_dropped', 0)}")
            lines.append(f"  groups kept: {r.get('groups_selected', 0)}")
            scan = r.get("hack_scan") or {}
            if scan.get("regime"):
                lines.append(f"  hack scan: {str(scan['regime']).replace('_', ' ')}")
        else:
            lines.append(
                f"  eligible {r.get('n_eligible', 0)} (reward >= {r.get('min_reward', 1.0)}), "
                f"junk {r.get('n_junk', 0)}, not passing {r.get('n_not_pass', 0)}"
            )
            lines.append(f"  privileged leaks dropped: {r.get('privileged_leaks_dropped', 0)}")
            lines.append(
                f"  distinct behaviors: {r.get('unique_behaviors', 0)}, "
                f"covered: {r.get('behaviors_covered', 0)}"
            )
            if r.get("note"):
                lines.append(f"  {r['note']}")
        for w in r.get("hygiene_warnings") or []:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return "<pre>" + str(self).replace("<", "&lt;") + "</pre>"

    def __repr__(self) -> str:
        return f"Selection(mode={self.mode!r}, n={len(self)})"

    # -- what comes next --------------------------------------------------

    def export(
        self,
        output: str | None = None,
        *,
        format: str = "openai",
        unroll: bool = False,
        validate: bool = True,
    ) -> dict[str, Any]:
        """Write the rows trainer-ready: ``export_dataset`` with this
        selection's system prompt and tools already filled in. ``format``
        is ``"openai"`` (chat JSONL with a ``loss_mask`` per message) or
        ``"trl"`` (what ``SFTTrainer`` loads); ``unroll=True`` makes one
        sample per agent turn."""
        from .simulations.export import export_dataset

        return export_dataset(
            list(self),
            output=output,
            system_prompt=self.system_prompt,
            tools=self.tools or None,
            format=format,
            unroll=unroll,
            validate=validate,
        )

    def push(self, name: str, **kwargs: Any) -> dict:
        """Upload to the platform as a dataset: ``push_rows`` with this
        selection's mode. ``gate=True`` (the default) refuses RL rows that
        carry no gradient."""
        from .simulations.ingest.platform import push_rows

        kwargs.setdefault("mode", self.mode)
        return push_rows(list(self), name, **kwargs)


def select(
    source: Any,
    *,
    mode: str | None = None,
    target: int = 1000,
    band: tuple[float, float] | None = None,
    endorsed: Sequence[str] = (),
    truncated: str = "drop",
    output: str | None = None,
) -> Selection:
    """Keep the rows worth training on, for SFT or RL: ``optimize`` as an object.

    * ``source``: a ``SimulationData``, a ``ScoredData``, a row list or a JSONL path.
    * ``mode``: ``"rl"`` or ``"sft"``; defaults to the run's own mode, else ``"rl"``.
    * ``target``: about how many rows to keep.
    * ``band``: the RL difficulty band as a pass-rate range, ``(0.2, 0.8)`` by
      default (Lambert 2025, chapter Reasoning; Yu et al. 2025 (DAPO),
      arXiv:2503.14476).
    * ``endorsed``: feature names the reward should track, so the hack scan
      can call a shortcut a hack.
    * ``truncated``: ``"drop"``, ``"keep"`` or ``"penalize"`` for rollouts cut
      at the token cap (DAPO's overlong handling).
    * ``output``: write the kept rows there as JSONL.

    In both modes a row whose reply quotes its own privileged context (the
    reference answer, the principle, the hidden world state) is dropped
    before any other gate and counted in the printed report, so ``export``
    never refuses a row this kept.
    """
    from .simulations.score.optimize import DEFAULT_BAND, optimize

    rows_in = getattr(source, "rows", None) if not hasattr(source, "trajectories") else None
    picked, report = optimize(
        rows_in if rows_in is not None else source,
        mode=mode,
        target=target,
        band=band or DEFAULT_BAND,
        endorsed=endorsed,
        truncated=truncated,
        output=output,
    )
    profile = getattr(source, "profile", None)
    return Selection(
        picked,
        report=report,
        mode=str(report.get("mode") or mode or "rl"),
        system_prompt=str(getattr(profile, "policy", "") or ""),
        tools=list(getattr(profile, "tools", None) or []),
    )


__all__ = ["Selection", "select"]
