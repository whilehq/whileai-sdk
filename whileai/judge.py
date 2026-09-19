"""``Judge``: an LLM grader as an object. Configure once, apply to rows.

    judge = wai.Judge(rubric=RUBRIC, model=wai.Anthropic("claude-haiku-4-5"))
    scored = data.grade(judge)      # reward 0/1 and a reason on every row
    verdict = judge(row)            # one row, the judge contract

A ``Judge`` honors the judge contract every grading call reads
(``{"reward": 0 or 1, "reason": str}``), so it drops in wherever a judge
callable does: ``data.grade``, ``run_judge``, ``evaluate``, ``judge_trust``,
a gated ``push``. The rubric says what doing the job means; without one
the judge grades the conduct floor only (nothing invented, nothing
skipped) and its name says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .config import current, spec_of


class Judge:
    """An LLM judge: a model, a rubric, and the context the agent was under.

    * ``rubric``: what doing the job means, as text. A ``Rubric`` object
      (``wai.simulations.Rubric``) is scored item by item instead.
    * ``model``: a backend object or spec string. Default: the judge from
      ``wai.configure``, else ``WHILEAI_JUDGE``, else the hosted Phi-4.
    * ``api_key``: for that model; the backend object's key, else the
      provider's environment variable.
    * ``policy`` and ``tools``: the agent's system prompt and tool schemas,
      so the judge sees the rules the agent was under. A ``SimulationData``
      supplies its own; set these when grading a bare row list.
    * ``use_privileged``: show the judge each row's ``privileged`` block
      (principle, reference, hidden state) the agent never saw.
    * ``name``: recorded on every graded row; defaults to the model name.

    Reference: LLM-as-a-judge, Lambert 2025, chapter Reward Modeling;
    Zheng et al. 2023, arXiv:2306.05685 (position and length bias, why
    ``judge_trust`` should follow).
    """

    kind = "llm"

    def __init__(
        self,
        rubric: Any = None,
        *,
        model: Any = None,
        api_key: str | None = None,
        policy: str = "",
        tools: Sequence[dict] | None = None,
        use_privileged: bool = False,
        name: str | None = None,
    ):
        self.rubric = rubric
        # a misspelled model string is refused here, not when the judge runs
        spec_of(model, kwarg="model")
        self._model = model
        self.api_key = api_key or getattr(model, "api_key", None)
        self.policy = policy
        self.tools = list(tools or [])
        self.use_privileged = use_privileged
        self._name = name
        self._item_judge: Any = None

    @property
    def spec(self) -> str | None:
        """The spec string this judge calls, resolved now."""
        explicit = spec_of(self._model, kwarg="model")
        if isinstance(explicit, str):
            return explicit
        return current().judge

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        spec = self.spec
        model = spec.split(":", 1)[1].split("@", 1)[0] if spec else "hosted"
        return f"judge:{model}" + ("" if self.rubric else ":conduct-floor")

    @property
    def __name__(self) -> str:  # what run_judge records as the judge's name
        return self.name

    def _prompt(self) -> str | None:
        if self.rubric is None or not isinstance(self.rubric, str):
            return None
        from .simulations.score.grade_llm import rubric_prompt

        return rubric_prompt(self.rubric)

    def __call__(self, row: dict) -> dict[str, Any]:
        """Grade one row under the judge contract."""
        from .simulations.score.rubric import Rubric

        if isinstance(self.rubric, Rubric):
            if self._item_judge is None:
                from .simulations.score.rubric import rubric_judge

                self._item_judge = rubric_judge(
                    self.rubric,
                    spec=self.spec,
                    api_key=self.api_key,
                    policy=self.policy,
                    tools=self.tools,
                )
            return self._item_judge(row)
        from .simulations.score.grade_llm import grade_one

        return grade_one(
            row,
            policy=self.policy,
            tools=self.tools,
            backend_spec=self.spec,
            api_key=self.api_key,
            prompt=self._prompt(),
            use_privileged=self.use_privileged,
        )

    def __repr__(self) -> str:
        rubric = "rubric" if self.rubric is not None else "conduct floor"
        return f"Judge({rubric}, model={self.spec or 'hosted'!r})"


__all__ = ["Judge"]
