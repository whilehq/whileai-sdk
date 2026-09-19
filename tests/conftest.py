"""Unit tests never hit the hosted GPU."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _offline_hosted_simulator(monkeypatch, tmp_path):
    # never read the developer's own ~/.whileai/credentials.json: a saved
    # account key would flip the hosted defaults to the account route
    monkeypatch.setenv("WHILEAI_HOME", str(tmp_path / "whileai-home"))

    def blocked(*_args, **_kwargs):
        raise OSError("hosted simulator disabled in unit tests")

    def embed_blocked(self, texts):
        raise OSError("hosted embedder disabled in unit tests")

    monkeypatch.setattr("whileai.simulations.generate.generator.complete", blocked)
    monkeypatch.setattr("whileai.simulations.generate.agents.complete", blocked)
    monkeypatch.setattr("whileai.simulations.score.llm_judge.complete", blocked)
    monkeypatch.setattr(
        importlib.import_module("whileai.simulations.score.grade_llm"), "complete", blocked
    )
    monkeypatch.setattr(
        "whileai.simulations.generate.embeddings.ModalEmbedder.embed", embed_blocked
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("WHILEAI_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
