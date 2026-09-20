"""wai.OPD, wai.OPSD, wai.Async and wai.prime_rl_config: method objects with cited
defaults, refused on a bad value, written into the TOML prime-rl reads.

Nothing here touches a model or a network. The TOML is parsed back with
tomllib where the interpreter has it (3.11+)."""

from __future__ import annotations

import sys

import pytest

import whileai as wai
from whileai.simulations import defaults
from whileai.simulations.training import train

TEACHER = wai.Endpoint(url="https://teacher.example/v1", model="Qwen/Qwen3-32B")


def _parse(text: str) -> dict:
    if sys.version_info < (3, 11):
        pytest.skip("tomllib is 3.11+")
    import tomllib

    return tomllib.loads(text)


# --- the objects ---------------------------------------------------------


def test_front_door_resolves_the_methods_and_stays_under_the_cap():
    assert wai.OPD is wai.methods.OPD
    assert wai.OPSD is wai.methods.OPSD
    assert wai.Async is wai.methods.Async
    assert wai.prime_rl_config is wai.methods.prime_rl_config
    assert "methods" in wai.__all__
    assert len(wai.__all__) <= 31  # rule 1 of docs/reference/style.md; 31 since `rows` (#613)
    assert wai.Backend is not None  # left the front door, still importable


def test_opd_defaults_are_the_named_constants():
    m = wai.OPD(TEACHER)
    assert m.divergence == defaults.OPD_DIVERGENCE == "reverse_kl"
    assert m.top_k == defaults.OPD_TOP_K
    assert m.samples == defaults.OPD_SAMPLES
    assert m.temperature == defaults.OPD_TEMPERATURE
    assert m.max_tokens == defaults.OPD_MAX_TOKENS
    assert m.teacher_ref == ("Qwen/Qwen3-32B", "https://teacher.example/v1")
    assert m.default_learning_rate(lora=True) == defaults.OPD_LEARNING_RATE_LORA
    assert m.default_learning_rate(lora=False) == defaults.OPD_LEARNING_RATE_FULL
    assert "Qwen/Qwen3-32B" in str(m) and "reverse_kl" in str(m)


def test_opd_teacher_accepts_a_spec_string_and_refuses_a_chat_api():
    assert wai.OPD("vllm:Qwen/Qwen3-32B@http://localhost:8000/v1").teacher_ref == (
        "Qwen/Qwen3-32B",
        "http://localhost:8000/v1",
    )
    with pytest.raises(ValueError, match="prompt log-probabilities"):
        wai.OPD(wai.OpenAI("gpt-4.1-mini"))
    with pytest.raises(ValueError, match="Endpoint"):
        wai.OPD("Qwen/Qwen3-32B")
    with pytest.raises(TypeError):
        wai.OPD(42)  # type: ignore[arg-type]


def test_opd_refuses_bad_knobs_and_names_the_constant():
    with pytest.raises(ValueError, match="divergence must be one of"):
        wai.OPD(TEACHER, divergence="kl")
    with pytest.raises(ValueError, match="OPD_TOP_K"):
        wai.OPD(TEACHER, top_k=0)
    with pytest.raises(ValueError, match="temperature"):
        wai.OPD(TEACHER, temperature=0)


def test_opsd_defaults_and_anchor_forms():
    m = wai.OPSD()
    assert m.privileged == defaults.OPSD_PRIVILEGED == "demonstration"
    assert m.divergence == defaults.OPSD_DIVERGENCE
    assert m.anchor_parts() == ("ema", defaults.OPSD_ANCHOR_ALPHA)
    assert m.samples == defaults.OPSD_SAMPLES == 1
    assert m.template == defaults.OPSD_TEMPLATE and "{demonstration}" in m.template
    assert m.default_learning_rate(lora=True) == defaults.OPSD_LEARNING_RATE
    assert wai.OPSD(anchor="initial").anchor_parts() == ("initial", None)
    assert wai.OPSD(anchor="live").anchor_parts() == ("live", None)
    assert wai.OPSD(anchor="ema:0.05").anchor_parts() == ("ema", 0.05)
    with pytest.raises(ValueError, match="OPSD_ANCHOR_ALPHA"):
        wai.OPSD(anchor="ema:1.5")
    with pytest.raises(ValueError, match="takes no rate"):
        wai.OPSD(anchor="initial:0.1")
    assert wai.OPSD(privileged="answer").privileged == "answer"  # any task field
    with pytest.raises(ValueError, match="privileged must be one of"):
        wai.OPSD(privileged="not a field!")
    with pytest.raises(ValueError, match="demonstration"):
        wai.OPSD(template="no slot here")


def test_async_defaults_and_refusals():
    a = wai.Async()
    assert a.method == "grpo" and a.inner_name == "grpo"
    assert a.off_policy_steps == defaults.ASYNC_OFF_POLICY_STEPS == 8
    assert a.correction == defaults.ASYNC_CORRECTION == "ipo"
    assert a.eps == defaults.ASYNC_IPO_EPS
    assert a.ratio == defaults.ASYNC_ICEPOP_RATIO
    assert wai.Async(wai.OPD(TEACHER)).inner_name == "opd"
    with pytest.raises(TypeError, match="not another Async"):
        wai.Async(wai.Async())
    with pytest.raises(ValueError, match="method must be one of"):
        wai.Async("ppo")
    with pytest.raises(ValueError, match="off_policy_steps"):
        wai.Async(off_policy_steps=-1)
    with pytest.raises(ValueError, match="bracket 1"):
        wai.Async(correction="icepop", ratio=(2.0, 5.0))
    assert "off_policy_steps=8" in str(a) and "ipo" in str(a)


def test_every_method_default_is_cited_in_defaults():
    """The constants the objects read carry a paper or a measurement, not a bare number."""
    import inspect

    src = inspect.getsource(defaults)
    for name in (
        "OPD_DIVERGENCE",
        "OPD_TOP_K",
        "OPD_SAMPLES",
        "OPD_MAX_TOKENS",
        "OPSD_PRIVILEGED",
        "OPSD_ANCHOR",
        "ASYNC_OFF_POLICY_STEPS",
        "ASYNC_CORRECTION",
        "PRIME_RL_GPUS",
    ):
        assert f"# {name} = " in src or f"/ {name} = " in src, name
    for arxiv in (
        "2306.13649",
        "2604.13016",
        "2601.19897",
        "2601.20802",
        "2510.13786",
        "2410.18252",
    ):
        assert arxiv in src


# --- the writer ----------------------------------------------------------


def test_prime_rl_config_grpo_writes_the_shape_prime_rl_reads(tmp_path):
    cfg = wai.prime_rl_config(
        "refunds-v1", "grpo", model="Qwen/Qwen3-4B", out=tmp_path / "grpo.toml"
    )
    assert cfg.path == str(tmp_path / "grpo.toml")
    assert (tmp_path / "grpo.toml").read_text(encoding="utf-8") == cfg.text
    d = _parse(cfg.text)
    assert d["max_steps"] == defaults.PRIME_RL_STEPS
    assert d["seq_len"] == defaults.PRIME_RL_SEQ_LEN
    assert d["deployment"] == {"gpus_per_node": 2, "num_infer_gpus": 1, "num_train_gpus": 1}
    assert d["model"]["name"] == "Qwen/Qwen3-4B"
    assert d["trainer"]["model"]["lora"] == {
        "rank": defaults.TRAINING_LORA_RANK,
        "alpha": defaults.TRAINING_LORA_ALPHA,
    }
    assert d["trainer"]["optim"]["lr"] == defaults.PRIME_RL_LEARNING_RATE_LORA
    orch = d["orchestrator"]
    assert orch["algo"] == {"type": "grpo"}
    assert orch["batch_size"] == defaults.PRIME_RL_BATCH
    assert orch["group_size"] == defaults.RL_ROLLOUTS_PER_PROMPT
    assert orch["max_off_policy_steps"] == defaults.ASYNC_OFF_POLICY_STEPS
    src = orch["train"]["source"][0]
    assert src["name"] == "refunds-v1"
    assert src["env"]["taskset"] == {"id": "refunds-v1"}  # which rows: the taskset's own field
    assert src["env"]["agent"] == {"harness": {"id": "null"}, "runtime": {"type": "subprocess"}}
    assert orch["eval"]["source"][0]["env"]["taskset"] == {"id": "refunds-v1"}
    assert orch["eval"]["source"][0]["name"] == "refunds-v1-eval"
    assert orch["eval"]["num_examples"] == defaults.PRIME_RL_EVAL_EXAMPLES
    assert "prime-rl default" in cfg.text  # the staleness bound is named even when not asked for
    assert cfg.command == f"uv run rl @ {cfg.path}"
    text = str(cfg)
    assert (
        "method grpo" in text and "2 GPUs (1 inference, 1 trainer)" in text and cfg.command in text
    )


def test_prime_rl_config_opd_writes_the_teacher_and_says_what_it_ignores():
    cfg = wai.prime_rl_config("math-v1", wai.OPD(TEACHER, divergence="jsd"), model="Qwen/Qwen3-4B")
    assert cfg.path is None
    d = _parse(cfg.text)
    assert d["orchestrator"]["algo"] == {
        "type": "opd",
        "teacher": {"name": "Qwen/Qwen3-32B", "base_url": "https://teacher.example/v1"},
    }
    assert d["orchestrator"]["group_size"] == defaults.OPD_SAMPLES
    assert (
        d["orchestrator"]["train"]["sampling"]["max_completion_tokens"] == defaults.OPD_MAX_TOKENS
    )
    assert d["trainer"]["optim"]["lr"] == defaults.OPD_LEARNING_RATE_LORA
    assert "teacher" in cfg.honored
    assert any(k.startswith("divergence=jsd") for k in cfg.ignored)
    assert any(k.startswith("top_k=") for k in cfg.ignored)
    assert cfg.method == "opd"


def test_prime_rl_config_opsd_writes_demo_key_and_warns_about_thinking_models():
    cfg = wai.prime_rl_config(
        "sci-qa", wai.OPSD(privileged="reference"), model="Qwen/Qwen3-8B", lora=False
    )
    d = _parse(cfg.text)
    algo = d["orchestrator"]["algo"]
    assert algo["type"] == "opsd" and algo["demo_key"] == "reference"
    assert algo["template"] == defaults.OPSD_TEMPLATE
    assert d["orchestrator"]["group_size"] == 1
    assert d["trainer"]["optim"]["lr"] == defaults.OPSD_LEARNING_RATE
    assert "lora" not in d["trainer"].get("model", {})
    assert any(
        k.startswith("anchor=") for k in cfg.ignored
    )  # prime-rl scores against the live policy
    assert any("2607.05184" in w for w in cfg.warnings)
    assert "ignores:" in str(cfg) and "warning:" in str(cfg)


def test_prime_rl_config_async_writes_the_bound_and_the_loss():
    cfg = wai.prime_rl_config(
        "refunds-v1",
        wai.Async("grpo", off_policy_steps=2, correction="icepop", ratio=(0.5, 5.0)),
        model="Qwen/Qwen3-4B",
        gpus=4,
    )
    d = _parse(cfg.text)
    assert d["orchestrator"]["max_off_policy_steps"] == 2
    assert d["trainer"]["loss"] == {"type": "icepop", "ratio_low": 0.5, "ratio_high": 5.0}
    assert d["deployment"] == {"gpus_per_node": 4, "num_infer_gpus": 2, "num_train_gpus": 2}
    assert cfg.method == "async grpo"
    ipo = wai.prime_rl_config("refunds-v1", wai.Async(wai.OPD(TEACHER)), model="Qwen/Qwen3-4B")
    assert _parse(ipo.text)["trainer"]["loss"] == {"type": "ipo", "eps": defaults.ASYNC_IPO_EPS}
    assert ipo.method == "async opd"


def test_prime_rl_config_refuses_what_prime_rl_cannot_honor():
    with pytest.raises(ValueError, match="'tis' is verl"):
        wai.prime_rl_config("refunds-v1", wai.Async(correction="tis"), model="Qwen/Qwen3-4B")
    with pytest.raises(ValueError, match="at least 2"):
        wai.prime_rl_config("refunds-v1", "grpo", model="Qwen/Qwen3-4B", gpus=1)
    with pytest.raises(ValueError, match="model"):
        wai.prime_rl_config("refunds-v1", "grpo", model="")
    with pytest.raises(ValueError, match="neither an installed taskset id"):
        wai.prime_rl_config("not a taskset id!", "grpo", model="Qwen/Qwen3-4B")
    with pytest.raises(TypeError, match="method must be"):
        wai.prime_rl_config("refunds-v1", object(), model="Qwen/Qwen3-4B")  # type: ignore[arg-type]


def test_prime_rl_config_overrides_land_verbatim_and_are_reported():
    cfg = wai.prime_rl_config(
        "refunds-v1",
        "grpo",
        model="Qwen/Qwen3-4B",
        steps=20,
        batch=8,
        **{
            "trainer.optim.lr": 2e-5,
            "seq_len": 4096,
            "orchestrator.train.sampling.top_p": 0.9,
            "source.env.agent.max_turns": 8,
            "train_source.env.taskset.dataset_split": "train",
            "eval_source.env.taskset.dataset_split": "test",
        },
    )
    d = _parse(cfg.text)
    assert d["orchestrator"]["train"]["source"][0]["env"]["agent"]["max_turns"] == 8
    assert d["orchestrator"]["eval"]["source"][0]["env"]["agent"]["max_turns"] == 8
    assert d["orchestrator"]["train"]["source"][0]["env"]["taskset"]["dataset_split"] == "train"
    assert d["orchestrator"]["eval"]["source"][0]["env"]["taskset"]["dataset_split"] == "test"
    assert d["max_steps"] == 20 and d["orchestrator"]["batch_size"] == 8
    assert d["trainer"]["optim"]["lr"] == 2e-5
    assert d["seq_len"] == 4096
    assert d["orchestrator"]["train"]["sampling"]["top_p"] == 0.9
    assert d["orchestrator"]["eval"]["interval"] == 5
    assert cfg.honored["trainer.optim.lr"] == "override, written as given"


def test_prime_rl_config_reads_an_exported_environment_directory(tmp_path):
    env = tmp_path / "refunds_env"
    env.mkdir()
    (env / "pyproject.toml").write_text(
        '[project]\nname = "refunds-env"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    cfg = wai.prime_rl_config(env, "grpo", model="Qwen/Qwen3-4B")
    assert cfg.taskset == "refunds-env"
    assert any("load_environment shape" in w for w in cfg.warnings)
    report = {"path": str(env)}  # the dict export_environment returns
    assert wai.prime_rl_config(report, "grpo", model="Qwen/Qwen3-4B").taskset == "refunds-env"


def test_hosted_train_points_a_method_object_at_prime_rl_config():
    with pytest.raises(TypeError, match="prime_rl_config"):
        train("ds_123", method=wai.OPD(TEACHER))
    with pytest.raises(TypeError, match="Async"):
        train("ds_123", method=wai.Async())
