"""wai.OPD, wai.OPSD, wai.Async and wai.prime_rl_config: method objects with cited
defaults, refused on a bad value, written into the TOML prime-rl reads. The
single-rollout methods (wai.FlashReinforce, wai.SAO, wai.BPCO) are refused there
with the reason, and the tests say so.

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
    assert wai.FlashReinforce is wai.methods.FlashReinforce
    assert wai.SAO is wai.methods.SAO
    assert wai.BPCO is wai.methods.BPCO
    assert wai.prime_rl_config is wai.methods.prime_rl_config
    for cls in (wai.FlashReinforce, wai.SAO, wai.BPCO):
        assert cls.samples == 1  # one rollout per prompt: the shape a production trace has
        assert isinstance(cls.name, str) and cls.name
    assert "methods" in wai.__all__
    assert "OPD" in wai.__all__ and "prime_rl_config" in wai.__all__  # method-routing task, #564
    assert len(wai.__all__) <= 33  # rule 1 of docs/reference/style.md; 33 since OPD/prime_rl_config
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
        "FLASH_REINFORCE_TRUST",
        "FLASH_REINFORCE_OFF_POLICY_STEPS",
        "SAO_RATIO",
        "SAO_GAE_ALPHA",
        "SAO_CRITIC_STEPS",
        "BPCO_CLIP",
        "BPCO_GAE_ALPHA",
        "BPCO_REWARD_RANGE",
        "BPCO_CRITIC_WARMUP",
    ):
        assert f"# {name} = " in src or f"/ {name} = " in src, name
    for arxiv in (
        "2306.13649",
        "2604.13016",
        "2601.19897",
        "2601.20802",
        "2510.13786",
        "2410.18252",
        "2607.07508",  # SAO
        "2608.23566",  # BPCO
    ):
        assert arxiv in src


def test_update_prints_what_it_admitted_and_why():
    """The single-rollout result type prints itself: admitted count, stats, notes."""
    up = wai.methods.Update(
        method="flash_reinforce",
        coefficients=[[0.5, 0.5], [0.0]],
        advantages=[[0.5, 0.5], [-0.5]],
        admitted=[True, False],
        stats={"admitted_share": 0.5},
        notes=["trajectory 1 dropped: mean KL to the sampler over trust"],
    )
    assert up.n == 2 and up.n_admitted == 1 and up.value_targets is None
    text = str(up)
    assert text.startswith("flash_reinforce update: 1 of 2 trajectories admitted")
    assert "admitted share: 0.5" in text and "trajectory 1 dropped" in text
    assert "<pre>" in up._repr_html_()


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


# --- the single-rollout methods on prime-rl -------------------------------
#
# What prime-rl main (2026-09-21) offers, read from the tree: orchestrator.algo
# is grpo, echo, max_rl, rae, hierarchical_grpo, opd, opsd, sft, debug
# (packages/prime-rl-configs/src/prime_rl/configs/algorithm.py); the reward
# baselines are the group mean (grpo, max_rl: zero over a group of one) or a
# per-agent EMA (rae, group_size 1 allowed); it hosts no value model; and
# trainer.loss is ipo, icepop or custom (configs/trainer.py), all per token,
# normalized by the global token count (trainer/rl/train.py rl_scale). So a
# single-rollout method is refused with the reason and the fix, and the one
# thing that runs one rollout per prompt there is "rae".


def test_prime_rl_config_rae_runs_one_rollout_per_prompt_and_names_its_baseline():
    cfg = wai.prime_rl_config(
        "refunds-v1",
        wai.Async("rae", correction="icepop"),
        model="Qwen/Qwen3-4B",
        **{"orchestrator.group_size": 1},
    )
    d = _parse(cfg.text)
    assert d["orchestrator"]["algo"] == {"type": "rae"}
    assert d["orchestrator"]["group_size"] == 1
    assert d["trainer"]["loss"]["type"] == "icepop"
    assert cfg.method == "async rae"
    assert any("EMA" in w and "group_size 1" in w for w in cfg.warnings)
    assert wai.Async("rae").inner_name == "rae"
    plain = wai.prime_rl_config("refunds-v1", "rae", model="Qwen/Qwen3-4B")
    assert _parse(plain.text)["orchestrator"]["algo"] == {"type": "rae"}
    assert any("not a group mean or a batch mean" in w for w in plain.warnings)


def test_prime_rl_config_refuses_flash_reinforce_and_names_rae():
    """No batch-mean baseline at group_size 1, no sequence trust region, no 1/T."""
    with pytest.raises(ValueError, match="batch-mean baseline is not offered") as e:
        wai.prime_rl_config("refunds-v1", wai.FlashReinforce(), model="Qwen/Qwen3-4B")
    text = str(e.value)
    assert "orchestrator/algo/grpo.py" in text  # where the group-mean advantage lives
    assert "no sequence trust region" in text and "1/T" in text
    assert "method.update(batch)" in text  # the fix that exists today
    assert "wai.Async('rae'" in text and "'orchestrator.group_size': 1" in text  # the nearest
    assert "different baseline" in text  # and it is named as a different method


def test_prime_rl_config_refuses_the_critic_methods_and_says_what_is_missing():
    with pytest.raises(ValueError, match="only ever hosts the trainable policy") as sao:
        wai.prime_rl_config("refunds-v1", wai.SAO(), model="Qwen/Qwen3-4B")
    text = str(sao.value)
    assert "icepop" in text and "ratio=(0.7, 6.0)" in text  # the half prime-rl has, named
    assert "different method" in text and "method.update(batch)" in text and "'values'" in text
    with pytest.raises(ValueError, match="no clip at all") as bpco:
        wai.prime_rl_config("refunds-v1", wai.BPCO(), model="Qwen/Qwen3-4B")
    text = str(bpco.value)
    assert "only ever hosts the trainable policy" in text
    assert "clip/mu" in text and "method.update(batch)" in text


def test_prime_rl_config_never_writes_a_file_for_a_refused_method(tmp_path):
    out = tmp_path / "sao.toml"
    with pytest.raises(ValueError):
        wai.prime_rl_config("refunds-v1", wai.SAO(), model="Qwen/Qwen3-4B", out=out)
    assert not out.exists()


def test_hosted_train_names_update_for_a_single_rollout_method():
    for method in (wai.FlashReinforce(), wai.SAO(), wai.BPCO()):
        with pytest.raises(TypeError, match=r"method\.update\(batch\)"):
            train("ds_123", method=method)


# --- teacher_beats_student: the check OPD's docstring names, made callable --------


def test_teacher_beats_student_reproduces_the_measured_opd_regression():
    # #issue 564: an OPD run lost 14.8 points because the teacher (Qwen3.5-9B,
    # 56.2) was never scored against the student's GRPO best (70.9) first; a
    # teacher no better than the student gives OPD nothing to pull toward, so
    # the run was doomed before it began.
    check = wai.methods.teacher_beats_student(0.562, 0.709)
    assert check["beats"] is False
    assert check["verdict"] == "does not beat"
    assert check["gap"] == pytest.approx(0.562 - 0.709)
    assert "0.562" in check["message"] and "0.709" in check["message"]
    assert "at or below the student" in check["message"]


def test_teacher_beats_student_clears_the_margin_and_names_opd():
    check = wai.methods.teacher_beats_student(0.85, 0.60)
    assert check["beats"] is True and check["verdict"] == "beats"
    assert check["margin"] == defaults.PROVE_EFFECT
    assert "wai.prime_rl_config(env, wai.OPD(teacher)" in check["message"]


def test_teacher_beats_student_reads_pass_at_and_distrusts_overlapping_intervals():
    ahead_but_overlapping = {"pass_at_1": 0.60, "ci95": (0.50, 0.70)}
    student = {"pass_at_1": 0.58, "ci95": (0.50, 0.66)}
    check = wai.methods.teacher_beats_student(ahead_but_overlapping, student)
    assert check["beats"] is False and check["verdict"] == "unclear"
    assert check["ci_overlap"] is True
    assert "confidence intervals overlap" in check["message"]
    with pytest.raises(TypeError, match="needs a pass rate"):
        wai.methods.teacher_beats_student(object(), 0.5)


# --- prime_rl_config: the OPD path warns loudly, on request or by default --------


def _opd(model: str = "Qwen3.5-9B") -> wai.OPD:
    return wai.OPD(teacher=wai.Endpoint(url="https://teacher.example/v1", model=model))


def test_prime_rl_config_warns_loudly_when_the_teacher_check_was_never_run():
    cfg = wai.prime_rl_config("refunds-v1", _opd(), model="Qwen/Qwen3-4B")
    (line,) = [w for w in cfg.warnings if "teacher_check" in w]
    assert "has not been scored" in line
    assert "wai.methods.teacher_beats_student" in line and "teacher_check=" in line


def test_prime_rl_config_relays_a_failing_teacher_check_and_names_the_fix():
    check = wai.methods.teacher_beats_student(0.562, 0.709)
    cfg = wai.prime_rl_config("refunds-v1", _opd(), model="Qwen/Qwen3-4B", teacher_check=check)
    assert not any("has not been scored" in w for w in cfg.warnings)
    (line,) = [w for w in cfg.warnings if "not reachable yet" in w]
    assert "at or below the student" in line


def test_prime_rl_config_records_a_passing_teacher_check_instead_of_warning():
    check = wai.methods.teacher_beats_student(0.85, 0.60)
    cfg = wai.prime_rl_config("refunds-v1", _opd(), model="Qwen/Qwen3-4B", teacher_check=check)
    assert not any("teacher_check" in w or "not reachable" in w for w in cfg.warnings)
    assert "checked:" in cfg.text and "0.85" in cfg.text


def test_prime_rl_config_warns_on_a_vocab_mismatch_and_names_the_fix():
    cfg = wai.prime_rl_config(
        "refunds-v1",
        _opd(),
        model="Qwen/Qwen3-4B",
        teacher_check=wai.methods.teacher_beats_student(0.85, 0.60),
        teacher_vocab_size=248320,
        student_vocab_size=151936,
    )
    (line,) = [w for w in cfg.warnings if "vocab" in w]
    assert "248320" in line and "151936" in line
    assert "does not degrade gracefully" in line and "silently drops the signal" in line
    assert "same tokenizer" in line or "same model family" in line


def test_prime_rl_config_warns_when_vocab_was_never_checked_but_not_on_a_match():
    check = wai.methods.teacher_beats_student(0.85, 0.60)
    unchecked = wai.prime_rl_config(
        "refunds-v1", _opd(), model="Qwen/Qwen3-4B", teacher_check=check
    )
    assert any("vocab size was not checked" in w for w in unchecked.warnings)

    matching = wai.prime_rl_config(
        "refunds-v1",
        _opd(),
        model="Qwen/Qwen3-4B",
        teacher_check=check,
        teacher_vocab_size=151936,
        student_vocab_size=151936,
    )
    assert not any("vocab" in w for w in matching.warnings)


def test_opsd_prime_rl_config_does_not_carry_the_opd_teacher_warnings():
    # OPSD's teacher is the student itself (with a hint); the tokenizer and
    # ceiling checks are OPD-only, and OPSD keeps its own warning unchanged.
    cfg = wai.prime_rl_config("refunds-v1", wai.OPSD(), model="Qwen/Qwen3-8B")
    assert not any("teacher_check" in w or "vocab" in w for w in cfg.warnings)
    assert any("costs points on thinking models" in w for w in cfg.warnings)
