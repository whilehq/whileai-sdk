"""The GRPO example, offline: the reward, the prompt builders and the
checked-in prompt set against the numbers the README states, and the Modal
scripts' imports and flags against the README."""

from __future__ import annotations

import inspect
import json

import pytest
from example_helpers import (
    EXAMPLES,
    assert_readme_matches_entrypoints,
    load_modal_script,
    load_script,
)

import whileai.simulations as wai

GRPO = EXAMPLES / "04-train/grpo"
README = GRPO / "README.md"


def _modules():
    reward = load_script("reward", GRPO / "reward.py")
    prompts = load_script("grpo_example_prompts", GRPO / "prompts.py")
    return reward, prompts


def _seed(reward, text):
    return {"prompt": text, "case": reward.case_for(text), "scenario_id": "s-" + text[:8]}


CALL = '<tool_call>\n{"name": "lookup_order", "arguments": {"order_id": "ORD-4017"}}\n</tool_call>'
REFUND = '<tool_call>{"name": "create_refund", "arguments": {"order_id": "ORD-4017", "amount": 20}}</tool_call>'


# ---------------------------------------------------------------- reward


def test_case_reads_the_order_id_and_domain():
    r, _ = _modules()
    assert r.case_for("please check ORD-4017 for me") == {"order_id": "ORD-4017", "in_domain": True}
    assert r.case_for("look at ord_991 please")["order_id"] == "ORD-991"
    assert r.case_for("I need help with a refund") == {"order_id": None, "in_domain": True}
    assert r.case_for("How tall is Kilimanjaro?") == {"order_id": None, "in_domain": False}


def test_score_rewards_the_rule_and_the_format():
    """Every row of the README's reward table, plus the +0.2 format bonus."""
    r, _ = _modules()
    with_id = r.case_for("please check ORD-4017 for me")
    assert r.score(CALL, with_id) == 1.0
    assert r.score(REFUND, with_id) == pytest.approx(0.2)  # well formed, wrong move
    wrong_id = CALL.replace("ORD-4017", "ORD-9999")
    assert r.score(wrong_id, with_id) == pytest.approx(0.3)
    assert r.score("Sure, what is your order number?", with_id) == pytest.approx(0.3)
    assert r.score("<tool_call>{not json</tool_call>", with_id) == 0.0

    no_id = r.case_for("I need help with a refund")
    assert r.score("Of course. What is the order id?", no_id) == 1.0
    assert r.score(CALL, no_id) == pytest.approx(0.2)  # invented id, format bonus only
    assert r.score("Refunded.", no_id) == pytest.approx(0.4)

    off = r.case_for("How tall is Kilimanjaro?")
    assert r.score("I can only help with orders and refunds.", off) == 1.0
    assert r.score("x" * 401, off) == pytest.approx(0.5)  # long off-topic reply: 0.3 + bonus
    assert r.score(CALL, off) == pytest.approx(0.2)
    assert r.score("", off) == 0.0


def test_reward_rows_feed_pass_at_and_delta():
    r, _ = _modules()
    prompts = [
        {
            "prompt": "check ORD-100 please",
            "case": r.case_for("check ORD-100 please"),
            "scenario_id": "s1",
        },
        {"prompt": "refund help", "case": r.case_for("refund help"), "scenario_id": "s2"},
    ]
    before = r.reward_rows(prompts, [[REFUND.replace("4017", "100"), "hm"], ["Refunded.", "ok"]])
    after = r.reward_rows(
        prompts,
        [
            [CALL.replace("4017", "100"), CALL.replace("4017", "100")],
            ["What is the order id?", "Order id?"],
        ],
    )
    assert wai.pass_at(before).pass_at_1 == 0.0 and wai.pass_at(after).pass_at_1 == 1.0
    assert before[0]["markers"]["tool_rule"] == pytest.approx(0.2)
    rep = wai.delta_report(before, after, target="pass_at_1", must_not_regress=["well_formed"])
    assert rep["target_delta"] == pytest.approx(1.0)


def test_build_prompts_offline_and_split():
    r, _ = _modules()
    items = r.build_prompts(24, seed=1)
    assert 8 <= len(items) <= 24 and all(set(i) == {"prompt", "case", "scenario_id"} for i in items)
    assert any(i["case"]["order_id"] for i in items) and any(
        not i["case"]["order_id"] for i in items
    )
    train, held = r.split_holdout(items, 0.25)
    assert len(train) + len(held) == len(items)
    assert r.split_holdout(items, 0.25) == (train, held)  # deterministic


def test_same_seed_writes_the_same_prompt_and_holdout_files(tmp_path):
    """whilehq/whileai-sdk#450: two runs at ``--seed 0`` got 112 and 119
    prompts and two different holdouts, so their before/after numbers were
    not paired. The data step is the offline call ``train_modal.main``
    makes at its defaults (200 situations, 20% held out); one seed writes
    byte-identical prompt, train and holdout files twice over, and another
    seed writes different ones. The counts are the README's."""
    r, _ = _modules()

    def write(seed: int, tag: str) -> dict[str, bytes]:
        items = r.build_prompts(200, seed=seed)
        train, held = r.split_holdout(items, 0.2)
        out: dict[str, bytes] = {}
        for name, rows in (("prompts", items), ("train", train), ("holdout", held)):
            path = tmp_path / f"{name}-{tag}.jsonl"
            path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
            out[name] = path.read_bytes()
        out["counts"] = f"{len(items)} {len(train)} {len(held)}".encode()
        return out

    a = write(0, "a")
    b = write(0, "b")
    assert a == b
    assert a["counts"] == b"117 92 25"
    text = README.read_text(encoding="utf-8")
    assert (
        "builds the same 117\nprompts (91 name an order id), the same 92 train and the same 25 holdout"
        in text
    )
    c = write(1, "c")
    assert c["prompts"] != a["prompts"] and c["holdout"] != a["holdout"]


# --------------------------------------------------- model-written prompts


def test_writer_messages_carry_the_category_rule():
    r, p = _modules()
    with_id = p.writer_messages(_seed(r, "please check ORD-4017 for me"), 0, 6)
    assert "ORD-4017" in with_id[1]["content"] and "exactly 6 strings" in with_id[1]["content"]
    no_id = p.writer_messages(_seed(r, "I need help with a refund"), 1, 4)
    assert (
        "Never include an id" in no_id[1]["content"]
        and "different kind of customer" in no_id[1]["content"]
    )
    off = p.writer_messages(_seed(r, "How tall is Kilimanjaro?"))
    assert "not about an order" in off[1]["content"]


def test_parse_messages_reads_an_array_or_quoted_lines():
    _, p = _modules()
    assert p.parse_messages('Sure:\n["a message", "another one"]\n') == ["a message", "another one"]
    assert p.parse_messages('1. "first line"\n- "second line",\nnot a message') == [
        "first line",
        "second line",
    ]
    assert p.parse_messages("") == []


def test_keep_enforces_category_and_drops_near_duplicates():
    r, p = _modules()
    s_id = _seed(r, "please check ORD-4017 for me")
    s_no = _seed(r, "I need help with a refund")
    s_off = _seed(r, "How tall is Kilimanjaro?")
    cands = [
        ("Hi, can you look into order ORD-4017? It arrived broken.", s_id),
        ("Hi can you look into order ORD-4017, it arrived broken", s_id),  # near duplicate
        ("Can you look into ORD-9999 for me?", s_id),  # wrong id
        ("Where is my refund? I ordered a jacket last week and nothing came.", s_no),
        ("Refund please, order ORD-1234", s_no),  # an id where none is allowed
        ("Do you sell gift cards?", s_off),
        ("Is my refund on order coming?", s_off),  # in domain where off topic is required
        ("short", s_id),
    ]
    kept = p.keep(cands)
    assert [k["prompt"] for k in kept] == [
        "Hi, can you look into order ORD-4017? It arrived broken.",
        "Where is my refund? I ordered a jacket last week and nothing came.",
        "Do you sell gift cards?",
    ]
    assert (
        kept[0]["scenario_id"] == s_id["scenario_id"] and kept[0]["case"]["order_id"] == "ORD-4017"
    )
    assert p.summary(kept) == {
        "prompts": 3,
        "scenarios": 3,
        "with_id": 1,
        "no_id": 1,
        "off_topic": 1,
    }


def test_load_prompts_rebuilds_cases_and_split_by_scenario(tmp_path):
    r, p = _modules()
    path = tmp_path / "prompts.jsonl"
    rows = [
        {"prompt": "Can you check ORD-4017?", "scenario_id": "a"},
        {"prompt": "Order ORD-4017 never arrived, help", "scenario_id": "a"},
        {"prompt": "I want a refund but lost the number", "scenario_id": "b"},
    ]
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    items = p.load_prompts(str(path))
    assert [i["case"]["order_id"] for i in items] == ["ORD-4017", "ORD-4017", None]
    train, held = r.split_holdout(items, 0.5)
    sides = {}
    for it in train:
        sides.setdefault(it["scenario_id"], set()).add("train")
    for it in held:
        sides.setdefault(it["scenario_id"], set()).add("held")
    assert all(len(v) == 1 for v in sides.values()), "a scenario landed on both sides"


def test_checked_in_prompt_set_is_well_formed():
    _, p = _modules()
    path = GRPO / "prompts.jsonl"
    assert path.exists(), (
        "recipes/04-train/grpo/prompts.jsonl is checked in; *.jsonl is ignored so it needs its own unignore line"
    )
    items = p.load_prompts(str(path))
    assert len(items) >= 200
    assert len({i["prompt"] for i in items}) == len(items)
    s = p.summary(items)
    assert min(s["with_id"], s["no_id"], s["off_topic"]) >= 20
    # Every row carries its seed's scenario and seed text, as prompts.py says.
    raw = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    assert all(set(row) == {"prompt", "scenario_id", "seed"} for row in raw)
    assert all(isinstance(row["scenario_id"], str) and row["scenario_id"] for row in raw)
    assert all(row["seed"].strip() for row in raw)


def test_prompt_set_numbers_in_the_readme_are_the_files():
    """The README's counts are recomputed from prompts.jsonl, not remembered."""
    r, p = _modules()
    items = p.load_prompts(str(GRPO / "prompts.jsonl"))
    text = README.read_text(encoding="utf-8")
    s = p.summary(items)
    assert s == {"prompts": 707, "scenarios": 67, "with_id": 576, "no_id": 74, "off_topic": 57}
    assert "707 prompts from 67 situations (576 with an id,\n74 without, 57 off topic" in text
    # The old hash split: 159 holdout prompts from 14 situations, no no-id ones.
    _, held = r.split_holdout(items, 0.2)
    h = p.summary(held)
    assert (h["prompts"], h["scenarios"], h["no_id"]) == (159, 14, 0)
    assert "holdout is 159 prompts from 14 situations" in text
    assert h["with_id"] * 4 == 612 and h["off_topic"] * 4 == 24
    # The stratified split: 163 holdout prompts, every category present.
    train, held = p.split_holdout_stratified(items, 0.2)
    h = p.summary(held)
    assert (h["prompts"], h["with_id"], h["no_id"], h["off_topic"]) == (163, 130, 15, 18)
    assert "holdout 163 prompts: 130 with an id, 15 without,\n18 off topic" in text
    # --balance 0.25 on that train split.
    b = p.summary(p.balance(train, 0.25))
    assert (b["prompts"], b["with_id"], b["no_id"], b["off_topic"]) == (857, 446, 177, 234)
    assert "train split 857 rows: 446\nwith an id, 177 without, 234 off topic" in text


def test_stratified_split_keeps_every_category_in_the_holdout():
    r, p = _modules()
    items = []
    for i in range(12):
        items.append(
            {
                "prompt": f"check ORD-{1000 + i} please",
                "case": r.case_for(f"check ORD-{1000 + i} please"),
                "scenario_id": f"id{i}",
            }
        )
    for i in range(3):
        items.append(
            {
                "prompt": f"I want a refund, lost the number {i}",
                "case": r.case_for("I want a refund, lost the number"),
                "scenario_id": f"no{i}",
            }
        )
    for i in range(3):
        items.append(
            {
                "prompt": f"Do you sell gift cards {i}?",
                "case": r.case_for("Do you sell gift cards?"),
                "scenario_id": f"off{i}",
            }
        )
    train, held = p.split_holdout_stratified(items, 0.2)
    assert len(train) + len(held) == len(items)
    held_cats = {p.category(i["case"]) for i in held}
    assert held_cats == {"with_id", "no_id", "off_topic"}
    assert {i["scenario_id"] for i in train} & {i["scenario_id"] for i in held} == set()
    again = p.split_holdout_stratified(items, 0.2)
    assert [i["prompt"] for i in again[1]] == [i["prompt"] for i in held]


CALL_TEXT = (
    '<tool_call>\n{"name": "lookup_order", "arguments": {"order_id": "ORD-1"}}\n</tool_call>'
)


def test_pass_by_category_reads_reward_and_tool_calls():
    _, p = _modules()
    rows = [
        {"prompt": "check ORD-1001", "reward": 1, "final_text": CALL_TEXT},
        {"prompt": "check ORD-1001", "reward": 0, "final_text": "Sure."},
        {"prompt": "Do you sell gift cards?", "reward": 1, "final_text": "No."},
    ]
    out = p.pass_by_category(rows)
    assert out["with_id"] == {"pass_at_1": 0.5, "tool_call_rate": 0.5, "rows": 2}
    assert out["off_topic"] == {"pass_at_1": 1.0, "tool_call_rate": 0.0, "rows": 1}


def test_balance_repeats_minority_categories_up_to_the_share():
    r, p = _modules()
    items = [_seed(r, f"check ORD-{1000 + i} please") for i in range(18)]
    items += [_seed(r, f"I want a refund, lost the number {i}") for i in range(3)]
    items += [_seed(r, f"Do you sell gift cards {i}?") for i in range(3)]
    out = p.balance(items, 0.25)
    s = p.summary(out)
    assert s["with_id"] == 18
    assert s["no_id"] / len(out) >= 0.25 and s["off_topic"] / len(out) >= 0.25
    # a single minority prompt stops at the repeat cap, not at the share
    capped = p.balance([*items[:18], items[18]], 0.25)
    assert p.summary(capped)["no_id"] == 6
    assert out[: len(items)] == items, "originals first, repeats appended"
    assert p.balance(items, 0.0) == items
    assert p.balance([], 0.5) == []


# -------------------------------------------------------- the Modal scripts


def test_train_modal_imports_and_its_flags_match_the_readme():
    mod = load_modal_script("grpo_train_modal", GRPO / "train_modal.py")
    writer = load_modal_script("grpo_write_prompts_modal", GRPO / "write_prompts_modal.py")
    assert_readme_matches_entrypoints(
        README,
        {
            "recipes/04-train/grpo/train_modal.py": mod.main,
            "recipes/04-train/grpo/write_prompts_modal.py": writer.main,
        },
    )
    remote = inspect.signature(mod.train).parameters
    for name in (
        "loss_type",
        "epsilon_high",
        "scale_rewards",
        "mask_truncated",
        "monitor_every",
        "stop_on",
        "gpu",
    ):
        assert name in remote, name
    assert remote["beta"].default == 0.04 and remote["loss_type"].default == "bnpo"
    assert remote["num_generations"].default == 8 and remote["steps"].default == 40
    assert mod.BASE_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
    assert writer.WRITER_MODEL == "Qwen/Qwen2.5-7B-Instruct"


def test_readme_names_only_things_that_exist():
    text = README.read_text(encoding="utf-8")
    for rel in ("reward.py", "prompts.py", "prompts.jsonl", "write_prompts_modal.py"):
        assert rel in text and (GRPO / rel).exists(), rel
    _, p = _modules()
    for fn in ("split_holdout_stratified", "case_for"):
        assert f"`{fn}`" in text
    assert hasattr(p, "split_holdout_stratified")
    for api in ("HackMonitor", "TrainerCallback", "simulate"):
        assert api in text and hasattr(wai, api)
    assert "tests/recipes/test_grpo_example.py" in text
