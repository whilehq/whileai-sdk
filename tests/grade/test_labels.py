"""Human labels with annotator records (Lambert 2025, chapter Preference
Data)."""

from __future__ import annotations

import json

import whileai.simulations as wai
from whileai.simulations.score.agreement import judge_agreement
from whileai.simulations.score.labels import annotator_agreement, attach_labels


def _row(i, reward=1):
    return {
        "prompt": f"ask {i}",
        "final_text": f"reply {i}",
        "scenario_id": f"s{i}",
        "rollout_index": 0,
        "reward": reward,
        "steps": [],
        "messages": [],
    }


def test_attach_labels_matches_by_identity_and_takes_the_majority(tmp_path):
    rows = [_row(0), _row(1), _row(2), _row(3)]
    labels = [
        {"scenario_id": "s0", "rollout_index": 0, "label": 1, "annotator": "ana"},
        {"scenario_id": "s0", "rollout_index": 0, "label": 1, "annotator": "ben"},
        {
            "prompt": "ask 1",
            "final_text": "reply 1",
            "label": 0,
            "annotator": "ana",
            "note": "wrong",
        },
        {"prompt": "ask 1", "final_text": "reply 1", "label": 1, "annotator": "ben"},
        {"scenario_id": "s2", "rollout_index": 0, "reward": 0},
        {"scenario_id": "s9", "rollout_index": 0, "label": 1},
        {"scenario_id": "s3", "rollout_index": 0, "label": "maybe"},
    ]
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in labels))
    out, report = attach_labels(rows, str(path), annotator="cal")
    assert report["labels"] == 7 and report["matched"] == 5
    assert report["unmatched"] == 1 and report["unmatched_keys"] == ["s9#0"]
    assert report["invalid"] == 1 and report["rows_labeled"] == 3 and report["ties"] == 1
    assert report["annotators"] == ["ana", "ben", "cal"]
    assert out[0]["gold_reward"] == 1 and len(out[0]["gold_labels"]) == 2
    assert "gold_reward" not in out[1] and out[1]["gold_labels"][0]["note"] == "wrong"
    assert out[2]["gold_reward"] == 0 and out[2]["gold_labels"][0]["annotator"] == "cal"
    assert out[2]["gold_labels"][0]["kind"] == "human" and out[2]["gold_labels"][0]["ts"]
    assert "gold_labels" not in out[3]
    assert any("split evenly" in w for w in report["warnings"])
    # the judge check reads the majority label as before
    agreement = judge_agreement(out)
    assert agreement["n"] == 2  # rows 0 and 2 carry a gold label


def test_attach_labels_mapping_form_appends_and_replace_resets():
    rows = [_row(0)]
    attach_labels(rows, {"s0#0": 0}, annotator="ana")
    attach_labels(rows, {"s0#0": 1}, annotator="ben")
    assert len(rows[0]["gold_labels"]) == 2 and "gold_reward" not in rows[0]
    attach_labels(rows, {"s0#0": 1}, annotator="cal")
    assert rows[0]["gold_reward"] == 1
    attach_labels(rows, {"s0#0": 0}, annotator="dee", replace=True)
    assert len(rows[0]["gold_labels"]) == 1 and rows[0]["gold_reward"] == 0


def test_annotator_agreement_reports_pairs_kappa_and_disagreements():
    rows = [_row(i) for i in range(6)]
    ana = {f"s{i}#0": (1 if i < 4 else 0) for i in range(6)}
    ben = {f"s{i}#0": (1 if i < 3 else 0) for i in range(6)}
    attach_labels(rows, ana, annotator="ana")
    attach_labels(rows, ben, annotator="ben")
    report = annotator_agreement(rows)
    assert report["per_annotator"]["ana"] == {"labels": 6, "pass_share": round(4 / 6, 4)}
    assert report["multi_labeled"] == 6 and report["unanimous"] == round(5 / 6, 4)
    assert report["pair"] == ["ana", "ben"] and 0.5 < report["kappa"] < 0.8
    assert report["disagreements"] == [{"prompt": "ask 3", "labels": {"ana": 1, "ben": 0}}]
    single = annotator_agreement([_row(0)])
    assert single["multi_labeled"] == 0 and single["kappa"] is None


def test_public_surface():
    assert "attach_labels" in wai.__all__ and "annotator_agreement" in wai.__all__
