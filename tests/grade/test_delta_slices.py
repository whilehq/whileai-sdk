"""The slice table is on by default when cases carry a slice, weakest first.

The shape mirrors an external case study on whileai 0.126
(gentlyventures.com/casestudies/whileai): 46 cases over six question types,
overall 0.46 -> 0.76, with "missing file" stuck near zero (0.00 -> 0.17 there)
and "month count" short of the rest (0.25 -> 0.62). The most useful view was
per question type; the report should print it without the user asking.
Everything here is fixed-seed and offline.
"""

from __future__ import annotations

from whileai.simulations.score.delta import (
    SLICE_MIN_TASKS,
    delta_report,
    format_delta_report,
    slice_min_tasks,
)

#: slice -> (tasks, passing before, passing after); 46 tasks, 21 -> 35 passing.
CASE_STUDY = {
    "missing_file": (8, 0, 1),
    "month_count": (8, 2, 5),
    "lookup": (10, 6, 9),
    "totals": (10, 7, 10),
    "date_range": (6, 4, 6),
    "rename": (4, 2, 4),
}


def _rows(spec=CASE_STUDY, *, key: str | None = "category"):
    """One row per case on each side, paired by task id. A case that
    passed before still passes after; the first ``after`` cases pass."""
    before, after = [], []
    for slice_name, (n, pa, pb) in spec.items():
        for t in range(n):
            base = {"task_id": f"{slice_name}-{t}", "prompt": f"{slice_name} question {t}"}
            if key:
                base[key] = slice_name
            before.append({**base, "reward": int(t < pa), "markers": {"well_formed": 1.0}})
            after.append({**base, "reward": int(t < pb), "markers": {"well_formed": 1.0}})
    return before, after


def _section(text: str) -> str:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("by "))
    end = next((i for i in range(start + 1, len(lines)) if not lines[i].startswith(" ")), None)
    return "\n".join(lines[start:end])


SNAPSHOT = """\
by category: 6 slices, 2 weak, weakest first
  (WEAK = after under 0.50 or no gain the 95% interval supports; low n = under 6 paired tasks, too few for any delta to be read)
  missing_file                 0.000 -> 0.125  +0.125 [+0.000..+0.375]  flat  (n=8 tasks, 8/8 rows)  WEAK: still low, no clear gain
  month_count                  0.250 -> 0.625  +0.375 [+0.000..+0.750]  flat  (n=8 tasks, 8/8 rows)  WEAK: no clear gain
  lookup                       0.600 -> 0.900  +0.300 [+0.000..+0.600]  flat  (n=10 tasks, 10/10 rows)
  totals                       0.700 -> 1.000  +0.300 [+0.000..+0.600]  flat  (n=10 tasks, 10/10 rows)
  date_range                   0.667 -> 1.000  +0.333 [+0.000..+0.833]  flat  (n=6 tasks, 6/6 rows)
  rename                       0.500 -> 1.000  +0.500 [+0.000..+1.000]  flat  (n=4 tasks, 4/4 rows)  low n"""


def test_low_n_threshold_is_the_exact_sign_test_floor():
    # 2 * 0.5**n <= alpha: five same-way tasks is p=0.0625, six is 0.031.
    assert SLICE_MIN_TASKS == slice_min_tasks(0.05) == 6
    assert slice_min_tasks(0.01) == 8


def test_slice_table_is_on_by_default_and_leads_with_the_stuck_slice():
    before, after = _rows()
    assert len(before) == 46
    report = delta_report(before, after, target="pass_at_1", n_boot=500, seed=0)
    assert report["by"] == "category" and report["by_source"] == "auto"
    assert set(report["groups"]) == set(CASE_STUDY)
    assert report["metrics"]["pass_at_1"]["mean_a"] == 21 / 46
    assert report["metrics"]["pass_at_1"]["mean_b"] == 35 / 46
    # the stuck slice is flagged first, and for both reasons
    assert report["groups_weak"][0] == "missing_file"
    assert report["groups"]["missing_file"]["weak_reasons"] == ["low", "no_gain"]
    # small slices are marked, not read
    assert report["groups"]["rename"]["low_n"] is True
    assert report["groups"]["missing_file"]["low_n"] is False
    text = format_delta_report(report)
    assert str(report) == text
    assert _section(text) == SNAPSHOT


def test_slice_key_is_also_read_and_by_still_wins():
    before, after = _rows(key="slice")
    report = delta_report(before, after, n_boot=200)
    assert report["by"] == "slice" and report["by_source"] == "auto"
    before, after = _rows()
    for row in before + after:
        row["markers"]["kind"] = row["category"][:1]
    report = delta_report(before, after, by="kind", n_boot=200)
    assert report["by"] == "kind" and report["by_source"] == "given"


def test_no_slice_field_leaves_the_output_unchanged():
    plain_before, plain_after = _rows(key=None)
    report = delta_report(plain_before, plain_after, target="pass_at_1", n_boot=300)
    assert report["groups"] is None and report["by_source"] is None
    assert report["groups_weak"] == []
    text = format_delta_report(report)
    assert "\nby " not in text and "WEAK" not in text
    # the slice section is the only thing a slice field adds
    before, after = _rows()
    off = delta_report(before, after, target="pass_at_1", n_boot=300, by=False)
    assert off["groups"] is None
    assert format_delta_report(off) == text


def test_one_slice_value_is_not_a_table():
    before, after = _rows({"only": (12, 3, 9)})
    report = delta_report(before, after, n_boot=200)
    assert report["groups"] is None
