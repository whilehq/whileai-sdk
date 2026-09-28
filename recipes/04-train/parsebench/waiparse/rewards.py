"""RL rewards built from ParseBench's own scorer code, applied to training data we
generate ground truth for. The benchmark's rules are the reward; its pages are not.
"""

import csv
import io
import random
import re

from parse_bench.evaluation.metrics.parse.rules_chart import ChartDataPointRule

# ParseBench's chart rules use these relative tolerances (count share in chart.jsonl).
TOLERANCES = [(0.01, 0.30), (0.05, 0.37), (0.10, 0.23), (0.20, 0.10)]


def chart_rules(csv_text: str, max_rules: int = 24, rng: random.Random | None = None) -> list[dict]:
    """Data-point rules from a chart's source table: value + [row label, column header]."""
    rng = rng or random.Random(0)
    rows = [r for r in csv.reader(io.StringIO(csv_text.strip())) if any(c.strip() for c in r)]
    if len(rows) < 2:
        return []
    header, body = rows[0], rows[1:]
    rules = []
    for row in body:
        for j in range(1, min(len(row), len(header))):
            v = row[j].strip()
            if _num(v) is None or not row[0].strip():
                continue
            labels = [row[0].strip()]
            if len(header) > 2 and header[j].strip():
                labels.append(header[j].strip())
            tol = rng.choices([t for t, _ in TOLERANCES], weights=[w for _, w in TOLERANCES])[0]
            rules.append(
                {
                    "type": "chart_data_point",
                    "labels": labels,
                    "value": v,
                    "max_diffs": 0,
                    "normalize_numbers": True,
                    "relative_tolerance": tol,
                    "id": f"r{len(rules)}",
                }
            )
    rng.shuffle(rules)
    return rules[:max_rules]


def chart_reward(output: str, rules: list[dict]) -> float:
    """Share of data-point rules the output passes (ParseBench ChartDataPointMatch)."""
    if not rules:
        return 0.0
    passed = 0
    for r in rules:
        try:  # noqa: SIM105  # a rule that cannot run counts as failed
            passed += bool(ChartDataPointRule(r).run(output)[0])
        except Exception:
            pass
    return passed / len(rules)


def _num(v: str) -> float | None:
    try:
        return float(re.sub(r"[,\s$€£%]", "", v))
    except ValueError:
        return None
