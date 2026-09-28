"""Surface-cue probes: what a bag of words, or length alone, gets on each test slice.

    python shortcut_probe.py

Two logistic regressions on the same training rows as the model: word uni-
and bigram counts (50k features), and the row's character length plus token
count only. A slice where either matches the encoder is solvable by surface
cues, and the recipe says so. Writes ``out/shortcut_probe.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from data import read_jsonl
from metrics import by_slice, table
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def length_features(texts: list[str]) -> np.ndarray:
    return np.array([[len(t), len(t.split()), t.count("\n")] for t in texts], dtype=float)


def main() -> None:
    train = read_jsonl(OUT / "train.jsonl")
    val = read_jsonl(OUT / "val.jsonl")
    test = read_jsonl(HERE / "test.jsonl")
    y_tr = np.array([r["label"] for r in train])
    result: dict = {}
    for name in ("bag_of_words", "length_only"):
        if name == "bag_of_words":
            vec = CountVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)
            x_tr = vec.fit_transform([r["text"] for r in train])
            x_va = vec.transform([r["text"] for r in val])
            x_te = vec.transform([r["text"] for r in test])
        else:
            x_tr = length_features([r["text"] for r in train])
            x_va = length_features([r["text"] for r in val])
            x_te = length_features([r["text"] for r in test])
            mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-9
            x_tr, x_va, x_te = (x_tr - mu) / sd, (x_va - mu) / sd, (x_te - mu) / sd
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(x_tr, y_tr)
        from metrics import choose_threshold

        cal = clf.predict_proba(x_va)[:, 1].tolist()
        thr = choose_threshold([r["label"] for r in val], cal)
        scores = clf.predict_proba(x_te)[:, 1].tolist()
        per = by_slice(test, scores, thr, n_boot=300)
        result[name] = {"threshold": thr, "per_slice": per}
        print(table(per, name))
    (OUT / "shortcut_probe.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
