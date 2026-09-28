"""Stage 1 of the cascade: a byte n-gram hashed linear classifier, sub-millisecond, no tokenizer.

    python byte_stage.py --train out/train_v4.jsonl --val out/val_v4.jsonl

Character 1- to 4-grams over the raw bytes (``analyzer="char_wb"``), hashed
into 2**20 features, logistic regression. Bytes see what WordPiece maps to
``[UNK]`` (homoglyphs, zero-width joins), so this is the layer that should
catch obfuscation cheaply; it is also the layer a paraphrase walks past. It is
reported as its own arm on both frozen tests and timed per row, single
thread. The cascade rule (stage 2 only inside the uncertain band) is
reported as a latency mix, not trained.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from data import read_jsonl
from metrics import by_slice, choose_threshold, table
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(OUT / "train.jsonl"))
    ap.add_argument("--val", default=str(OUT / "val.jsonl"))
    ap.add_argument("--tag", default="byte")
    a = ap.parse_args()
    train = read_jsonl(Path(a.train))
    val = read_jsonl(Path(a.val))
    vec = HashingVectorizer(
        analyzer="char_wb", ngram_range=(1, 4), n_features=2**20, alternate_sign=False, norm="l2"
    )
    t0 = time.time()
    x_tr = vec.transform([r["text"] for r in train])
    clf = LogisticRegression(max_iter=3000, C=4.0)
    clf.fit(x_tr, [r["label"] for r in train])
    fit_s = time.time() - t0
    cal = clf.predict_proba(vec.transform([r["text"] for r in val if r["label"] == 0]))[
        :, 1
    ].tolist()
    thr = choose_threshold([0] * len(cal), cal)
    result = {"tag": a.tag, "threshold": thr, "fit_seconds": round(fit_s, 1), "n_train": len(train)}
    for name, path in (("test", HERE / "test.jsonl"), ("test_hard", HERE / "test_hard.jsonl")):
        rows = read_jsonl(path)
        scores = clf.predict_proba(vec.transform([r["text"] for r in rows]))[:, 1].tolist()
        per = by_slice(rows, scores, thr, n_boot=1000)
        result[name] = {"per_slice": per, "scores": scores}
        print(table(per, f"byte-stage ({name})"))
    # latency per row, single-threaded, 512-token-sized text (about 2,000 characters)
    text = " ".join(["the quick brown fox jumps over the lazy dog"] * 50)[:2000]
    for _ in range(20):
        clf.predict_proba(vec.transform([text]))
    ts = []
    for _ in range(300):
        t = time.perf_counter()
        clf.predict_proba(vec.transform([text]))
        ts.append((time.perf_counter() - t) * 1000)
    result["latency_ms_2000_chars"] = {
        "p50": round(float(np.percentile(ts, 50)), 3),
        "p99": round(float(np.percentile(ts, 99)), 3),
    }
    print("latency per 2,000-char row:", result["latency_ms_2000_chars"])
    with open(OUT / f"{a.tag}_model.pkl", "wb") as fh:
        pickle.dump(clf, fh)
    result["model_kb"] = round((OUT / f"{a.tag}_model.pkl").stat().st_size / 1024)
    (OUT / f"scores_{a.tag}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
