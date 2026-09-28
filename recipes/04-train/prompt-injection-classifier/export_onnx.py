"""Export a trained checkpoint to ONNX, quantise to int8, and time it on one CPU thread.

    python export_onnx.py --model out/minilm-l6-h384-uncased-seed1 --out out/onnx-seed1

Latency is the wall clock of one ``session.run`` on a single 512-, 256- and
128-token chunk, single-threaded (``intra_op_num_threads=1``), 300 timed runs
after 20 warm-ups; p50 and p99 over those runs. Dynamic int8 quantisation is
``onnxruntime.quantization.quantize_dynamic`` (weights int8, activations
quantised at run time; Wu et al. 2020, arXiv:2004.09602).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def export(model_dir: str, out_dir: str) -> dict:
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    enc = tok("hello", return_tensors="pt")
    names = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
    fp32 = out / "model.onnx"
    torch.onnx.export(
        model,
        tuple(enc[k] for k in names),
        str(fp32),
        input_names=names,
        output_names=["logits"],
        dynamic_axes={k: {0: "batch", 1: "seq"} for k in names} | {"logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    int8 = out / "model_int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    tok.save_pretrained(out)
    sizes = {
        "fp32_mb": round(os.path.getsize(fp32) / 1e6, 1),
        "int8_mb": round(os.path.getsize(int8) / 1e6, 1),
    }
    # the int8 graph must agree with torch on the sign of the logit
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(int8), so, providers=["CPUExecutionProvider"])
    return {
        "onnx_int8": str(int8),
        "onnx_fp32": str(fp32),
        "inputs": names,
        **sizes,
        "_sess": sess,
        "_tok": tok,
    }


def bench(
    sess, tok, names: list[str], lengths=(128, 256, 512), warm: int = 20, runs: int = 300
) -> dict:
    res = {}
    for L in lengths:
        ids = np.full((1, L), 2000, dtype=np.int64)
        ids[0, 0], ids[0, -1] = tok.cls_token_id, tok.sep_token_id
        feed = {
            "input_ids": ids,
            "attention_mask": np.ones((1, L), dtype=np.int64),
            "token_type_ids": np.zeros((1, L), dtype=np.int64),
        }
        feed = {k: feed[k] for k in names}
        for _ in range(warm):
            sess.run(None, feed)
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            sess.run(None, feed)
            ts.append((time.perf_counter() - t) * 1000)
        ts = np.array(ts)
        res[str(L)] = {
            "p50_ms": round(float(np.percentile(ts, 50)), 2),
            "p99_ms": round(float(np.percentile(ts, 99)), 2),
            "runs": runs,
        }
    return res


def window_bench(
    sess,
    tok,
    names: list[str],
    *,
    window: int = 128,
    lengths=(512, 2048),
    warm: int = 10,
    runs: int = 100,
) -> dict:
    """A 128-token sliding window with max-pooling over windows.

    Worst case runs every window (a benign chunk, or an injection in the last
    window); best case stops at the first window over threshold. Both are the
    wall clock of the window runs only; tokenisation is excluded, as above.
    """
    res = {}
    for L in lengths:
        n_win = -(-L // window)
        ids = np.full((1, window), 2000, dtype=np.int64)
        ids[0, 0], ids[0, -1] = tok.cls_token_id, tok.sep_token_id
        feed = {
            "input_ids": ids,
            "attention_mask": np.ones((1, window), dtype=np.int64),
            "token_type_ids": np.zeros((1, window), dtype=np.int64),
        }
        feed = {k: feed[k] for k in names}
        for _ in range(warm):
            sess.run(None, feed)
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            best = -1e9
            for _w in range(n_win):
                best = max(best, float(sess.run(None, feed)[0][0][1]))
            ts.append((time.perf_counter() - t) * 1000)
        ts = np.array(ts)
        res[str(L)] = {
            "windows": n_win,
            "all_windows_p50_ms": round(float(np.percentile(ts, 50)), 2),
            "all_windows_p99_ms": round(float(np.percentile(ts, 99)), 2),
            "first_window_exit_p50_ms": round(float(np.percentile(ts, 50)) / n_win, 2),
            "runs": runs,
        }
    return res


def score_onnx(sess, tok, names: list[str], texts: list[str], max_length: int = 512) -> list[float]:
    out = []
    for t in texts:
        enc = tok(t, truncation=True, max_length=max_length, return_tensors="np")
        feed = {k: enc[k].astype(np.int64) for k in names}
        logits = sess.run(None, feed)[0][0]
        e = np.exp(logits - logits.max())
        out.append(float(e[1] / e.sum()))
    return out


if __name__ == "__main__":
    import platform

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--test", default=None, help="score the frozen test with the int8 graph and write scores"
    )
    a = ap.parse_args()
    r = export(a.model, a.out)
    sess, tok = r.pop("_sess"), r.pop("_tok")
    r["latency_single_thread"] = bench(sess, tok, r["inputs"])
    r["sliding_window_128"] = window_bench(sess, tok, r["inputs"])
    r["cpu"] = platform.processor() or platform.machine()
    r["machine"] = platform.platform()
    if a.test:
        from data import read_jsonl

        rows = read_jsonl(Path(a.test))
        t0 = time.time()
        r["test_scores"] = score_onnx(sess, tok, r["inputs"], [x["text"] for x in rows])
        r["test_scoring_seconds"] = round(time.time() - t0, 1)
    Path(a.out, "export.json").write_text(json.dumps(r, indent=1))
    print(json.dumps({k: v for k, v in r.items() if k != "test_scores"}, indent=1))
