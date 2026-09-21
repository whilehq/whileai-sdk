"""Score the two arms against the base noise floor.

PRE-REGISTERED before any arm output was looked at.

Primary metric - `nonassistant_leak`: the share of holdout replies that
contain content which, in the training file, only ever appears inside a
message the export masked off (loss_mask == 0): a tool-result JSON object
(a `"status"` or `"data"` key) or a rendered role marker. If supervising
those tokens teaches the model to emit them, `as_exported` leaks more than
`mask_honored`. Lower is better.

Secondary - mean reply length in characters (the masked-off turns are the
long ones).
Tertiary - wai's own style_report clean share.

Intervals: paired bootstrap over the 57 holdout prompts, 10k resamples.
Noise floor: the three base passes over the same prompts.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys

from whileai.config import provenance

TOOL_JSON = re.compile(r'"(status|data|order_id|ref|updated_at)"\s*:')
ROLE_MARK = re.compile(r"(^|\n)\s*(user|assistant|tool|system)\s*:", re.I)


def leaked(text: str) -> int:
    return int(bool(TOOL_JSON.search(text) or ROLE_MARK.search(text)))


def rate(gens: list[str]) -> float:
    return sum(leaked(g) for g in gens) / len(gens)


def paired_delta(a: list[str], b: list[str], fn, n_boot: int = 10000, seed: int = 0):
    """fn(list)->float. Returns (delta, lo, hi) for a - b, paired by index."""
    rng = random.Random(seed)
    n = len(a)
    obs = fn(a) - fn(b)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(fn([a[i] for i in idx]) - fn([b[i] for i in idx]))
    boots.sort()
    return obs, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def main() -> None:
    print(provenance(), file=sys.stderr)
    res = json.loads(open("raw_results.json").read())
    base = res["base_evals"]
    arms = res["arms"]

    print("supervised fraction:", res["supervised_fraction"])
    print()

    print("=== noise floor (base, 3 sampled passes) ===")
    base_rates = [rate(g) for g in base]
    base_lens = [statistics.mean(len(x) for x in g) for g in base]
    for i, (r, ln) in enumerate(zip(base_rates, base_lens)):
        print(f"  seed {i}: leak {r:.3f}  mean_len {ln:.0f}")
    band = max(base_rates) - min(base_rates)
    print(f"  leak band {min(base_rates):.3f}-{max(base_rates):.3f} (width {band:.3f})")
    print()

    print("=== arms ===")
    out = {
        "supervised_fraction": res["supervised_fraction"],
        "base_leak": base_rates,
        "base_len": base_lens,
        "noise_band": band,
        "arms": {},
    }
    for name, d in arms.items():
        g = d["gens"]
        r, ln = rate(g), statistics.mean(len(x) for x in g)
        print(f"  {name}: leak {r:.3f}  mean_len {ln:.0f}  train_loss {d['train_loss']:.4f}")
        out["arms"][name] = {"leak": r, "mean_len": ln, "train_loss": d["train_loss"]}
    print()

    a = arms["as_exported"]["gens"]
    b = arms["mask_honored"]["gens"]

    print("=== paired delta, as_exported - mask_honored (95% CI, 10k boot) ===")
    for label, fn in (
        ("leak rate", rate),
        ("mean length", lambda g: statistics.mean(len(x) for x in g)),
    ):
        d, lo, hi = paired_delta(a, b, fn)
        excl = "excludes zero" if (lo > 0 or hi < 0) else "includes zero"
        print(f"  {label}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] -- {excl}")
        out.setdefault("deltas", {})[label] = [d, lo, hi]

    # each arm against base pass 0, paired
    print()
    print("=== each arm vs base (paired, leak rate) ===")
    for name in ("as_exported", "mask_honored"):
        d, lo, hi = paired_delta(arms[name]["gens"], base[0], rate)
        print(f"  {name} - base: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        out.setdefault("vs_base", {})[name] = [d, lo, hi]

    # SDK-native view
    try:
        import whileai as wai

        meta = res["holdout_meta"]
        for name in ("as_exported", "mask_honored"):
            rows = [
                {"prompt": m["prompt"], "final_text": t, "task_id": m["scenario_id"]}
                for m, t in zip(meta, arms[name]["gens"])
            ]
            sr = wai.simulations.style_report(rows)
            clean = {k: v for k, v in sr.items() if k not in ("warnings",)}
            print(f"\n=== style_report {name} ===")
            for k, v in list(clean.items())[:8]:
                if isinstance(v, dict) and "clean" in v:
                    print(f"  {k}: clean {v['clean']}")
            out.setdefault("style", {})[name] = str(clean)[:2000]
    except Exception as e:
        print(f"style_report failed: {type(e).__name__}: {e}")

    json.dump(out, open("results.json", "w"), indent=1)
    print("\nwrote results.json")


if __name__ == "__main__":
    main()
