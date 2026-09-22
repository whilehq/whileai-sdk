import json
import random
from collections import Counter, defaultdict

import whileai as wai

tasks = json.load(open("tasks_all.json"))

# unique ids; keep scenario for the split so near-copy scenarios cannot straddle it
seen = Counter()
for t in tasks:
    sc = t["task_id"]
    seen[sc] += 1
    t["scenario"] = sc
    t["task_id"] = f"{sc}#{seen[sc]}"
    # the text that actually varies: the customer's own turns
    t["prompt"] = "\n".join(m.get("content") or "" for m in t["prefix"] if m["role"] == "user")

by_scn = defaultdict(list)
for t in tasks:
    by_scn[t["scenario"]].append(t)
scns = sorted(by_scn)
random.Random(0).shuffle(scns)

# holdout: half the scenarios, and it must be able to discriminate --
# keep every tier and both targets represented rather than sampling traffic
cut = int(len(scns) * 0.5)
hold_scn, train_scn = set(scns[:cut]), set(scns[cut:])
holdout = [t for s in hold_scn for t in by_scn[s]]
train = [t for s in train_scn for t in by_scn[s]]

print(f"scenarios {len(scns)}: {len(train_scn)} train, {len(hold_scn)} holdout")
print("train", len(train), Counter(t["target"] for t in train))
print("holdout", len(holdout), Counter(t["target"] for t in holdout))
print("holdout tiers", Counter(t["tier"] for t in holdout).most_common())
print("holdout domains", Counter(t["domain"] for t in holdout).most_common())

clean, rep = wai.decontaminate(train, against=holdout)
print("\n--- decontaminate (prompt = the customer turns only)")
print("n_contaminated", rep["n_contaminated"], "kept", len(clean))
print("rules_skipped", rep.get("rules_skipped"))
print("notes", rep.get("notes"))

# control: what the same call says when the system policy is in the prompt
ctl_tr = [dict(t, prompt=t["system"][:2000] + "\n" + t["prompt"]) for t in train]
ctl_ho = [dict(t, prompt=t["system"][:2000] + "\n" + t["prompt"]) for t in holdout]
_, rep2 = wai.decontaminate(ctl_tr, against=ctl_ho)
print("\n--- control: same rows, system policy prepended")
print("n_contaminated", rep2["n_contaminated"], "of", len(ctl_tr))

json.dump(
    {
        "train": clean,
        "holdout": holdout,
        "decontam": {
            "n_contaminated": rep["n_contaminated"],
            "kept": len(clean),
            "rules_skipped": rep.get("rules_skipped"),
            "notes": rep.get("notes"),
            "with_policy_in_prompt": rep2["n_contaminated"],
        },
    },
    open("split.json", "w"),
)
print("\nwrote split.json")
