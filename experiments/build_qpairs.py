"""Same-image question pairs: identical image and boxes, opposite relation word.

For each image in the test pairs we ask both the group question and its
opposite (left <-> right). Labels are opposite by construction, while the
target's absolute position is identical. A position-only rule is therefore at
chance by construction.

Output: data/derived/spatial_qpairs_test.jsonl
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT

GROUPS = ["chair|to the left of|table", "man|to the left of|person",
          "man|to the right of|woman", "person|to the right of|man",
          "tree|to the left of|building", "woman|to the left of|man"]
OPP = {"to the left of": "to the right of", "to the right of": "to the left of",
       "above": "below", "below": "above"}

rows = [json.loads(l) for l in open(ROOT / "data/derived/spatial_pairs_test.jsonl")]
rows = [r for r in rows if r["group"] in GROUPS]
counts, sel = {}, []
for r in rows:
    n = counts.get(r["group"], 0)
    if n >= 40:
        continue
    counts[r["group"]] = n + 1
    sel.append(r)

out = []
for r in sel:
    a, rel, b = r["group"].split("|")
    for role in ("source", "target"):
        base = 1 if role == "source" else 0
        iid = r[f"{role}_image_id"]
        for q, lab, tag in ((r["question"], base, "true"),
                            (f"Is the {a} {OPP[rel]} the {b}?", 1 - base, "opp")):
            out.append({"pair_id": f"{r['pair_id']}:{role}:{tag}",
                        "group": r["group"], "image_id": iid,
                        "question": q, "label": lab})
with open(ROOT / "data/derived/spatial_qpairs_test.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"{len(sel)} records, {len(out)} examples")
