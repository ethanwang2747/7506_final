import json, re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/audit"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "testdev_balanced": ROOT / "data/raw/gqa/testdev_balanced_instructions/testdev-00000-of-00001.parquet",
    "testdev_all": ROOT / "data/raw/gqa/testdev_all_instructions/testdev-00000-of-00001.parquet",
    "val_balanced": ROOT / "data/raw/gqa/val_balanced_instructions/val-00000-of-00001.parquet",
}
SG = ROOT / "data/raw/gqa/val_sceneGraphs.json"
TRAIN_SG = ROOT / "data/raw/gqa/train_sceneGraphs.json"

PAT = re.compile(r"select: (.+?) \((\d+)\)->verify rel: (.+),([^,]+),o \(([^)]+)\)")


def audit_pool(path):
    df = pd.read_parquet(path)
    df["detailed"] = df.types.map(lambda t: t.get("detailed") if isinstance(t, dict) else None)
    sp = df[df.detailed.isin(["relVerify", "relVerifyCo"])].copy()
    sp["simple"] = sp.semanticStr.str.count("->") == 1
    sp = sp[sp.simple].copy()
    parsed = sp.semanticStr.map(lambda x: PAT.search(x).groups() if PAT.search(x) else None)
    sp = sp[parsed.notna()].copy()
    sp[["subject", "subject_id", "object", "relation", "object_id"]] = pd.DataFrame(parsed.dropna().tolist(), index=sp.index)
    present = sp[~sp.object_id.eq("-")]
    missing = sp[sp.object_id.eq("-")]
    g = sp.groupby("question").agg(n=("id", "size"), n_img=("imageId", "nunique"), n_ans=("answer", "nunique"))
    multi = g[(g.n_img >= 2) & (g.n_ans >= 2)]
    gp = present.groupby("question").agg(n=("id", "size"), n_img=("imageId", "nunique"), n_ans=("answer", "nunique"))
    multi_present = gp[(gp.n_img >= 2) & (gp.n_ans >= 2)]
    return {
        "rows": int(len(df)),
        "unique_images": int(df.imageId.nunique()),
        "simple_rel_verify_rows": int(len(sp)),
        "yes_ref_present": int(len(present)),
        "no_ref_missing": int(len(missing)),
        "exact_question_multi_image_both_answers": int(len(multi)),
        "exact_question_multi_image_both_answers_ref_present": int(len(multi_present)),
    }


def coverage(sg_path, image_ids):
    with open(sg_path) as f:
        sg = json.load(f)
    keys = set(sg)
    stripped = {x[1:] if x.startswith("n") else x for x in image_ids}
    return {"scene_graph_images": len(keys), "covered": len(keys & stripped), "total": len(stripped)}


def main():
    out = {}
    for name, path in SOURCES.items():
        out[name] = audit_pool(path)
        print(name, out[name])
    testdev = pd.read_parquet(SOURCES["testdev_balanced"])
    val = pd.read_parquet(SOURCES["val_balanced"])
    out["coverage"] = {
        "testdev_balanced": {
            "val": coverage(SG, set(testdev.imageId.astype(str))),
            "train": coverage(TRAIN_SG, set(testdev.imageId.astype(str))),
        },
        "val_balanced": {"val": coverage(SG, set(val.imageId.astype(str)))},
    }
    print("coverage", json.dumps(out["coverage"], indent=2))
    (OUT / "gqa_pairing_audit.json").write_text(json.dumps(out, indent=2))
    print("saved", OUT / "gqa_pairing_audit.json")


if __name__ == "__main__":
    main()
