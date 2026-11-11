import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT / "data/raw/gqa/val_sceneGraphs.json"


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def valid(rec, ref, W, H, axis, min_area=0.015, min_gap=0.12, max_iou=0.10):
    a, b = rec["bbox"], ref["bbox"]
    if a[2] * a[3] < min_area * W * H or b[2] * b[3] < min_area * W * H:
        return False
    if iou(a, b) > max_iou:
        return False
    acx, acy = a[0] + a[2] / 2, a[1] + a[3] / 2
    bcx, bcy = b[0] + b[2] / 2, b[1] + b[3] / 2
    if axis == "vertical" and abs(acy - bcy) / H < min_gap:
        return False
    if axis == "horizontal" and abs(acx - bcx) / W < min_gap:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(SG) as f:
        sg = json.load(f)
    src = ROOT / f"data/derived/spatial_pairs_{args.split}.jsonl"
    rows = [json.loads(l) for l in open(src)]
    out = []
    stats = {"input": len(rows), "kept": 0}
    for r in rows:
        rel = r["group"].split("|")[1]
        axis = "vertical" if rel in ("above", "below") else "horizontal"
        ok = True
        for im_id, o, p in [(r["source_image_id"], r["source_target"], r["source_reference"]),
                            (r["target_image_id"], r["target_target"], r["target_reference"])]:
            g = sg[im_id]
            if not valid(o, p, g["width"], g["height"], axis):
                ok = False; break
        if ok:
            out.append(r)
    out_path = Path(args.out) if args.out else ROOT / f"data/derived/spatial_pairs_{args.split}_strict.jsonl"
    with open(out_path, "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print("input", len(rows), "kept", len(out), out_path)


if __name__ == "__main__":
    main()
