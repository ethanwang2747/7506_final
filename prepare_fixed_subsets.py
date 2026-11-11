"""构建 GQA / TextVQA / CV-Bench 2D 的固定评测子集（seed=42）。

输入：data/raw/ 下从 HF 镜像下载的原始 parquet。
输出：
    data/fixed/{gqa,textvqa,cvbench_2d}_{test,dev}.jsonl
    data/fixed/images/{gqa,textvqa,cvbench}/
    data/fixed/manifest.json

选择规则：
    GQA       按 types.semantic 分层，500 test + 100 dev
    TextVQA   按是否有 OCR token 分层，500 test + 100 dev
    CV-Bench  按 task x source 分层，1338 test + 100 dev（覆盖全部 2D 样本）
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
from PIL import Image
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parent
RAW = PROJECT / "data" / "raw"
FIXED = PROJECT / "data" / "fixed"
IMAGES = FIXED / "images"
SEED = 42

GQA_INSTR = RAW / "gqa/testdev_balanced_instructions/testdev-00000-of-00001.parquet"
GQA_IMAGES = RAW / "gqa/testdev_balanced_images/testdev-00000-of-00001.parquet"
TV_PARTS = [RAW / f"textvqa/data/validation-0000{i}-of-00003.parquet" for i in range(3)]
CV_PARQUET = RAW / "cvbench/test_2d.parquet"

SOURCES = {
    "gqa": {
        "repo": "lmms-lab-encoder/GQA",
        "revision": "a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8",
        "files": [str(GQA_INSTR.relative_to(PROJECT)), str(GQA_IMAGES.relative_to(PROJECT))],
    },
    "textvqa": {
        "repo": "lmms-lab-encoder/textvqa",
        "revision": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "files": [str(p.relative_to(PROJECT)) for p in TV_PARTS],
    },
    "cvbench_2d": {
        "repo": "nyu-visionx/CV-Bench",
        "revision": "bc284db50d036958861cb60cdd7b77612052ce0d",
        "files": [str(CV_PARQUET.relative_to(PROJECT))],
    },
}


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray, memoryview)):
        return None
    return str(o)


def save_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=json_default) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_bytes(value):
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, memoryview):
        value = value.tobytes()
    if not isinstance(value, (bytes, bytearray)) or not value:
        raise ValueError("empty or non-bytes image payload")
    return bytes(value)


def save_image(value, path):
    data = image_bytes(value)
    with Image.open(BytesIO(data)) as im:
        im.verify()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return f"{path.relative_to(PROJECT).as_posix()}"


def split_three(df, n_test, n_dev, stratify_col):
    test_df, rest_df = train_test_split(
        df, train_size=n_test, stratify=df[stratify_col], random_state=SEED
    )
    dev_df, _ = train_test_split(
        rest_df, train_size=n_dev, stratify=rest_df[stratify_col], random_state=SEED
    )
    return test_df, dev_df


def prepare_gqa():
    print("[GQA]", flush=True)
    df = pd.read_parquet(GQA_INSTR)
    df["semantic"] = df["types"].apply(lambda d: d.get("semantic"))
    df["structural"] = df["types"].apply(lambda d: d.get("structural"))
    df["detailed"] = df["types"].apply(lambda d: d.get("detailed"))

    test_df, dev_df = split_three(df, 500, 100, "semantic")

    images = pd.read_parquet(GQA_IMAGES)
    img_map = {str(r.id): r.image for r in images.itertuples(index=False)}
    shutil.rmtree(IMAGES / "gqa", ignore_errors=True)
    selected_images = sorted(set(test_df.imageId) | set(dev_df.imageId))
    for image_id in selected_images:
        if image_id not in img_map:
            raise KeyError(f"GQA image missing: {image_id}")
        save_image(img_map[image_id], IMAGES / "gqa" / f"{image_id}.jpg")
    print(f"  images saved: {len(selected_images)}", flush=True)

    def records(split_df, split):
        out = []
        for r in split_df.itertuples(index=False):
            out.append(
                {
                    "dataset": "gqa",
                    "split": split,
                    "id": str(r.id),
                    "image_id": str(r.imageId),
                    "question": r.question,
                    "answer": r.answer,
                    "full_answer": r.fullAnswer,
                    "is_balanced": bool(r.isBalanced),
                    "types": r.types,
                    "groups": r.groups,
                    "semantic_str": r.semanticStr,
                    "image_path": f"data/fixed/images/gqa/{r.imageId}.jpg",
                }
            )
        return out

    save_jsonl(records(test_df, "test"), FIXED / "gqa_test.jsonl")
    save_jsonl(records(dev_df, "dev"), FIXED / "gqa_dev.jsonl")
    return {
        "pool": len(df),
        "test": len(test_df),
        "dev": len(dev_df),
        "stratify": "types.semantic",
        "images": len(selected_images),
        "test_semantic": test_df["semantic"].value_counts().to_dict(),
        "dev_semantic": dev_df["semantic"].value_counts().to_dict(),
        "test_detailed": test_df["detailed"].value_counts().to_dict(),
    }


def prepare_textvqa():
    print("[TextVQA]", flush=True)
    cols = [
        "image_id",
        "question_id",
        "question",
        "answers",
        "ocr_tokens",
        "image_width",
        "image_height",
        "image_classes",
        "set_name",
    ]
    df = pd.concat([pd.read_parquet(p, columns=cols) for p in TV_PARTS], ignore_index=True)
    df["has_ocr"] = df["ocr_tokens"].apply(lambda x: len(x) > 0)

    test_df, dev_df = split_three(df, 500, 100, "has_ocr")
    selected = pd.concat([test_df, dev_df], ignore_index=True)
    selected_qids = set(int(x) for x in selected.question_id)

    img_map = {}
    for part in TV_PARTS:
        ds = pads.dataset(part, format="parquet")
        table = ds.to_table(
            columns=["question_id", "image"],
            filter=pads.field("question_id").isin(list(selected_qids)),
        )
        for row in table.to_pylist():
            img_map[int(row["question_id"])] = row["image"]
    if len(img_map) != len(selected_qids):
        raise RuntimeError(f"TextVQA image rows missing: {len(img_map)} vs {len(selected_qids)}")

    shutil.rmtree(IMAGES / "textvqa", ignore_errors=True)
    selected_images = sorted(set(selected.image_id))
    seen = {}
    for r in selected.itertuples(index=False):
        seen.setdefault(r.image_id, img_map[int(r.question_id)])
    for image_id in selected_images:
        save_image(seen[image_id], IMAGES / "textvqa" / f"{image_id}.jpg")
    print(f"  images saved: {len(selected_images)}", flush=True)

    def records(split_df, split):
        out = []
        for r in split_df.itertuples(index=False):
            out.append(
                {
                    "dataset": "textvqa",
                    "split": split,
                    "image_id": str(r.image_id),
                    "question_id": int(r.question_id),
                    "question": r.question,
                    "answers": list(r.answers),
                    "ocr_tokens": list(r.ocr_tokens),
                    "image_width": int(r.image_width),
                    "image_height": int(r.image_height),
                    "image_classes": list(r.image_classes),
                    "image_path": f"data/fixed/images/textvqa/{r.image_id}.jpg",
                }
            )
        return out

    save_jsonl(records(test_df, "test"), FIXED / "textvqa_test.jsonl")
    save_jsonl(records(dev_df, "dev"), FIXED / "textvqa_dev.jsonl")
    return {
        "pool": len(df),
        "test": len(test_df),
        "dev": len(dev_df),
        "stratify": "has_ocr",
        "images": len(selected_images),
        "test_has_ocr": test_df["has_ocr"].value_counts().to_dict(),
        "dev_has_ocr": dev_df["has_ocr"].value_counts().to_dict(),
    }


def prepare_cvbench():
    print("[CV-Bench 2D]", flush=True)
    df = pd.read_parquet(CV_PARQUET)
    df["strat"] = df["task"].astype(str) + "|" + df["source"].astype(str)
    dev_df, test_df = train_test_split(
        df, train_size=100, stratify=df["strat"], random_state=SEED
    )
    selected = pd.concat([test_df, dev_df], ignore_index=True)

    shutil.rmtree(IMAGES / "cvbench", ignore_errors=True)
    for r in selected.itertuples(index=False):
        save_image(r.image, IMAGES / "cvbench" / f"{int(r.idx):04d}.png")
    print(f"  images saved: {len(selected)}", flush=True)

    def records(split_df, split):
        out = []
        for r in split_df.itertuples(index=False):
            out.append(
                {
                    "dataset": "cvbench_2d",
                    "split": split,
                    "idx": int(r.idx),
                    "type": r.type,
                    "task": r.task,
                    "question": r.question,
                    "choices": list(r.choices),
                    "answer": r.answer,
                    "prompt": r.prompt,
                    "filename": r.filename,
                    "source": r.source,
                    "source_dataset": r.source_dataset,
                    "source_filename": r.source_filename,
                    "target_class": None if pd.isna(r.target_class) else r.target_class,
                    "target_size": None if pd.isna(r.target_size) else int(r.target_size),
                    "bbox": r.bbox,
                    "image_path": f"data/fixed/images/cvbench/{int(r.idx):04d}.png",
                }
            )
        return out

    save_jsonl(records(test_df, "test"), FIXED / "cvbench_2d_test.jsonl")
    save_jsonl(records(dev_df, "dev"), FIXED / "cvbench_2d_dev.jsonl")
    return {
        "pool": len(df),
        "test": len(test_df),
        "dev": len(dev_df),
        "stratify": "task|source",
        "images": len(selected),
        "test_task": test_df["task"].value_counts().to_dict(),
        "dev_task": dev_df["task"].value_counts().to_dict(),
        "test_source": test_df["source"].value_counts().to_dict(),
    }


def verify_and_manifest(splits):
    files = [
        FIXED / "gqa_test.jsonl",
        FIXED / "gqa_dev.jsonl",
        FIXED / "textvqa_test.jsonl",
        FIXED / "textvqa_dev.jsonl",
        FIXED / "cvbench_2d_test.jsonl",
        FIXED / "cvbench_2d_dev.jsonl",
    ]
    file_info = {}
    for path in files:
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        missing = [r["image_path"] for r in records if not (PROJECT / r["image_path"]).exists()]
        if missing:
            raise FileNotFoundError(f"{path.name}: {len(missing)} missing images, e.g. {missing[:3]}")
        file_info[path.name] = {
            "count": len(records),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        print(f"  verified {path.name}: {len(records)} records, {len(file_info[path.name]['sha256'])} sha", flush=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "sources": SOURCES,
        "splits": splits,
        "files": file_info,
        "notes": [
            "GQA uses the testdev balanced split; 500 test + 100 dev sampled by types.semantic.",
            "TextVQA uses the 5,000-question validation split; 500 test + 100 dev sampled by has_ocr.",
            "CV-Bench uses the full 2D set; 1,338 test + 100 dev sampled by task x source.",
            "Images are stored once per image id; image_path is relative to the project root.",
        ],
    }
    with open(FIXED / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  manifest written: {FIXED / 'manifest.json'}", flush=True)
    return manifest


def main():
    FIXED.mkdir(parents=True, exist_ok=True)
    splits = {
        "gqa": prepare_gqa(),
        "textvqa": prepare_textvqa(),
        "cvbench_2d": prepare_cvbench(),
    }
    verify_and_manifest(splits)
    print(json.dumps(splits, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
