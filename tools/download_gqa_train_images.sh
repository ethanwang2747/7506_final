#!/usr/bin/env bash
# Download GQA train images from the ModelScope mirror (lmms-lab/GQA).
#
# The mirror stores the images as 22 parquet shards (~480 MB each, ~10.5 GB
# total) instead of the 21.8 GB images.zip on the Stanford host, which is
# unusably slow from this network. Files are resumable; rerunning skips
# shards that already have the expected size.
#
# Usage:
#   tools/download_gqa_train_images.sh            # all 22 shards
#   MAX_SHARDS=4 tools/download_gqa_train_images.sh
#
# Output: data/raw/gqa/train_all_images/train-000NN-of-00022.parquet
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/raw/gqa/train_all_images"
BASE="https://www.modelscope.cn/api/v1/datasets/lmms-lab/GQA/repo?Revision=master&FilePath=train_all_images"
MAX_SHARDS="${MAX_SHARDS:-22}"

mkdir -p "$OUT"

for i in $(seq 0 $((MAX_SHARDS - 1))); do
  name=$(printf "train-%05d-of-00022.parquet" "$i")
  url="$BASE/$name"
  dst="$OUT/$name"
  expected=485491749
  if [ -f "$dst" ] && [ "$(stat -c%s "$dst")" -ge $((expected * 95 / 100)) ]; then
    echo "[skip] $name ($(du -h "$dst" | cut -f1))"
    continue
  fi
  echo "[get ] $name"
  curl -L --fail --retry 8 --retry-delay 3 --retry-all-errors \
    -C - -o "$dst" "$url"
  echo "[done] $name $(du -h "$dst" | cut -f1)"
done

echo "finished: $(ls -1 "$OUT"/*.parquet 2>/dev/null | wc -l) shards in $OUT"
