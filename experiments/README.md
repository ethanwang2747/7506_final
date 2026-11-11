# Experiment pipeline

All commands run from the project root with the project environment:

```bash
cd /path/to/7506project
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## Data

The GQA val scene graphs, val questions, and val images are in `data/raw/gqa/`. The testdev images cannot be used for regional analysis because the released scene graphs do not cover them.

## Audit GQA pairing confound

```bash
python experiments/audit_gqa_pairs.py
```

This writes `results/audit/gqa_pairing_audit.json` with the object-presence counts and scene-graph coverage.

## Build pairs

```bash
python experiments/build_val_pairs.py --pairs-per-group 40
```

This writes `data/derived/spatial_pairs_{dev,test}.jsonl`, `data/derived/semantic_pairs_{dev,test}.jsonl`, and `data/derived/manifest.json`. The image-level split is `sha1(image_id) % 5 == 0 -> dev`.

## E0: interface and identity-patch validation

```bash
python experiments/e0_validate.py --task spatial --group "woman|to the left of|man" --split dev --splitting 0
python experiments/e0_validate.py --task semantic --group "fence|material|wood" --split dev --splitting 0
```

## E1: clean behavior and input dependence

```bash
python experiments/e1_behavior_map.py --task spatial --split dev --pairs-per-group 20 --splitting 0
python experiments/e1_behavior_map.py --task spatial --split test --pairs-per-group 20 --splitting 0
python experiments/e1_behavior_map.py --task semantic --split dev --pairs-per-group 20 --splitting 0
python experiments/e1_behavior_map.py --task semantic --split test --pairs-per-group 20 --splitting 0
```

## Mean vectors

```bash
python experiments/estimate_means.py
```

This writes `data/derived/mean_vectors.pt`, estimated from clean dev source forwards.

## E2: stage-wise regional mean replacement

```bash
python experiments/e2_ablation.py --task spatial --split dev --pairs-per-group 10
python experiments/e2_ablation.py --task spatial --split test --pairs-per-group 20
python experiments/e2_ablation.py --task semantic --split dev --pairs-per-group 10
python experiments/e2_ablation.py --task semantic --split test --pairs-per-group 20
python experiments/e2_bilateral.py
python experiments/e2_splitting.py --task spatial --split test --groups "woman|to the left of|man,man|to the right of|woman,tree|to the left of|building,chair|to the left of|table" --pairs-per-group 5
python experiments/e4_crop_cumulative.py
```

## E3: activation patching

Full visual block:

```bash
python experiments/e3_patching.py --task spatial --split dev --pairs-per-group 10
python experiments/e3_patching.py --task spatial --split test --pairs-per-group 20
python experiments/e3_patching.py --task semantic --split dev --pairs-per-group 10
python experiments/e3_patching.py --task semantic --split test --pairs-per-group 20
```

Exploratory regional patching:

```bash
python experiments/e3_regional.py --split test --pairs-per-group 20
python experiments/e3_text_position_patch.py
```

Splitting-resolution checks:

```bash
python experiments/estimate_means.py --splitting 1
python experiments/e2_splitting.py --task spatial --split test --groups "woman|to the left of|man,man|to the right of|woman,tree|to the left of|building,chair|to the left of|table" --pairs-per-group 5
python experiments/e3_splitting.py --split test --groups "woman|to the left of|man,man|to the right of|woman,tree|to the left of|building,chair|to the left of|table" --pairs-per-group 5
```

Coverage ablation (single token / small region / full object / full block):

```bash
python experiments/e2_coverage.py --task spatial --group "chair|to the left of|table" --split test --pairs-per-group 10
python experiments/e2_coverage.py --task semantic --group "fence|material|wood" --split test --pairs-per-group 10
```

## E4: alternative explanations

```bash
python experiments/e4_probe.py
python experiments/e4_region_probe.py
python experiments/e4_stage_probe.py
python experiments/e4_question_context.py
python experiments/e4_joint.py
python experiments/flip_counterfactual.py
python experiments/make_representation_figure.py
```

## Statistics and tables

```bash
python experiments/stats.py --e1 results/e1/spatial_test_split0.json --out results/stats/spatial_test_e1_summary.json
python experiments/stats.py --e2 results/e2/spatial_test.json --e3 results/e3/spatial_test_full.json --out results/stats/spatial_test_summary.json
python experiments/stats.py --e2 results/e2/semantic_test.json --e3 results/e3/semantic_test_full.json --out results/stats/semantic_test_summary.json
python experiments/report_tables.py
```

`results/summary.md` contains the combined tables. The report draft is `report/report_v5_draft.md`.
