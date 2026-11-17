# Experiment pipeline

All commands run from the project root with the project environment:

```bash
cd /path/to/7506project
source .venv/bin/activate
export PYTHONPATH=.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Optional: local model directory
# export SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct
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

## E5: readout-gap experiments

Cache frozen features once, then train readers without re-running the backbone. The feature files are large and are ignored by Git.

```bash
export SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct
export PYTHONPATH=.

python experiments/extract_readout_features.py \
  --pairs data/derived/spatial_pairs_dev_strict.jsonl \
  --out results/readout/features_dev_strict.pt
python experiments/extract_readout_features.py \
  --pairs data/derived/spatial_pairs_dev_strict.jsonl --mode flip --only-flipped \
  --out results/readout/features_dev_strict_flip.pt
python experiments/extract_readout_features.py \
  --pairs data/derived/spatial_pairs_test.jsonl \
  --groups "woman|to the left of|man,man|to the right of|woman,man|to the left of|person,person|to the right of|man,tree|to the left of|building,chair|to the left of|table" \
  --max-per-group 40 --out results/readout/features_test_core.pt
python experiments/extract_readout_features.py \
  --pairs data/derived/spatial_pairs_test.jsonl \
  --groups "woman|to the left of|man,man|to the right of|woman,man|to the left of|person,person|to the right of|man,tree|to the left of|building,chair|to the left of|table" \
  --max-per-group 40 --mode flip --out results/readout/features_test_core_flip.pt

python experiments/readout_experiment.py \
  --train-pairs results/readout/features_dev_strict.pt results/readout/features_dev_core.pt \
  --train-flip results/readout/features_dev_strict_flip.pt \
  --test-pairs results/readout/features_test_core.pt \
  --test-flip results/readout/features_test_core_flip.pt \
  --out results/readout/main.json --seeds 2 --epochs 40 --patience 8 \
  --batch-size 64 --bootstrap 300

python experiments/region_readout.py \
  --out results/readout/region_probe.json --bootstrap 500
```

The interpretation and tables are in `results/readout_summary.md`.

### Flip-consistency readout

A training-free improvement: compare the yes/no margin on each image with the margin on its horizontal flip. No boxes and no training; two forwards per image.

```bash
# For target images, extract their flipped features as well
python experiments/extract_readout_features.py \
  --pairs data/derived/spatial_pairs_test.jsonl \
  --groups "woman|to the left of|man,man|to the right of|woman,man|to the left of|person,person|to the right of|man,tree|to the left of|building,chair|to the left of|table" \
  --max-per-group 40 --mode flip --flip-role target --only-flipped \
  --out results/readout/features_test_core_flip_target.pt

python experiments/flip_consistency.py \
  --pairs results/readout/features_test_core.pt \
  --flip-source results/readout/features_test_core_flip.pt \
  --flip-target results/readout/features_test_core_flip_target.pt \
  --out results/readout/flip_consistency_test.json --bootstrap 1000
```

The strict-test variant follows the same pattern with the `*_strict` payloads.

## E6: same-image fine-grained discrimination

Build same-image positive/negative pairs for existence, attribute, and behavior questions. Both questions use the same image, so image content and layout are fixed; only the question changes.

```bash
python experiments/build_finegrained_pairs.py --max-per-task 200 --per-image 4
python experiments/finegrained_experiment.py \
  --pairs data/derived/finegrained_pairs_test.jsonl \
  --out results/finegrained/test.json
python experiments/finegrained_blank_control.py \
  --results results/finegrained/test.json \
  --out results/finegrained/test_blank.json

# Clear predicate-swap behavior pilot
python experiments/finegrained_experiment.py \
  --pairs data/derived/finegrained_predicate_test.jsonl \
  --out results/finegrained/predicate_test.json
python experiments/finegrained_blank_control.py \
  --results results/finegrained/predicate_test.json \
  --out results/finegrained/predicate_test_blank.json
```

The interpretation is in `results/finegrained_summary.md`.

## E7: description vs cued yes/no

Compare the original yes/no margin with a one-sentence spatial description on the same images.

```bash
python experiments/describe_spatial.py \
  --pairs data/derived/spatial_pairs_test.jsonl \
  --pairs-per-group 10 \
  --out results/description/spatial_test.json
```

The results and a table of established description/hallucination/action benchmarks are in `results/description_summary.md`.

## E8: POPE external validation

POPE is the standard object-existence / hallucination benchmark. This uses a
300-question stratified sample (100 per split) and 229 COCO val2014 images.

```bash
python experiments/pope_experiment.py \
  --sample data/raw/pope/sample_300.jsonl \
  --out results/pope/pope_300.json
```

The summary is in `results/pope_summary.md`.

## E9: TIFA/COCO description coverage

Generate free descriptions and compare cued TIFA fact recall with whether the
description covers the same fact.

```bash
python experiments/tifa_experiment.py \
  --sample data/raw/tifa/sample_100.jsonl \
  --out results/tifa/tifa_100.json
python experiments/tifa_judge.py \
  --results results/tifa/tifa_100.json \
  --out results/tifa/tifa_100_judge.json
```

The summary is in `results/tifa_summary.md`.

## E10: TempCompass video pilot

Small video event/direction/order evaluation with a black-video control.

```bash
python experiments/tempcompass_experiment.py \
  --out results/tempcompass/tempcompass_30.json
python experiments/tempcompass_judge.py \
  --results results/tempcompass/tempcompass_30.json \
  --out results/tempcompass/tempcompass_30_judge.json
```

The summary is in `results/tempcompass_summary.md`.

## E11: Something-Something V2 action pilot

Small real-action discrimination test with a black-video control.

```bash
python experiments/ssv2_experiment.py \
  --sample data/raw/ssv2/sample_30.tsv \
  --out results/ssv2/ssv2_30.json
```

The summary is in `results/ssv2_summary.md`; the overall external synthesis is
in `results/external_benchmarks_summary.md`.

### Improvement checks

- POPE flip-combination/calibration: `python experiments/pope_calibration.py`
- TIFA structured prompt variant:
  `python experiments/tifa_experiment.py --sample data/raw/tifa/sample_100.jsonl --out results/tifa/tifa_100_struct.json --max-new-tokens 64 --prompt "Describe the image in detail. Mention the main objects, their colors and materials, what the people are doing, and the spatial relationships between the main objects."`
  then judge with `tifa_judge.py`.

## E12: attention audit

Extract text-to-image attention on spatial dev pairs and test whether any head
selectively attends to target/reference tokens.

```bash
SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct PYTHONPATH=. \
python experiments/attention_audit.py \
  --out results/attention/audit_dev_last.json
```

The summary is in `results/attention_summary.md`.

### Attention head causal test

```bash
SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct PYTHONPATH=. \
python experiments/attention_head_intervention.py \
  --out results/attention/head_intervention_dev.json
```

The result is in `results/attention_causal_summary.md`.
