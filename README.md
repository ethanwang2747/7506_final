# From Visual Evidence to Decisions

Stage-wise causal tracing of spatial relations in a vision-language model.

DASE7506 Advanced Machine Learning Methods final project, The University of Hong Kong.

## Overview

This project studies how `HuggingFaceTB/SmolVLM2-2.2B-Instruct` turns visual evidence into yes/no decisions about object relations. The pipeline builds canonical GQA-derived source/target pairs, measures clean behavior and input dependence, and applies stage-wise regional mean replacement and activation patching at five stages of the vision-to-language path. Supporting experiments cover semantic relations, image splitting, question-context binding, coverage, and a same-image flip counterfactual.

- Report draft: [`report/report_v5_draft.md`](report/report_v5_draft.md)
- Combined result tables: [`results/summary.md`](results/summary.md)
- Experiment dashboard: [`results/dashboard.md`](results/dashboard.md)
- Readout-gap experiments: [`results/readout_summary.md`](results/readout_summary.md) and report section [`report/readout_gap_section.md`](report/readout_gap_section.md) (training-free flip-consistency readout: test AUC 0.600 → 0.690; strict test 0.633 → 0.813)
- Fine-grained extension: [`results/finegrained_summary.md`](results/finegrained_summary.md) (same-image existence / attribute / behavior pairs, with blank-image and flip controls)
- Description vs yes/no pilot: [`results/description_summary.md`](results/description_summary.md) (targeted spatial description; also lists established benchmarks to reuse)
- POPE external validation: [`results/pope_summary.md`](results/pope_summary.md) (300 questions; accuracy 0.783, AUC 0.890, conservative yes rate 0.33)
- Full experiment commands: [`experiments/README.md`](experiments/README.md)

## Repository layout

| Path | Contents |
|---|---|
| `experiments/` | Pair construction, E0 validation, E1 behavior map, E2 ablation, E3 patching, E4 alternative mechanisms, readout probes, statistics, figures |
| `prepare_fixed_subsets.py` | Builds the fixed GQA / TextVQA / CV-Bench subset annotations and image files |
| `data/fixed/` | Fixed subset annotations (`*.jsonl`, `manifest.json`); extracted images are not tracked |
| `data/derived/` | Generated pairs and mean vectors; not tracked except `manifest.json` and `experiment_config.json` |
| `results/` | Experiment JSONs, summary tables, and figures |
| `logs/` | Stdout logs from experiment runs |
| `report/` | Report draft |
| `project_proposal_v*.md` | Proposal and design notes |

## Environment

Tested with Python 3.12 and one CUDA GPU.

```bash
python -m venv .venv
source .venv/bin/activate

# Install the PyTorch build matching your CUDA version first, then:
pip install torch==2.11.0 torchvision==0.26.0
pip install transformers==5.18.0 accelerate==1.15.0 datasets==5.1.0 \
    pyarrow==26.0.1 pandas==3.1.1 numpy==2.5.4 \
    pillow==10.2.0 scikit-learn==1.9.1 matplotlib==3.11.2
```

Model weights are downloaded from the Hugging Face Hub on first use. Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` only after the model is cached.

If the model is stored locally (for example a ModelScope download under `models/`), point the pipeline at it:

```bash
export SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct
```

`experiments/common.py` reads `SMOLVLM_MODEL`; `models/` is ignored by Git.

> `requirement.txt` is the course project brief, not a Python dependency file.

## Data not tracked in Git

The raw downloads (about 3 GB), extracted images (about 390 MB), and generated intermediates are excluded. They are reproducible from these public sources:

| Dataset | Hugging Face repo | Revision |
|---|---|---|
| GQA | `lmms-lab-encoder/GQA` | `a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8` |
| TextVQA | `lmms-lab-encoder/textvqa` | `9c0699cd19768ac5ab97568f6b3cbac4c0062884` |
| CV-Bench 2D | `nyu-visionx/CV-Bench` | `bc284db50d036958861cb60cdd7b77612052ce0d` |

Place the downloaded files under `data/raw/` with the layout expected by `prepare_fixed_subsets.py`:

```text
data/raw/gqa/
  val_sceneGraphs.json
  train_sceneGraphs.json
  val_balanced_instructions/val-00000-of-00001.parquet
  val_balanced_images/val-0000{0,1,2}-of-00003.parquet
  testdev_balanced_instructions/testdev-00000-of-00001.parquet
  testdev_balanced_images/testdev-00000-of-00001.parquet
  testdev_all_instructions/testdev-00000-of-00001.parquet
data/raw/textvqa/data/validation-0000{0,1,2}-of-00003.parquet
data/raw/cvbench/test_2d.parquet
```

Then build the fixed subsets:

```bash
source .venv/bin/activate
python prepare_fixed_subsets.py
```

This writes `data/fixed/*.jsonl`, `data/fixed/manifest.json`, and `data/fixed/images/`. The image subsets are required by all model experiments.

## Reproducing the experiments

The full sequence is in [`experiments/README.md`](experiments/README.md). The main path is:

```bash
python experiments/build_val_pairs.py --pairs-per-group 40
python experiments/estimate_means.py
python experiments/e0_validate.py --task spatial --split dev --splitting 0
python experiments/e1_behavior_map.py --task spatial --split test --pairs-per-group 20 --splitting 0
python experiments/e2_ablation.py --task spatial --split test --pairs-per-group 20
python experiments/e3_patching.py --task spatial --split test --pairs-per-group 20
python experiments/report_tables.py
```

All scripts resolve paths relative to the repository root, so the checkout can live anywhere.
