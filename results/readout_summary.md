# Readout gap experiments: results summary

This note records the first readout experiment round on frozen SmolVLM2-2.2B features. The goal is to test whether spatial relation information is present in the visual representation but not used by the model's own yes/no readout, and whether a lightweight reader can close the gap.

All numbers are from cached features and can be reproduced without re-running the backbone. The feature files (`results/readout/*.pt`) are not tracked in Git. A report-ready section is in [`report/readout_gap_section.md`](../report/readout_gap_section.md).

## Setup

- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, no image splitting, 81 visual tokens.
- Stages: S1 (connector output), S6 and S12 (text-layer inputs).
- Train: `spatial_pairs_dev_strict.jsonl` (1,206 pairs) plus the six core groups from `spatial_pairs_dev.jsonl`.
- Test: 40 pairs per core group from `spatial_pairs_test.jsonl` (480 examples), plus the same source images horizontally flipped.
- Strict test: the same six groups from `spatial_pairs_test_strict.jsonl`.
- The reader sees visual tokens and the mean question embedding (`E_mean`). It does not see bounding boxes unless an oracle probe is explicitly marked.

## 1. Global no-box readers

The same training set is used for all readers. Values are AUC over source/target examples; flip is original vs horizontally flipped source image.

| Reader | test pairs | flip |
|---|---:|---:|
| Original model margin | 0.600 | 0.559 |
| Position-only (global, bbox) | 0.568 | 0.521 |
| Final prompt token H12 (linear) | 0.490 | 0.522 |
| Question-only (linear, E_mean) | 0.500 | 0.500 |
| Mean-pooled S1/S6/S12 + question | 0.474 | 0.533 |
| Attention over S1/S6/S12 + question | 0.443 | 0.497 |
| Attention + training-time region loss (1 seed, 15 epochs) | 0.548 | 0.529 |

The global trained readers do not improve on the model's own margin. The question-only and final-token probes are at chance, so the readers are not exploiting a question or answer-token bias. The attention reader trained with a region loss improves over the unsupervised attention reader, but remains below the original model. A training-free flip-consistency readout does improve the practical readout substantially (section 2).

## 2. Training-free flip-consistency readout

For a left/right question, a relation-consistent model should give a higher yes margin on the image and a lower yes margin on its horizontal flip. The score is therefore `margin(original) - margin(flipped)`. It needs two forwards per image, but no training, no bounding boxes, and no labels.

| Split | Readout | AUC | 95% CI | Accuracy | Pair accuracy |
|---|---|---:|---|---:|---:|
| dev core | Original margin | 0.694 | [0.642, 0.745] | 0.628 | 0.760 |
| dev core | **Consistency** | **0.744** | **[0.678, 0.802]** | **0.653** | **0.785** |
| test core | Original margin | 0.600 | [0.558, 0.645] | 0.571 | 0.604 |
| test core | Flipped margin | 0.507 | [0.457, 0.553] | 0.479 | 0.504 |
| test core | **Consistency** | **0.690** | **[0.643, 0.736]** | **0.619** | **0.683** |
| test core | Sum of margins | 0.556 | [0.512, 0.601] | 0.525 | 0.567 |
| strict test | Original margin | 0.633 | [0.543, 0.727] | 0.614 | 0.632 |
| strict test | **Consistency** | **0.813** | **[0.735, 0.886]** | **0.728** | **0.807** |

Paired bootstrap over pairs:

- dev core: ΔAUC = +0.051 [−0.018, 0.116]; Δpair accuracy = +0.025. The direction is consistent, but this interval includes zero.
- test core: ΔAUC = **+0.091** [0.032, 0.149]; Δpair accuracy = +0.079.
- strict test: ΔAUC = **+0.180** [0.067, 0.286]; Δpair accuracy = +0.175.

Per-group test core:

| Group | Original | Consistency | Consistency 95% CI | Accuracy |
|---|---:|---:|---|---:|
| chair / left / table | 0.719 | 0.743 | [0.657, 0.831] | 0.662 |
| man / left / person | 0.583 | 0.554 | [0.424, 0.685] | 0.512 |
| man / right / woman | 0.658 | 0.768 | [0.685, 0.857] | 0.662 |
| person / right / man | 0.576 | 0.550 | [0.401, 0.690] | 0.537 |
| tree / left / building | 0.524 | 0.696 | [0.586, 0.801] | 0.600 |
| woman / left / man | 0.698 | 0.821 | [0.723, 0.901] | 0.738 |

Four of six groups improve; the two person/man groups remain near chance. The consistency score is symmetric under flipping by construction, so evaluating it on the flip split would be circular. The cross-image source/target test is the meaningful evaluation.

## 3. Oracle region readout

A per-group logistic reader pools target/reference object tokens at S1, S6, and S12 using GQA boxes. This is an **oracle diagnostic**: boxes are used at test time and this is not a deployable reader.

Aggregate mean of per-group AUCs (six groups):

| Split | Region | Region, position-residualized | Position-only | Original margin |
|---|---:|---:|---:|---:|
| test | 0.812 | 0.687 | 0.961 | 0.626 |
| test, flip-augmented training | 0.859 | 0.560 | 0.970 | 0.626 |
| strict test | 0.883 | 0.749 | 0.994 | 0.665 |
| flip | 0.829 | 0.705 | 0.950 | 0.584 |
| flip, flip-augmented training | 0.881 | 0.564 | 0.958 | 0.584 |
| strict flip | 0.936 | 0.765 | 0.988 | 0.701 |

Per-group test results:

| Group | Original | Region | Region 95% CI | Residual | Position |
|---|---:|---:|---|---:|---:|
| chair / left / table | 0.719 | 0.747 | [0.647, 0.842] | 0.722 | 0.853 |
| man / left / person | 0.583 | 0.743 | [0.633, 0.847] | 0.602 | 0.969 |
| man / right / woman | 0.658 | 0.891 | [0.811, 0.958] | 0.751 | 0.992 |
| person / right / man | 0.576 | 0.743 | [0.633, 0.847] | 0.602 | 0.969 |
| tree / left / building | 0.524 | 0.861 | [0.785, 0.928] | 0.695 | 0.993 |
| woman / left / man | 0.698 | 0.891 | [0.811, 0.958] | 0.751 | 0.992 |

Per-group flip results:

| Group | Original | Region | Region 95% CI | Residual | Position |
|---|---:|---:|---|---:|---:|
| chair / left / table | 0.634 | 0.731 | [0.651, 0.811] | 0.738 | 0.893 |
| man / left / person | 0.567 | 0.814 | [0.742, 0.891] | 0.731 | 0.939 |
| man / right / woman | 0.594 | 0.898 | [0.839, 0.949] | 0.673 | 0.968 |
| person / right / man | 0.551 | 0.814 | [0.742, 0.891] | 0.731 | 0.939 |
| tree / left / building | 0.498 | 0.819 | [0.734, 0.898] | 0.685 | 0.991 |
| woman / left / man | 0.662 | 0.898 | [0.839, 0.949] | 0.674 | 0.968 |

## 4. Controls

- **Random labels.** Permuting dev labels gives mean AUC 0.496 for the region probe and 0.504 for the position probe, so the region/position numbers are not a label-leak artifact.
- **Position residualization.** Regressing the region features on the 12-dimensional box description and training on the residual lowers the region probe to 0.687 (test) and 0.705 (flip). The remaining signal is above chance but much smaller than the raw region probe.
- **Flip augmentation.** Training with horizontal flips increases the raw region probe (0.812 to 0.859 on test; 0.829 to 0.881 on flips) but the position-residualized version falls to 0.560/0.564. The augmented reader is therefore using position more strongly; the residual estimate is sensitive to this augmentation.
- **Question-only baseline.** A linear probe on `E_mean` is exactly 0.500 on both splits because the question vector is identical for the two members of a pair.

## 5. Interpretation

The results support a **localization/position-readout gap**, not a claim that visual relation information is rich but linearly unreadable:

1. The original model's margin is weak (0.63 test, 0.58 flip).
2. A per-group oracle readout over target/reference regions reaches 0.81 test and 0.83 flip, so the frozen features contain information that a simple reader can use.
3. A position-only oracle is even stronger (0.96 test, 0.95 flip). The paired benchmark is therefore largely solvable from object positions alone; any "relation readout" claim must control this shortcut.
4. After position residualization, the region probe retains only 0.69 test / 0.71 flip. This is evidence for some position-independent signal, but not for a strong, uniformly decodable relation code.
5. A training-free flip-consistency readout improves AUC from 0.600 to 0.690 on test and from 0.633 to 0.813 on strict test. This shows that the model has usable relation evidence, but it is not calibrated into a single-image margin; comparing the image with its mirror removes part of the image-level bias.
6. No-box trained readers over all 81 visual tokens do not learn to localize; mean-pooling and global attention are at or below chance. Region supervision during training helps only partially. The bottleneck is finding the right object tokens, not the linear classifier on top.
7. The same-image horizontal flip is **not** a clean position control: a position-only reader reaches 0.95 on it because flipping moves objects to the other side of the image. Position-residualized or position-matched pairs are required for a strong relation claim.

## 6. Limitations and next steps

- Oracle boxes are used at test time in the region probe; the result is an upper bound, not a deployable VLM.
- Test pairs come from GQA val with noisy relation labels; the strict subset is smaller.
- Per-group training sets are small (14–52 pairs), so per-group CIs are wide.
- Flip augmentation changes the residualized estimate; a position-matched evaluation set is the cleaner next control.
- The flip-consistency readout is practical but needs two forwards per image and does not localize objects.
- A stronger reader would need a learned object proposal/localization stage. The current no-box attention reader is a negative result for that step.

## Reproduction

```bash
export SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct
export PYTHONPATH=.

# Cache features (already done locally)
python experiments/extract_readout_features.py --pairs data/derived/spatial_pairs_dev_strict.jsonl --out results/readout/features_dev_strict.pt
python experiments/extract_readout_features.py --pairs data/derived/spatial_pairs_dev_strict.jsonl --mode flip --only-flipped --out results/readout/features_dev_strict_flip.pt
python experiments/extract_readout_features.py --pairs data/derived/spatial_pairs_test.jsonl --groups "woman|to the left of|man,man|to the right of|woman,man|to the left of|person,person|to the right of|man,tree|to the left of|building,chair|to the left of|table" --max-per-group 40 --out results/readout/features_test_core.pt
python experiments/extract_readout_features.py --pairs data/derived/spatial_pairs_test.jsonl --groups "woman|to the left of|man,man|to the right of|woman,man|to the left of|person,person|to the right of|man,tree|to the left of|building,chair|to the left of|table" --max-per-group 40 --mode flip --out results/readout/features_test_core_flip.pt

# Global readers
python experiments/readout_experiment.py --train-pairs results/readout/features_dev_strict.pt results/readout/features_dev_core.pt --train-flip results/readout/features_dev_strict_flip.pt --test-pairs results/readout/features_test_core.pt --test-flip results/readout/features_test_core_flip.pt --out results/readout/main.json --seeds 2 --epochs 40 --patience 8 --batch-size 64 --bootstrap 300

# Oracle region probe
python experiments/region_readout.py --out results/readout/region_probe.json --bootstrap 500

# Training-free flip-consistency readout
python experiments/flip_consistency.py \
  --pairs results/readout/features_test_core.pt \
  --flip-source results/readout/features_test_core_flip.pt \
  --flip-target results/readout/features_test_core_flip_target.pt \
  --out results/readout/flip_consistency_test.json --bootstrap 1000
python experiments/flip_consistency.py \
  --pairs results/readout/features_test_core_strict.pt \
  --flip-source results/readout/features_test_core_strict_flip.pt \
  --flip-target results/readout/features_test_core_strict_flip_target.pt \
  --out results/readout/flip_consistency_strict.json --bootstrap 1000
```

Raw outputs: `results/readout/main.json`, `results/readout/region_probe.json`, `results/readout/flip_consistency_test.json`, `results/readout/flip_consistency_strict.json`, `results/readout/aux_smoke.json`.
