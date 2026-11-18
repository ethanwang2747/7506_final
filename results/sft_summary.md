# SFT results summary

All adapters are LoRA (r=8, alpha=16, q_proj/v_proj) on SmolVLM2-2.2B with the
vision tower frozen. Training uses 2,954 GQA train images, 11,816 side questions
and 3,600 relation questions; evaluation uses the same 6 groups / 240 pairs /
480 examples as every other experiment. Full step-by-step log:
`docs/sft_experiment_log.md`.

## 1. Data and training

| item | value |
|---|---|
| train images | 2,954 |
| side questions | 11,816 (left/right 5,908, top/bottom 5,908) |
| relation questions | 3,600 (6 groups, 600 each, balanced) |
| LoRA | r=8, alpha=16, q_proj/v_proj, 2.57M trainable (0.11%) |
| batch | 1 with gradient accumulation 8, bf16 |
| loss | answer tokens + end-of-utterance only |
| peak memory | 5.55 GB |
| speed | ~0.21 s/step |

## 2. 2x2: which training task teaches what

| adapter | side horizontal | side vertical | relation acc | relation AUC | delta AUC vs base |
|---|---:|---:|---:|---:|---:|
| base | 0.506 | 0.492 | 0.569 | 0.600 | - |
| side-only | 0.645 | 0.682 | 0.617 | 0.717 | +0.117 [0.073, 0.160] |
| relation-only | 0.551 | 0.492 | 0.683 | 0.777 | +0.177 [0.124, 0.229] |
| mixed | 0.689 | 0.704 | 0.656 | 0.750 | +0.150 [0.102, 0.199] |
| full (1 epoch) | **0.714** | **0.802** | **0.721** | **0.799** | **+0.199 [0.145, 0.254]** |

Intervals: paired bootstrap by pair id, 2,000 draws.

- Side-only training transfers to relations: +0.117 AUC, CI excludes 0. The
  accuracy delta is +0.044 [-0.004, 0.094], so the gain is in ranking.
- Relation-only training leaves side reporting at base level (0.551/0.492):
  relation labels do not teach the readout.
- Direct relation supervision is strong (+0.177), and a full epoch of mixed
  training is stronger still (+0.199, accuracy +0.150 [0.098, 0.202]).

## 3. Stability across seeds

| seed | relation acc | relation AUC | delta AUC |
|---|---:|---:|---:|
| side-only seed 0 | 0.617 | 0.717 | +0.117 [0.073, 0.160] |
| side-only seed 1 | 0.594 | 0.705 | +0.105 [0.060, 0.150] |

The ranking gain is stable; the accuracy gain is not (both CIs include 0).

## 4. Cross-group generalization

Relation supervision for `chair|left of|table` and `tree|left of|building` was
removed; side data still covers all training images.

| group | type | base acc/AUC | cross-group acc/AUC | delta AUC |
|---|---|---:|---:|---:|
| chair left of table | held out | 0.562 / 0.723 | 0.713 / 0.772 | +0.049 |
| tree left of building | held out | 0.500 / 0.523 | 0.662 / 0.802 | +0.279 |
| man left of person | trained | 0.537 / 0.579 | 0.613 / 0.627 | +0.048 |
| man right of woman | trained | 0.675 / 0.661 | 0.775 / 0.813 | +0.152 |
| person right of man | trained | 0.575 / 0.581 | 0.575 / 0.629 | +0.048 |
| woman left of man | trained | 0.562 / 0.707 | 0.762 / 0.900 | +0.193 |

Overall 0.569/0.600 -> 0.683/0.759; delta AUC +0.159 [0.109, 0.208]. The
held-out average gain (+0.164) is at least as large as the trained groups'
(+0.110), so the learned readout is not group-specific memorization.

## 5. Composition check

| condition | base (diff-half) | mixed (diff-half) |
|---|---:|---:|
| true sides | 0.921 | 0.903 |
| swapped sides | 0.120 | 0.277 |

Composition does not degrade; the model still follows swapped statements.

## 5.1 Blank-image control

With a uniform gray image, both the base model and the full adapter score
0.500 accuracy / 0.500 AUC on the relation task. The improvement requires the
image and is not a language prior.

## 6. Caveats

- 4,000-step pilots (1/4 epoch), one seed per adapter except side-only.
- Eval groups overlap the training groups except in the cross-group run;
  image ids are always disjoint.
- No mirror or question-reversal augmentation yet; no position-residualized
  evaluation of the adapters yet.
- Adapters are not committed (regenerable from the logged commands).
