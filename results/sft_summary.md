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

| adapter / seed | relation acc | relation AUC | delta AUC |
|---|---:|---:|---:|
| side-only seed 0 | 0.617 | 0.717 | +0.117 [0.073, 0.160] |
| side-only seed 1 | 0.594 | 0.705 | +0.105 [0.060, 0.150] |
| full seed 0 | 0.721 | 0.799 | +0.199 [0.145, 0.254] |
| full seed 1 | 0.665 | 0.781 | +0.181 [0.126, 0.238] |

The ranking gain is stable; the side-only accuracy gain is not (both CIs include 0).
On the confirmation set the two full seeds are nearly identical (0.802 / 0.804).

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

Caveat: most of the held-out gain is already carried by side-only training
(side-only gains on the same groups are +0.036 and +0.260; the extra gain from
the remaining groups' relation supervision is only +0.014 and +0.019). This run
therefore supports "side supervision transfers across groups", not "relation
supervision generalizes across groups". The full-vs-4k comparison is also
confounded by training steps.

## 4.1 Fresh confirmation set

Built from 71,302 GQA train images that appear in no pair file and no SFT batch
(image-disjoint from every prior evaluation; the base model may have seen them
during pretraining). 240 pairs / 480 examples, 40 positive and 40 negative per
group.

| metric | base | full adapter |
|---|---:|---:|
| relation accuracy | 0.510 | **0.700** |
| relation AUC | 0.531 | **0.802** |
| flip-consistency AUC | 0.750 | **0.821** |
| flip-consistency pair accuracy | 0.750 | 0.787 |

The gain replicates on unseen images: delta AUC +0.271 [0.215, 0.328], delta
accuracy +0.192 [0.140, 0.246] (paired bootstrap by pair). The adapter's
confirmation AUC (0.802) is essentially identical to the original test (0.799).
The base number on the original test (0.600) was optimistic; the adapter number
was not.

## 4.2 Regression checks

POPE (300 questions): accuracy 0.783 -> **0.850**, AUC 0.890 -> **0.914**,
F1 0.732 -> **0.852**.

Fine-grained (600 pairs): overall 0.832 -> 0.823, attribute 0.765 -> **0.805**,
behavior 0.865 -> 0.820, existence 0.865 -> 0.845; gray-image control 0.807 ->
0.802. No catastrophic regression.

TIFA (100 images / 466 facts) exposes a trade-off:

| setting | cued recall | judge coverage | mean words |
|---|---:|---:|---:|
| base, default prompt | 0.753 | 0.788 | 39.6 |
| full, default prompt | **0.867** | 0.588 | 12.2 |
| full, asked for four sentences | **0.867** | **0.702** | 36.3 |

The adapter follows the "one or two sentences" instruction more closely, so the
coverage drop is partly a length/style effect; at matched length coverage is
0.702 vs base 0.788. Cued recall improves, so this is not a knowledge regression.

Logit lens (core 80 pairs): the readable signal moves earlier and strengthens —
base L18/L19/final = 0.636/0.646/0.634; full = 0.831/0.847/0.864.

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

## 5.2 Flip consistency stacks with SFT

| metric | base | full adapter |
|---|---:|---:|
| margin AUC | 0.600 | 0.799 |
| flip-consistency AUC | 0.687 | **0.811** |
| margin pair accuracy | 0.596 | 0.783 |
| flip pair accuracy | 0.663 | **0.817** |

The two interventions are complementary, but SFT captures most of the flip gain
(base +0.087 AUC, full +0.012); pair accuracy still gains +0.034.

## 5.3 Question-reversal robustness and augmentation

Reversed questions ("Is Y right of X?" for "Is X left of Y?") test whether the
adapter learned the relation or a lexical pattern.

| adapter | original AUC | reversed AUC | delta |
|---|---:|---:|---:|
| base | 0.600 | 0.575 | -0.025 |
| mixed 4k | 0.750 | 0.737 | -0.013 |
| mixed 4k + reversal/mirror augmentation | 0.737 | 0.734 | **-0.003** |
| full 1 epoch | 0.799 | 0.770 | -0.029 |

The standard adapter is already robust to reversal. Augmentation at 4k steps
tightens the reversal gap but slightly lowers original accuracy, so it is a wash.

## 6. Caveats

- The 4,000-step pilots are 1/4 epoch; one seed per adapter except side-only.
  A second full-epoch seed is running.
- The original test set has been used for many comparisons; treat it as a
  development benchmark. The fresh confirmation set is the clean held-out check.
- Confirmation images are train-derived (image-disjoint from SFT, but possibly
  seen by the base model in pretraining).
- No mirror or question-reversal augmentation yet; no position-residualized
  evaluation of the adapters. Position residualization is not a success gate:
  the label is defined by object positions, so geometry is legitimate evidence.
- Adapters are not committed (regenerable from the logged commands).
