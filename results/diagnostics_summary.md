# Diagnostic summary: where the relation answer is lost

This note reports the diagnostic chain that follows `docs/diagnostic_handbook.md`. It is
not an improvement experiment. The goal is to locate the bottleneck before touching the
model, using probes, the logit lens, and causal region interventions on the same frozen
SmolVLM2-2.2B and the same GQA same-image pairs.

Data: `data/derived/spatial_pairs_dev_strict.jsonl` (2412 examples, 494 images, 1206 pairs)
for all probes; `spatial_pairs_test*.jsonl` for the readout references. Boxes come from GQA
scene graphs. Every probe is 5-fold cross-validated grouped by image id, so no image is
shared between train and test, and every number is out-of-fold.

## 1. The answer is almost fully determined by object geometry

A per-group logistic probe on the 12-dimensional box description (centres, sizes,
differences) reaches **AUC 0.991** on the 480 test-core examples. The task is therefore
nearly deterministic from the two boxes: `chair left of table` is answered by the sign of
`dx`.

The model's own yes/no margin on the same examples reaches **AUC 0.600**. More importantly,
the model's margin and the geometry rule are almost unrelated:

| quantity | value |
|---|---:|
| per-group position-only AUC | 0.991 |
| model margin AUC | 0.600 |
| Spearman(position score, model margin) | 0.183 |
| sign agreement | 0.562 |

This is the core "sees but does not answer" fact in this benchmark: the evidence that
decides the label is trivially decodable from the two objects, and the model's decision
barely tracks it.

## 2. Object identity is present in the visual tokens

`experiments/per_token_probe_map.py` trains one probe per cell of the 9x9 visual grid
(figure: `results/figures/per_token_probe_dev.png`).

**Name pointing.** Train a 20-way name probe on object cells (top-20 names cover 78% of
mentioned objects), then ask for each test example whether the target-name score and the
reference-name score peak inside the corresponding box.

| stage | top-1 hit (both objects) | both hit | chance |
|---|---:|---:|---:|
| S1 connector | 0.540 [0.496, 0.589] | 0.317 | 0.205 |
| S6 | 0.595 [0.560, 0.633] | 0.358 | 0.205 |
| S12 | 0.619 [0.580, 0.658] | 0.398 | 0.205 |

Object identity is locally decodable and improves with depth. The model therefore has the
ingredients needed to find the two objects.

**Queried-object membership.** A per-cell probe for "this token belongs to one of the two
queried objects" is much weaker (AUC 0.582 / 0.584 / 0.587; argmax hit 0.506 / 0.463 / 0.451
against chance 0.390). Knowing *what* a patch looks like is easier than knowing that it is
one of the two objects the question is about.

## 3. No single token answers the relation

For each relation type separately (question fixed within the type), a per-cell probe is
trained to predict whether the image satisfies the relation.

| target | mean cell acc (S1/S6/S12) | best cell | mean-pooled tokens |
|---|---|---:|---|
| left of (n=1052) | 0.494 / 0.496 / 0.495 | 0.524-0.541 | 0.488 / 0.484 / 0.479 |
| right of (n=1052) | 0.494 / 0.496 / 0.494 | 0.524-0.541 | 0.488 / 0.484 / 0.478 |
| above (n=154) | 0.494 / 0.495 / 0.490 | 0.591-0.597 | 0.494 / 0.474 / 0.513 |
| below (n=154) | 0.495 / 0.495 / 0.491 | 0.591-0.597 | 0.494 / 0.474 / 0.513 |
| model correctness (n=2412) | 0.544 / 0.542 / 0.542 | 0.583-0.604 | 0.588 / 0.582 / 0.586 |

The relation decision is not linearly readable from any single token, nor from the mean of
all tokens. Correctness is only slightly above chance and no better than its pooled
baseline. Local object evidence exists (section 2), but the relation is not assembled at the
token level.

A matched control makes the same point without the 9x9 grid: under the same grouped CV, a
global logistic reader on `[mean all tokens + question]` scores AUC 0.474, and a reader on
`[pooled target + pooled reference + difference + question]` scores AUC 0.492
(`results/diagnostics/region_pool_reference.json`). Pooled content does not carry the
relation in a group-general way.

Caveat: the earlier per-group oracle-region probe reaches 0.81 because a per-group reader
also learns that group's rule and reads geometry off the region features; it is not evidence
that pooled region content contains a group-general relation code. The honest upper bound
for "information available to a simple reader" is the geometry probe in section 1, not 0.81.

## 4. The decision appears only in the last layers

`experiments/logit_lens.py` applies the model's own final norm and LM head to the last-token
hidden state after each text layer. On the 80-pair test-core set:

| layer | 12 | 14 | 16 | 17 | 18 | 19 | 20 | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AUC | 0.507 | 0.493 | 0.557 | 0.567 | 0.636 | **0.646** | 0.621 | 0.634 |

Through layer 14 the lens is at chance; the signal appears at layer 18 and peaks at 19 — in
the last quarter of the 24-layer stack. Intermediate lens margins also carry a large
positive yes-bias, so only AUC is interpretable (the accuracy column is at 0.50 because the
threshold is off; the final margin has its own bias).

On a 240-pair strict sample the same profile holds with a lower ceiling (AUC 0.50-0.55 in
layers 1-14, 0.547 at layer 22, final 0.548).

## 5. The model uses the target region, but only partly

From the existing same-image flip counterfactual runs (`results/flip/*.json`, 376 unique
pairs, 17 groups), aggregated by `experiments/aggregate_region_patching.py`.

Baseline: the model ranks the original image above its mirror in 58.8% of pairs (left
67.5%, right 58.8%, above 53.8%, **below 35.7%**).

**Ablation** (replace a region's tokens by the layer mean; mean margin change, negative
means the region supports "yes"):

| stage | target | reference | irrelevant | background |
|---|---:|---:|---:|---:|
| S1 | -0.382 [-0.465, -0.306] | -0.141 [-0.200, -0.079] | +0.041 [-0.001, 0.081] | +0.033 [-0.002, 0.068] |
| S6 | -0.316 [-0.395, -0.243] | -0.066 [-0.124, -0.004] | +0.076 [0.038, 0.113] | +0.104 [0.073, 0.135] |

**Patching** (copy a region from the original image into the mirrored one; fraction of the
flip gap recovered, median):

| stage | target | reference | irrelevant | random |
|---|---:|---:|---:|---:|
| S1 | 0.333 [0.252, 0.464] | 0.201 [0.129, 0.256] | 0.072 [0.007, 0.126] | 0.003 [0.001, 0.014] |
| S6 | 0.331 [0.250, 0.471] | 0.200 [0.083, 0.250] | 0.071 [0.005, 0.143] | 0.002 [0.000, 0.012] |

The target region is causally used and the effect is specific (controls near zero), but only
about one third of the flip gap is carried by the region tokens; the rest is carried
elsewhere (global context, position, language prior).

## 6. Synthesis

```
pixels -> visual tokens        object identity decodable (name pointing 0.54-0.62)
       -> region selection      target region causally used (ablation -0.38, patch R 0.33)
       -> composition           no single token answers; pooled tokens at chance
       -> late decision         relation signal only readable at layers 18-19
       -> weak answer           margin AUC 0.60; rho with geometry 0.18
```

- It is not an encoding failure: object identity is present and localizable.
- It is not an attention-routing failure: attention mass on target/reference is below
  background, head ablation/amplification has no effect, and latent steering is null
  (`results/attention_summary.md`, `results/attention_causal_summary.md`,
  `results/latent_steering_summary.md`).
- It is a **late composition and readout failure**: the model has local object evidence but
  does not turn object positions into a relation decision until the final layers, and the
  result is only weakly aligned with the geometry that defines the answer.

## 7. What this implies for an improvement

Any intervention should be judged by whether it moves the late decision, not by whether it
changes attention maps. The diagnosis suggests three directions, in order of cost:

1. **Decomposition prompting (training-free).** Ask for each object's side separately
   ("Is the chair on the left half or the right half?"), then either combine the two answers
   with the relation rule or feed them back into the relation question. This tests whether
   the model can report positions when asked directly, which is the missing composition step.
2. **Question-side reversal consistency (training-free).** Ask both "Is X left of Y?" and
   "Is Y left of X?"; inconsistent answer pairs flag a failed composition. This is the
   question-space analogue of the image-flip consistency that already gives the only
   deployable gain.
3. **Explicit region supervision (light training).** Train a small head that pools two
   proposal regions and compares them, with boxes only at train time. The earlier
   mask-BCE reader failed with ~1.8k pairs; this diagnosis says the target is right but the
   supervision was too weak.

## 8. Reproduction

```bash
python experiments/per_token_probe_map.py            # per-token maps, ~25 min CPU
python experiments/logit_lens.py --max-pairs 240 --batch-size 2     --out results/diagnostics/logit_lens_test.json   # ~1 min GPU
python experiments/logit_lens.py --pairs data/derived/spatial_pairs_test.jsonl     --groups "chair|to the left of|table,woman|to the left of|man"     --max-per-group 40 --batch-size 2     --out results/diagnostics/logit_lens_core.json
python experiments/aggregate_region_patching.py      # existing flip runs
python experiments/region_pool_reference.py          # matched pooled readers
python experiments/geometry_vs_model.py              # position rule vs margin
```

## 9. Caveats

- Probes are correlational. The name and membership probes use boxes at train time; the
  causal claims come only from the ablation/patching table.
- The per-token answer probes contain no strong question vector. The matched global
  region-pool reader is the fair control, and it is also at chance; the supportable claim is
  "not linearly readable by a simple global reader", not "not representable at all".
- The lens readout is biased at intermediate layers; use AUC, not the sign.
- `dev_strict` is the hard subset (model accuracy 0.573); absolute levels are lower than on
  the full dev set. All comparisons here are within-set.


---

## 10. Stage 0/1 update (2026-11-18): the bottleneck is position readout

Four new measurements on the same model and test-core pairs refine the diagnosis.

### 10.1 Where is an object's side decodable?

`experiments/extract_side_features.py` + `side_probe.py`: last-token states at layers
12/16/18/19/20/23, 480 examples, GroupKFold by image.

| probe | L12 | L16 | L18 | L19 | L20 | L23 |
|---|---:|---:|---:|---:|---:|---:|
| target left half (raw) | 0.523 | 0.594 | 0.622 | 0.653 | 0.647 | 0.633 |
| reference left half (raw) | 0.487 | 0.553 | 0.566 | 0.544 | 0.555 | 0.519 |
| target top half (raw) | 0.768 | 0.805 | 0.795 | 0.792 | 0.789 | 0.773 |
| reference top half (raw) | 0.767 | 0.791 | 0.810 | 0.782 | 0.790 | 0.793 |
| answer | 0.489 | 0.590 | 0.583 | 0.589 | 0.590 | 0.567 |

Group-centering (to remove object-category effects) leaves the pattern: at L19
target-left 0.649, reference-left 0.556, target-top 0.746, reference-top 0.701,
answer 0.608. Continuous centre-x regression gives R^2 ~ 0 (target) and negative
(reference): the last token has a coarse vertical code, but no usable horizontal
position.

### 10.2 With the question fixed, does one token answer?

`experiments/build_train_pairs.py` + `extract_train_features.py` +
`per_group_token_probe.py`: 600 train pairs per group from GQA train images
(balanced, held-out val images), 80 test examples per group.

| group | mean cell acc | best cell | model acc |
|---|---:|---:|---:|
| chair left of table | 0.52 | 0.70 (S6) | 0.56 |
| man left of person | 0.50 | 0.69 (S1) | 0.55 |
| man right of woman | 0.53 | 0.71 (S12) | 0.65 |
| person right of man | 0.51 | 0.68 (S12) | 0.59 |
| tree left of building | 0.52 | 0.66 (S1) | 0.53 |
| woman left of man | 0.52 | 0.69 (S6) | 0.55 |

Mean accuracy is at chance; the best cell is 0.61-0.71, i.e. noisy and not
better than the model itself. The earlier chance-level per-token result was not
only a missing-question artifact.

### 10.3 Can the model report a side when asked?

`side_report.py`: 960 items per option order, 3,840 generations.

| axis | accuracy | prediction distribution |
|---|---:|---|
| horizontal (left/right) | 0.506 | "left" 953, "right" 18 (order 0: "left" 960/960) |
| vertical (top/bottom) | 0.492 | "top" 592, "bottom" 352 (truth majority baseline 0.596) |

The answers do not change when the option order is reversed: horizontal is a
fixed "left" prior, vertical a fixed "top" prior. The model is not reading the
box; it is emitting a prior. On the vertical axis it is below the truth majority
baseline (0.492 vs 0.596), i.e. worse than always answering "bottom".

### 10.4 Is the position in the visual tokens?

Pooled target-region features (oracle box) predict "target on the left half"
with AUC **0.949 / 0.943 / 0.948** at S1/S6/S12. The visual representation
contains horizontal position; the decision state does not.

### 10.5 Is composition intact?

`side_given_relation.py` gives the model the true sides in the prompt, plus a
swapped-side manipulation check:

| condition | overall acc | acc when the two objects are in different halves (n=267) |
|---|---:|---:|
| true sides | 0.760 | **0.921** |
| swapped sides | 0.271 | 0.120 |

The model follows the statements (swapping them flips the answer) and composes
them almost perfectly when they are supplied. This shows that the **text-side**
composition circuit works on verbal premises; it does not by itself prove that
the full visual composition circuit is intact. The supportable claim is that the
dominant failure lies in the path from localization/binding to the side
representation and its text readout.

### 10.6 Revised diagnosis

```
pixels -> visual tokens        horizontal side decodable at AUC 0.95 (oracle box)
       -> decision state       target-left 0.65, reference-left 0.56 at L19
       -> generation           constant "left" / "top" prior, accuracy 0.49-0.51
       -> composition          intact: 0.92 with true sides, 0.12 with swapped
```

The dominant bottleneck is the **visual-to-text readout of object position**,
with localization/binding as a contributing stage; encoding holds and text-side
composition is available. This also explains why decomposition prompting fails
as a zero-shot intervention: the sub-question ("which half?") cannot be
answered.

Implications:

1. Stage 1's "ask each side first" prompting path is closed; the sub-capability
   is absent.
2. SFT should first teach side reporting (visual -> text), then the relation.
   The relation stage can reuse the already-working composition.
3. A two-stage system is a legitimate alternative: a small side reader on visual
   features (0.95 with oracle boxes) plus the relation prompt (0.92). Deployable
   only if a proposal stage supplies the boxes.


---

## 11. SFT pilot (2026-11-18): teaching side reporting transfers to relations

Data: 2,954 GQA train images. 11,816 side questions (which half is object X in)
plus 3,600 relation questions (600 per group, balanced). Model: SmolVLM2-2.2B,
vision tower frozen, LoRA r=8 on q_proj/v_proj (2.57M trainable, 0.11%), bf16,
batch 1, 4,000 steps, 14 minutes, 5.55 GB peak.

| metric | base | + LoRA (side 77% / relation 23%) | delta |
|---|---:|---:|---:|
| side report, horizontal | 0.506 | **0.689** | +0.183 |
| side report, vertical | 0.492 | **0.704** | +0.212 |
| relation accuracy | 0.569 | **0.656** | +0.087 |
| relation AUC | 0.600 | **0.750** | +0.150 |
| composition, true sides (diff-half) | 0.921 | 0.903 | -0.018 |
| composition, swapped sides (diff-half) | 0.120 | 0.277 | +0.157 |

The side answers stop being a constant prior (the prediction distribution becomes
close to 50/50), and the relation gain appears in 5 of 6 groups. Composition does
not degrade: the model still follows swapped side statements.

**2x2 control.** Four adapters were trained with the same config (4,000 steps,
LoRA r=8), differing only in the training tasks:

| adapter | side horizontal | side vertical | relation acc | relation AUC | delta AUC vs base |
|---|---:|---:|---:|---:|---:|
| base | 0.506 | 0.492 | 0.569 | 0.600 | - |
| side-only | 0.645 | 0.682 | 0.617 | 0.717 | +0.117 [0.071, 0.161] |
| relation-only | 0.551 | 0.492 | 0.683 | 0.777 | +0.177 [0.124, 0.229] |
| mixed | 0.689 | 0.704 | 0.656 | 0.750 | +0.150 [0.102, 0.199] |
| full (1 epoch, seed 0) | 0.714 | 0.802 | 0.721 | 0.799 | +0.199 [0.145, 0.254] |
| full (1 epoch, seed 1) | - | - | 0.665 | 0.781 | +0.181 [0.126, 0.238] |

Intervals are paired bootstrap by pair id (2,000 draws). The side-only delta in
accuracy is +0.044 [-0.008, 0.094], so the transfer is in ranking rather than in
the thresholded answer. A second side-only seed gives +0.105 AUC [0.060, 0.150]
(accuracy +0.021 [-0.027, 0.069]), so the ranking gain is stable across seeds. Relation-only training leaves side reporting at base
level, so position readout needs explicit supervision; direct relation
supervision is stronger for the relation task itself.

This supports the diagnosis: teaching the missing readout (object side) moves
the relation decision even without any relation labels, while relation labels
alone do not teach the readout. A full epoch of mixed training is stronger
(+0.199 AUC, +0.150 accuracy), and the blank-image control stays at 0.500 for
both the base model and the full adapter, so the gain requires the image.

**Cross-group generalization.** A mixed adapter trained with the relation
supervision for two groups removed (`chair|left of|table`, `tree|left of|building`)
still improves both held-out groups: chair 0.723 -> 0.772 AUC, tree 0.523 ->
0.802. Most of this is already carried by side-only training (side-only gains on
the same groups are +0.036 and +0.260); the extra gain from the remaining
groups' relation supervision is only +0.014 and +0.019. The cross-group run
therefore supports "side supervision transfers across groups", not "relation
supervision generalizes across groups". The full-vs-4k comparison is also
confounded by training steps (15k vs 4k).

**Fresh confirmation set.** A new set of 240 pairs was built from 71,302 GQA
train images that are in no pair file and in no SFT batch (image-disjoint from
every prior evaluation; the base model may have seen them in pretraining). Base
scores 0.510 accuracy / 0.531 AUC there, while the full adapter scores 0.700 /
0.802; flip consistency gives 0.750 for base and 0.821 for full. The SFT gain
replicates on unseen images, and the adapter's number is essentially unchanged
from the original test (0.799 vs 0.802). The base number on the original test
(0.600) was therefore optimistic; the adapter result was not.

**Regressions.** POPE improves (0.783 -> 0.850 accuracy, F1 0.732 -> 0.852).
Fine-grained tasks are stable: overall 0.832 -> 0.823, attribute 0.765 -> 0.805,
behavior 0.865 -> 0.820, existence 0.865 -> 0.845; the gray-image control is
unchanged (0.807 -> 0.802). TIFA shows a trade-off: cued recall rises (0.753 ->
0.867) while free-form coverage falls (0.788 -> 0.588), mostly because the
adapter follows the "one or two sentences" instruction and writes 12.2 words
instead of 39.6; at matched length coverage is 0.702 vs 0.788. The adapter's
logit lens moves the readable signal to layers 17-18 (0.667 / 0.831) and
strengthens it (final 0.864 vs 0.634).
