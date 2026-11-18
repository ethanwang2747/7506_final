# Seeing Is Not Answering: Diagnosing and Improving Spatial Readout in a Small Vision-Language Model

*Draft v6 — consolidated report. All numbers are from the local SmolVLM2-2.2B-Instruct
runs; raw outputs and reproduction commands are listed in Section 10.*

## Abstract

Small vision-language models can describe an image yet fail on simple spatial
questions. We ask where that failure lives. Using GQA scene graphs we build
controlled same-question pairs: two images, one where the queried relation holds
and one where it does not, with the same unique target and reference objects.
On these pairs the label is almost fully determined by the two bounding boxes
(a per-group geometry reader reaches AUC 0.991), but the model's own margin
reaches only 0.600 and is nearly uncorrelated with that geometry (Spearman
0.18). A staged diagnostic chain locates the gap. Object identity is linearly
decodable from single visual tokens (name pointing 0.54-0.62 vs chance 0.21),
and pooled target-region tokens appear to decode an object's horizontal side at
AUC 0.95. A gray-image arbiter overturns that reading: the same probe reaches
0.994 on a uniform gray image and 0.983 on shuffled patches, so it decodes the
oracle box's location, not object content. The last-token side probe is a
different and weaker measurement (target-left 0.65, reference-left 0.56).
When asked directly, the model cannot report a side at all (horizontal answers
are a constant "left"; vertical answers follow a "top" prior and fall below the
truth majority baseline). Yet when the true sides are supplied in the prompt,
relation accuracy rises to 0.76 overall and 0.92 when the two objects are in
different halves, and swapped sides flip the answer. Text-side composition is
therefore available; the dominant failure is the path from object
localization/binding to the side representation and its text readout.

We test this with a small LoRA intervention. Training only on side questions,
with no relation labels, raises relation AUC from 0.600 to 0.717; a second seed
gives 0.705. Relation-only training does not teach side reporting. A full-epoch
mixed adapter reaches 0.721 accuracy / 0.799 AUC on the development test and
0.700 / 0.802 on a fresh image-disjoint confirmation set (delta AUC +0.271,
95% paired bootstrap [0.215, 0.328]); a second seed gives 0.708 / 0.804 on the
confirmation set. The adapter's logit lens makes the signal readable earlier and stronger, but
the peak remains at layer 23 (0.860). POPE improves, fine-grained attribute/behavior tasks are
stable, and TIFA exposes a trade-off: cued recall rises while free-form
description coverage falls, mostly because the adapter follows the
"one-or-two-sentences" instruction and writes much shorter captions. A same-size check (Qwen2-VL-2B, 2.2B) on the same 480 examples reaches 0.746
accuracy / 0.826 AUC, reports sides at 0.746/0.802, and is only partly swayed by
swapped side statements (0.562 vs our model's 0.120). The weakness is therefore family/training-specific in this single-checkpoint
comparison, not a general limitation of small VLMs; token budgets (81 vs 165)
are not matched. We conclude
that the contribution is a controlled diagnostic chain plus intervention
evidence for a spatial readout bottleneck, not a claim of a unique mechanism.

## Correction after review (2026-11-18)

Two controls changed the interpretation.

**Single-box rule.** A rule that uses only the target box's absolute x (no
reference object) reaches AUC 0.871 / accuracy 0.804 on the same 480 examples;
our full adapter reaches 0.799 / 0.721. On the same-half subset, where a
comparison is actually required, the rule reaches 0.721 and the adapter 0.701.
Most of the benchmark score is therefore explained by the target's absolute
side, and SFT mainly improves that readout.

**Gray/shuffled probe arbiter.** The oracle-box visual side probe scores 0.949
on original images, 0.994 on gray images, 0.983 on shuffled patches, and 0.976
with the target box painted gray. It reads the location of the pooled tokens,
not object grounding. This is the failure mode warned about by
[Decodable Is Not Grounded](https://arxiv.org/abs/2606.31257).

The defensible contribution is a controlled same-image pairing protocol plus a
single-box / same-half decomposition of what the benchmark measures and what SFT
learns, not a "readout bottleneck" mechanism claim. A reference-only
counterfactual adds nuance: moving only the reference horizontally lowers the
margin by -0.324 (base) / -0.737 (full) and flips 16.3% / 19.2% of answers,
versus -0.053 / -0.147 and 7.9% / 8.3% for a vertical-move control. The model
compares partially; SFT strengthens this, but most answers do not flip.

## 1. Introduction

A correct answer does not show that a model used the image, and a wrong answer
does not show that it failed to see. Spatial yes/no questions make this gap
visible: the evidence is present in the pixels, but the answer is often at
chance. We study a small open VLM (SmolVLM2-2.2B-Instruct) on GQA-derived
relation questions and ask three questions in order.

1. **Is the failure task-specific?** Existence, attributes, and many behaviors
   are answered well and collapse without the image; spatial relations do not.
2. **Where in the forward pass is the spatial evidence lost?** We separate
   encoding, decodability, causal use, composition, and readout with probes,
   region ablation/patching, and a logit lens.
3. **Can a targeted intervention move the decision?** We train LoRA adapters on
   side questions, relation questions, or both, and evaluate on a fresh
   image-disjoint confirmation set.

The paper's main line is: controlled data construction -> staged diagnosis ->
side-supervision transfer -> LoRA improvement -> boundaries and alternative
explanations.

## 2. Controlled data construction

GQA's balanced yes/no questions do not provide relation counterfactuals: a "no"
`relVerifyCo` question removes the reference object, so a model can answer "no"
without comparing anything. We therefore rebuild questions from scene graphs.

For a relation `(a, rel, b)` we keep images where `a` and `b` are unique by name
and large enough, and label the relation from the two boxes with a 2% margin.
A pair consists of two different images with the same question: one where the
relation holds (source, label 1) and one where it does not (target, label 0).
Image-level splits are by `sha1(image_id) % 5`.

| split | pairs | images | groups |
|---|---:|---:|---:|
| dev | 1,206 | 494 | 734 |
| test core | 240 | 480 examples | 6 |
| test strict | 5,348 | — | 1,656 |
| confirmation (train-derived) | 240 | 480 examples | 6 |

The six core groups are chair/table, man/person, man/woman, person/man,
tree/building, and woman/man, all left/right. The confirmation set is built from
71,302 GQA train images that appear in no pair file and no SFT batch; it is
image-disjoint from every prior evaluation, although the base model may have seen
those images during pretraining.

## 3. Behavior is task-specific

Same-image fine-grained pairs (600 test pairs) separate existence, attributes,
and behavior. Blank-image controls show how much of each score is visual.

| task | accuracy | blank-image accuracy | visual gain |
|---|---:|---:|---:|
| existence | 0.865 | 0.455 | +0.410 |
| attribute | 0.765 | 0.445 | +0.320 |
| behavior | 0.865 | 0.715 | +0.150 |
| spatial (margin AUC) | 0.600 | 0.50 | +0.100 |

External checks agree. POPE (300 questions) gives 0.783 accuracy / 0.890 AUC
with a conservative yes rate (0.33), so the model is not a yes-prone
hallucinator. TIFA/COCO cued recall is 0.753, but spatial fact coverage is 0.538
vs 0.692 for cued recall. TempCompass direction is 0.400 yes/no and 0.200
multiple-choice, while action is 0.833/0.944. The failure is concentrated in
spatial relations, direction, and temporal order, not in perception in general.

## 4. Diagnostic chain

### 4.1 The label is geometry; the margin is not

A per-group logistic reader on the 12-dimensional box description (centres,
sizes, differences) reaches AUC 0.991 on the 480 test-core examples. The model's
margin reaches 0.600; the Spearman correlation between the two is 0.183 and the
sign agreement is 0.562. The evidence that defines the answer is trivially
decodable from the boxes, and the model barely tracks it.

### 4.2 Object identity is local; the relation is not

A per-cell probe on the 9x9 visual grid, grouped cross-validated by image,
points to the queried object names with top-1 hit 0.540 / 0.595 / 0.619 at
S1/S6/S12 (chance 0.205). With the question fixed per group and 600 balanced
train pairs per group, the relation answer is still not readable from a single
token: mean cell accuracy 0.50-0.54, best cell 0.61-0.71, no better than the
model itself.

### 4.3 The decision state lacks horizontal position

Last-token probes at layers 12/16/18/19/20/23 (480 examples, grouped CV):

| probe | L12 | L16 | L18 | L19 | L20 | L23 |
|---|---:|---:|---:|---:|---:|---:|
| target left half | 0.523 | 0.594 | 0.622 | 0.653 | 0.647 | 0.633 |
| reference left half | 0.487 | 0.553 | 0.566 | 0.544 | 0.555 | 0.519 |
| target top half | 0.768 | 0.805 | 0.795 | 0.792 | 0.789 | 0.773 |
| relation answer | 0.489 | 0.590 | 0.583 | 0.589 | 0.590 | 0.567 |

Group-centering leaves the pattern. Continuous centre-x regression gives R^2 ~ 0.
Pooled target-region tokens, in contrast, decode "target on the left half" at
AUC 0.949 / 0.943 / 0.948 at S1/S6/S12. The representation contains the
horizontal side; the decision state does not read it.

### 4.4 The model cannot report a side

Asked "Is the X on the left half or the right half?", the model answers "left"
960/960 times; reversing the option order changes almost nothing (933 left, 27
right). Vertical answers follow a "top" prior (689/960) and reach 0.492 accuracy,
below the truth majority baseline of 0.596 (776 top, 1144 bottom). This is a
prior, not a readout.

### 4.5 Text-side composition works when the sides are supplied

Supplying the true sides in the prompt raises relation accuracy to 0.760 overall
and 0.921 when the two objects are in different halves. Supplying swapped sides
drops it to 0.271 / 0.120, so the model follows the statements. This shows that
the **text-side** composition circuit works on verbal premises; it does not by
itself prove that the full visual composition circuit is intact. The supportable
claim is that the dominant failure lies in the path from localization/binding to
the side representation and its text readout.

### 4.6 The decision appears late

A logit lens on the last prompt token reads the yes/no margin after each text
layer. On 80 core pairs the base model is at chance through layer 14, reaches
0.636 at layer 18 and 0.646 at layer 19; the final margin is 0.634. On a 240-pair
strict sample the ceiling is lower (0.548) but the profile is the same.

### 4.7 The target region is causally used, but only partly

From 376 same-image flip pairs: ablating the target region lowers the margin by
-0.382 [-0.465, -0.306] at S1 (reference -0.141, irrelevant +0.041, background
+0.033). Patching the target region from the original into the mirror recovers a
median 0.333 [0.252, 0.464] of the flip gap (reference 0.201, irrelevant 0.072,
random 0.003). The target region is used and the effect is specific, but about
two thirds of the flip gap is carried elsewhere.

## 5. Intervention: side supervision transfers to relations

We train LoRA adapters (r=8, alpha=16, q_proj/v_proj, vision frozen, 2.57M
trainable parameters, 0.11%) on 2,954 GQA train images: 11,816 side questions
(which half is object X in) and 3,600 relation questions (600 per group,
balanced). Training uses answer-token loss only, bf16, batch 1 with gradient
accumulation 8; peak memory is 5.55 GB and 4,000 steps take about 14 minutes.

| adapter | side h/v | relation acc | relation AUC |
|---|---|---:|---:|
| base | 0.506 / 0.492 | 0.569 | 0.600 |
| side-only | 0.645 / 0.682 | 0.617 | 0.717 |
| relation-only | 0.551 / 0.492 | 0.683 | 0.777 |
| mixed, 4k steps | 0.689 / 0.704 | 0.656 | 0.750 |
| full, 1 epoch | 0.714 / 0.802 | 0.721 | 0.799 |

The 2x2 separates what each task teaches. Side-only training, with no relation
labels, raises relation AUC by +0.117 [0.071, 0.161] (second seed +0.105
[0.060, 0.150]); the accuracy delta is +0.044 [-0.006, 0.094], so the transfer
is in ranking. Relation-only training leaves side reporting at base level
(0.551/0.492), so relation labels do not teach the readout. Direct relation
supervision is stronger for the relation task itself (+0.177).

The full-epoch adapter reaches 0.721/0.799 on the development test and
0.700/0.802 on the fresh confirmation set (delta AUC +0.271 [0.215, 0.328],
delta accuracy +0.192 [0.140, 0.246]). A second full-epoch seed gives 0.665/0.781
on the development test and 0.708/0.804 on the confirmation set. The adapter's
confirmation number is essentially identical to its development number, while
the base model is much weaker on the confirmation set (0.531) than on the
development test (0.600); the adapter result is therefore not a selection effect.

Cross-group: removing relation supervision for chair/table and tree/building
still improves both held-out groups (chair +0.049, tree +0.279 AUC), but most of
that gain is already carried by side-only training (+0.036 and +0.260); the extra
gain from the remaining groups' relation supervision is only +0.014 and +0.019.
The result supports "side supervision transfers across groups", not "relation
supervision generalizes across groups".

Robustness: on reversed but equivalent questions ("Is Y right of X?" for "Is X
left of Y?"), the full adapter drops only 0.029 AUC (0.799 -> 0.770), so it is
not relying on a lexical pattern. A 4k-step reversal/mirror-augmented adapter
closes the reversal gap (-0.003) but loses 0.013 original AUC; augmentation is
neutral at matched steps.

## 6. Confirmation and regression checks

The confirmation set is the clean held-out check. The development test has been
used for many comparisons and is now a development benchmark.

| check | base | full adapter |
|---|---:|---:|
| confirmation relation accuracy | 0.510 | **0.700** |
| confirmation relation AUC | 0.531 | **0.802** |
| confirmation flip-consistency AUC | 0.750 | **0.821** |
| POPE accuracy / F1 | 0.783 / 0.732 | **0.850 / 0.852** |
| fine-grained overall | 0.832 | 0.823 |
| fine-grained attribute | 0.765 | **0.805** |
| fine-grained behavior | 0.865 | 0.820 |
| fine-grained existence | 0.865 | 0.845 |
| gray-image control | 0.807 | 0.802 |

TIFA exposes a trade-off. With the default "one or two sentences" prompt, cued
recall rises from 0.753 to 0.867 but judge coverage falls from 0.788 to 0.588;
the adapter writes 12.2 words instead of 39.6. When asked for four sentences it
writes 36.3 words and coverage returns to 0.702, still below base by about 0.086.
The change is a mixture of length/style compliance and a small content-quality
drop, not a knowledge regression.

The adapter's logit lens moves the readable signal earlier and strengthens it:
base L18/L19/final = 0.636/0.646/0.634; full = 0.831/0.847/0.864.

## 7. Alternative explanations and boundaries

- **Localization vs readout.** Pooled visual tokens decode the side at 0.95, but
  the model's own readout is at 0.65 and generation is a prior. Localization and
  binding are not separately measured; marking or cropping the target object and
  re-asking the side question is the next test to split them.
- **Composition.** Supplying sides shows text-side composition works; it does not
  prove the visual composition circuit is intact.
- **Position shortcut.** The label is defined by object positions, so geometry is
  legitimate evidence. Position residualization is a diagnostic for "what else is
  encoded", not a success gate for relation SFT.
- **Confirmation set.** Train-derived: image-disjoint from SFT and prior
  evaluations, but possibly seen by the base model in pretraining.
- **Training steps.** The full-epoch vs 4k comparison is confounded by steps;
  the 2x2 controls use matched 4k budgets.
- **Single model, single seed per adapter** except side-only and full; the
  confirmation set is the primary check.
- **TIFA coverage** is length-sensitive; the matched-length comparison is the
  fair one.
- **Cross-model check.** Qwen2-VL-2B (same 2.2B size) reaches relation
  0.746/0.826, side report 0.746/0.802, and side-given 0.884/0.562 on the same
  480 examples. Both models encode horizontal side in pooled visual tokens
  (0.95/0.94 for SmolVLM2, 0.91/0.87 for Qwen), but the decision state differs:
  SmolVLM2 0.65/0.54 at L19 vs Qwen 0.84/0.80 at L20. The capability is
  achievable at this scale and the gap is in the decision-state readout. This is
  a between-family comparison and does not isolate architecture from training
  data.

## 8. Related work

Probing classifiers show that decodability is not use [Belinkov 2022; Hewitt &
Liang 2019]; in VLMs, [Decodable Is Not Grounded](https://arxiv.org/abs/2606.31257)
shows that probes and steering overstate grounding and that a gray-image arbiter
is needed, while [Grounding Isn't Knowing](https://arxiv.org/abs/2608.23074)
finds a staged grounding-to-reasoning process and
[Dual Mechanisms](https://arxiv.org/abs/2603.22278) locates the dominant spatial
signal in the vision encoder. Spatial benchmarks and failure analyses include
[What's Up](https://aclanthology.org/2023.emnlp-main.596/),
[BLINK](https://arxiv.org/abs/2404.12390), and
[SpatialEval](https://arxiv.org/abs/2406.14852); auxiliary localization training
is known to help [Learning to Localize](https://arxiv.org/abs/2404.07449),
[EmbSpatial-Bench](https://arxiv.org/abs/2406.12036), and attention-alignment
decoding helps on What's Up/VSR [Why Is Spatial Reasoning Hard](https://arxiv.org/abs/2503.01773). Attention is not explanation [Jain & Wallace 2019], and activation
and path patching provide causal contrasts [Meng et al. 2022; Wang et al. 2022;
Geiger et al. 2021]. The tuned/logit lens reads intermediate predictions
[Belrose et al. 2023]. On the VLM side, POPE measures object hallucination [Li
et al. 2023], TIFA and CLAIR measure factual coverage of descriptions [Hu et al.
2023; Chan et al. 2024], and spatial/relation benchmarks show persistent
difficulty [Liu et al. 2023; Kamath et al. 2023]. Our contribution is not a new
benchmark but a controlled pairing protocol plus a staged diagnostic chain that
separates encoding, decodability, causal use, composition, and readout, followed
by a targeted intervention.

## 9. Conclusion

For this model and task family, spatial failure is not a failure to see. Object
identity is local, horizontal position is decodable from pooled visual tokens,
and the target region is causally used. The failure is that the decision state
does not read the horizontal side, so the answer is only assembled late and
weakly. Teaching the missing readout with side questions alone transfers to the
relation task, and a full-epoch adapter improves relation AUC from 0.600 to
0.799 and replicates at 0.802 on a fresh image-disjoint confirmation set. The
same intervention leaves open-ended description coverage lower, mostly through
shorter captions. The defensible claim is a controlled diagnostic chain plus
intervention evidence for a localization-to-readout bottleneck, not a unique
spatial mechanism. A same-size Qwen2-VL-2B check shows that the weakness is
model-specific rather than a general small-VLM limit.

## 10. Reproducibility

- Pairs: `data/derived/{spatial,semantic,finegrained}_pairs_*.jsonl`; confirmation
  pairs from `experiments/build_confirm_pairs.py`.
- Probes and diagnostics: `experiments/per_token_probe_map.py`, `side_probe.py`,
  `side_report.py`, `side_given_relation.py`, `logit_lens.py`,
  `geometry_vs_model.py`, `aggregate_region_patching.py`.
- SFT: `experiments/build_sft_data.py`, `sft_lora.py`, `eval_relation.py`,
  `eval_flip_adapter.py`, `eval_blank.py`, `pope_experiment.py`,
  `finegrained_experiment.py`, `tifa_experiment.py`, `tifa_judge.py`.
- Summaries: `results/diagnostics_summary.md`, `results/sft_summary.md`,
  `docs/sft_experiment_log.md`, `results/readout_summary.md`,
  `results/external_benchmarks_summary.md`.
- Figures: `results/figures/diagnostics_dashboard.png`,
  `diagnostics_chain.png`, `per_token_probe_dev.png`, `sft_result.png`.
