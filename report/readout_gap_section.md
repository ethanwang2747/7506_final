# Readout-gap experiments (report section draft)

This section is a drop-in draft for the final report. It reports the new experiments on frozen features, a training-free readout improvement, and the position-control caveat.

## Motivation

The stage-wise ablation and patching results in the main draft show that output dependence is measurable at S0–S12 and disappears at S18, but they do not distinguish three possibilities: the visual evidence is missing, the evidence is present but not linearly readable, or the evidence is readable but the model's readout does not use it. This section addresses that gap directly.

We cache frozen features at S1 (connector output), S6, and S12 (text-layer inputs), then train lightweight readers on dev pairs and evaluate on held-out test pairs from six spatial groups and on same-image horizontal flips. The readers see visual tokens and a mean question embedding; they do not see boxes unless a probe is explicitly marked as oracle.

## Training-free flip-consistency readout

For a left/right question, a relation-consistent model should give a higher yes margin on the image and a lower yes margin on its horizontal flip. We therefore score each image as

\[
s(x,q) = m(x,q) - m(\mathrm{flip}(x),q),
\]

where \(m\) is the candidate-margin readout. This requires two forwards per image, but no training, no boxes, and no labels.

| Split | Readout | AUC | 95% CI | Accuracy | Pair accuracy |
|---|---|---:|---|---:|---:|
| dev core | original margin | 0.694 | [0.642, 0.745] | 0.628 | 0.760 |
| dev core | **flip consistency** | **0.744** | **[0.678, 0.802]** | **0.653** | **0.785** |
| test core | original margin | 0.600 | [0.558, 0.645] | 0.571 | 0.604 |
| test core | flipped margin | 0.507 | [0.457, 0.553] | 0.479 | 0.504 |
| test core | **flip consistency** | **0.690** | **[0.643, 0.736]** | **0.619** | **0.683** |
| strict test | original margin | 0.633 | [0.543, 0.727] | 0.614 | 0.632 |
| strict test | **flip consistency** | **0.813** | **[0.735, 0.886]** | **0.728** | **0.807** |

Paired bootstrap over pairs gives ΔAUC = +0.051 [−0.018, 0.116] on dev core, +0.091 [0.032, 0.149] on test core, and +0.180 [0.067, 0.286] on strict test. Dev is an independent check because the readout is training-free; its interval includes zero, while the test intervals do not. The improvement is not uniform: four of six groups improve, while the two person/man groups remain near chance.

## Oracle readout upper bound

A per-group logistic reader that pools target and reference object tokens using GQA boxes reaches:

| Split | Region readout | Region, position-residualized | Position-only | Original margin |
|---|---:|---:|---:|---:|
| test | 0.812 | 0.687 | 0.961 | 0.626 |
| strict test | 0.883 | 0.749 | 0.994 | 0.665 |
| flip | 0.829 | 0.705 | 0.950 | 0.584 |
| strict flip | 0.936 | 0.765 | 0.988 | 0.701 |

The region probe is an oracle diagnostic: it uses ground-truth boxes at test time. The position-only column shows why this matters. Object positions alone nearly solve the paired benchmark. After residualizing the region features against the 12-dimensional box description, the region readout drops to 0.69 test and 0.71 flip. The remaining signal is above chance, but it is much smaller than the raw region result.

## Extension: existence, attributes, and behavior

The same same-image design can be applied beyond left/right. We build 200 positive/negative pairs per task from GQA val scene graphs, with both questions on the same image:

| Task | Positive | Negative | Full pair acc | Blank-image acc | Visual gain |
|---|---|---|---:|---:|---:|
| existence | a present object | an absent confusable/random object | 0.865 | 0.455 | +0.410 [0.325, 0.490] |
| attribute | a true colour/material | a false colour/material on the same object | 0.765 | 0.445 | +0.320 [0.245, 0.400] |
| behavior | a true relation with object A | the same relation with an absent confusable object B | 0.865 | 0.715 | +0.150 [0.075, 0.225] |
| behavior (predicate swap, n=38) | sitting on / riding / lying on | standing on / walking on / sitting on | 0.895 | 0.658 | +0.237 [0.105, 0.368] |

The blank-image control replaces the image with a uniform gray canvas, so the model sees only the question text. Existence and attribute questions fall to chance on the blank image; the model is genuinely using visual evidence and answering those questions well. Behavior/object binding retains a substantial language prior (0.715 blank), but the real image still adds +0.15.

This is the key qualification of the project's title. The model is **not generically “seeing but not answering”**. It reads existence, colour, and object-binding evidence well; flip agreement is 0.91–0.95. The failure is specific to spatial relation/position readout, where the same model is near 0.60 and no-box readers are at chance. The report should therefore claim a **task-specific spatial readout/localization gap**, not a general VLM readout failure.

### POPE external validation

The self-built existence result also holds on the standard POPE benchmark. On 300 balanced questions (100 per random/popular/adversarial split, 229 COCO images), accuracy is **0.783** [0.730, 0.835], AUC 0.890. The model is conservative: yes rate 0.33, precision 0.929, recall 0.613. Adversarial questions are hardest (0.740), popular easiest (0.830). Flip agreement is 0.953, and the flip-combined score gives accuracy 0.800 / AUC 0.899. This is consistent with the self-built existence pairs (0.865 full, 0.455 blank) and supports the conclusion that existence is not a generic readout failure.

## Description vs cued yes/no

A targeted production pilot asks for a one-sentence spatial description of the same source/target images. On 60 test pairs, yes/no pair accuracy is 0.517 [0.383, 0.633], while the description is correct for only 0.117 [0.033, 0.200] of source images and 0.050 [0.000, 0.117] of target images; no pair has both descriptions correct. Between 78% and 82% of descriptions contain no left/right word at all, often using “next to”, “in front of”, or “behind” instead. The model can sometimes answer the cued question but does not produce the requested relation. This is a stronger “seeing but not answering” signal and motivates using established caption/description benchmarks with SPICE/TIFA/CLAIR-style metrics rather than relying only on candidate margins.

### TIFA/COCO description coverage

On 100 COCO images and 466 TIFA yes/no facts (almost all positive), cued fact recall is 0.753 [0.709, 0.793]. A CLAIR-style local text judge gives caption coverage 0.788 [0.741, 0.833]; keyword coverage is 0.431. Overall, free descriptions are **not** uniformly worse than cued answers. The exception is spatial facts: cued recall 0.692 versus caption coverage 0.538 (keyword 0.231). Captions describe the objects but often omit “near”, “under”, “in front of”, or “next to”. This is consistent with the targeted spatial-description pilot and supports a spatial-specific production failure rather than a generic captioning failure.

### Video: TempCompass 30-video pilot

On 30 short videos (6 per temporal dimension, 8 frames each), yes/no accuracy is 0.605 with video versus 0.377 with a black video; multiple-choice is 0.552 versus 0.431; open-description coverage is 0.241 versus 0.000. The model therefore uses video, but the gain is very uneven. Action is strong (yes/no 0.833, MC 0.944), while direction is near chance (yes/no 0.400, MC 0.200), and speed/order descriptions cover none of the references. This is consistent with the image result: direction/spatial/temporal-order evidence is the weak part, while object/action recognition is comparatively strong.

## Learned no-box readers do not close the gap

Global readers trained on 3,788 dev examples (mean-pooled or attention-pooled S1/S6/S12 tokens plus the question) do not beat the original margin:

| Reader | test AUC | flip AUC |
|---|---:|---:|
| original margin | 0.600 | 0.559 |
| mean-pooled visual + question | 0.474 | 0.533 |
| attention-pooled visual + question | 0.443 | 0.497 |
| attention + training-time region loss | 0.548 | 0.529 |
| question-only linear probe | 0.500 | 0.500 |
| final-token H12 linear probe | 0.490 | 0.522 |

The question-only probe is exactly 0.500 because the question vector is identical for the two members of a pair, and the final-token probe does not recover the relation. The negative result is not a classifier-capacity issue: the same features support the oracle region readout. The bottleneck is finding the object tokens, not fitting a linear boundary.

## Position-control caveat

The same-image horizontal flip is not a clean position control. A position-only reader reaches 0.95 on the flip split because flipping moves objects to the other side of the image; an absolute-position cue can therefore rank original versus flipped. The flip-consistency score is symmetric by construction, and it is evaluated on cross-image source/target pairs, not on the flip split. A stronger relation claim still requires position-matched pairs or a learned object proposal stage.

## Suggested changes to the main report

- **Abstract.** Add: “A training-free flip-consistency readout raises AUC from 0.60 to 0.69 on held-out pairs and from 0.63 to 0.81 on the strict subset; an oracle region readout reaches 0.81/0.83, while a position-only oracle reaches 0.96/0.95, showing that the paired benchmark is strongly position-confounded. The failure is task-specific: same-image existence and attribute questions are answered well and fall to chance without the image, whereas spatial relations do not.”
- **Conclusion.** Replace the statement that the failure cannot yet be localized to readout with: “The model has usable relation evidence, but it is not calibrated into a single-image margin; a mirror-consistency readout recovers part of it. The remaining gap is dominated by object localization and by the position shortcut in the benchmark. The same model answers same-image existence, attribute, and behavior questions well, so the limitation is a task-specific spatial binding/readout gap rather than a general inability to use visual evidence.”
- **Limitations.** Add the oracle-box caveat and the position-confound caveat above. The current evidence does not support a claim of a rich, position-invariant relation code.

## Reproduction

See `results/readout_summary.md` and `experiments/README.md` (section “E5: readout-gap experiments”). Raw outputs are in `results/readout/`.
