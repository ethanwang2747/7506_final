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

- **Abstract.** Add: “A training-free flip-consistency readout raises AUC from 0.60 to 0.69 on held-out pairs and from 0.63 to 0.81 on the strict subset; an oracle region readout reaches 0.81/0.83, while a position-only oracle reaches 0.96/0.95, showing that the paired benchmark is strongly position-confounded.”
- **Conclusion.** Replace the statement that the failure cannot yet be localized to readout with: “The model has usable relation evidence, but it is not calibrated into a single-image margin; a mirror-consistency readout recovers part of it. The remaining gap is dominated by object localization and by the position shortcut in the benchmark.”
- **Limitations.** Add the oracle-box caveat and the position-confound caveat above. The current evidence does not support a claim of a rich, position-invariant relation code.

## Reproduction

See `results/readout_summary.md` and `experiments/README.md` (section “E5: readout-gap experiments”). Raw outputs are in `results/readout/`.
