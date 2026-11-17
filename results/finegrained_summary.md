# Fine-grained same-image discrimination: existence, attributes, behavior

This note extends the readout experiments beyond left/right spatial relations. The question is whether “seeing but not answering” is a general VLM failure or a task-specific one.

## Design

We build same-image positive/negative question pairs from GQA val scene graphs. Both questions use the same image, so image content, object identity, and spatial layout are fixed; only the question word changes.

| Task | Positive | Negative | Pairs (test) |
|---|---|---|---:|
| existence | a present object | an absent confusable/random object | 200 |
| attribute | a true colour/material | a false colour/material on the same object | 200 |
| behavior | a true relation with object A | the same relation with an absent confusable object B | 200 |

Models are evaluated by the candidate-margin difference `m_pos - m_neg`; pair accuracy is the fraction of pairs with a positive difference. Controls:

- **horizontal flip**: truth is preserved for these tasks; measures readout stability;
- **grayscale image**: removes colour but keeps shape;
- **blank image**: a uniform gray canvas; measures language priors without visual content.

## Test results (600 pairs)

| Task | Full pair acc | Blank pair acc | Visual gain (95% CI) | Full AUC | Flip acc | Gray acc | Yes rate pos/neg |
|---|---:|---:|---|---:|---:|---:|---:|
| existence | 0.865 | 0.455 | **+0.410 [0.325, 0.490]** | 0.857 | 0.855 | 0.880 | 0.61 / 0.11 |
| attribute | 0.765 | 0.445 | **+0.320 [0.245, 0.400]** | 0.737 | 0.755 | 0.690 | 0.33 / 0.07 |
| behavior | 0.865 | 0.715 | **+0.150 [0.075, 0.225]** | 0.820 | 0.895 | 0.850 | 0.57 / 0.16 |
| overall | 0.832 | 0.538 | **+0.293 [0.248, 0.338]** | 0.810 | 0.835 | 0.807 | 0.50 / 0.11 |

Paired bootstrap over the same pairs; blank image means the model sees only the question text.

### Dev replication

The same experiment on 600 dev pairs gives the same ordering:

| Task | Full pair acc | Blank pair acc | Visual gain (95% CI) |
|---|---:|---:|---:|
| existence | 0.900 | 0.475 | +0.425 [0.345, 0.510] |
| attribute | 0.745 | 0.405 | +0.340 [0.265, 0.415] |
| behavior | 0.865 | 0.660 | +0.205 [0.125, 0.285] |
| overall | 0.837 | 0.513 | +0.323 [0.277, 0.370] |

The absolute levels differ slightly, but the pattern is stable: existence and attribute questions collapse to chance without the image; behavior retains the largest language prior and the smallest visual gain.

### Confusable vs random negatives

| Negative type | n | Full acc | Blank acc | Visual gain |
|---|---:|---:|---:|---:|
| confusable object | 310 | 0.845 | 0.606 | +0.239 [0.174, 0.306] |
| random object | 90 | 0.933 | 0.511 | +0.422 [0.300, 0.544] |

Confusable negatives are harder, and their visual gain is smaller than the random-negative gain. This is the expected pattern if the model uses object evidence but also has priors about plausible objects.

### Predicate-swap behavior pilot

A second behavior pilot changes only the predicate on the same subject/object (e.g., sitting on vs standing on, riding vs walking on). It is smaller because it needs both objects to be unique and the negative predicate to be unannotated.

| Split | n | Full acc | Blank acc | Visual gain | AUC |
|---|---:|---:|---:|---:|---:|
| dev | 16 | 0.812 | 0.625 | +0.187 | 0.844 |
| test | 22 | 0.955 | 0.682 | +0.273 | 0.890 |
| combined | 38 | 0.895 | 0.658 | +0.237 [0.105, 0.368] | 0.864 |

This is consistent with the object-swap behavior result: the model discriminates well, but a substantial part of the margin ranking survives without the image. The pilot is too small for a group-level conclusion.

### Attribute subtypes

Colour questions: 183 pairs, full accuracy 0.754, blank 0.410, gray 0.672. Material questions: only 17 pairs, full accuracy 0.882, blank 0.824, gray 0.882; the material result is dominated by a language prior and is too small to interpret separately.

## Interpretation

The model is **not generically “seeing but not answering”**:

1. Existence and attribute questions are answered well (0.87 and 0.77), and the blank-image control drops to chance (0.46/0.45). The model is genuinely using the image for these questions.
2. Behavior/object-binding questions are also answered well (0.87), but the blank-image control is above chance (0.715). Part of the performance is a language prior about which object/verb combination is plausible; the visual gain is still positive (+0.15).
3. Flip consistency is high (agreement 0.91–0.95) and does not reveal a large instability, unlike the spatial left/right task.
4. In contrast, the original spatial left/right task has AUC around 0.60, and global no-box readers are at or below chance. The “seeing but not answering” failure is therefore **task-specific**: it is strongest for spatial relation/position binding, not for all visual semantics.

A useful summary is a 2×2:

|  | Image used | Image not sufficient |
|---|---|---|
| answered well | existence, attribute, behavior | — |
| answered poorly | spatial left/right | spatial left/right |

The spatial failure is not explained by a generic inability to read visual features; the same model reads existence and attributes well. It is specifically the relation/position readout that fails.

## Limitations

- Object absence is defined by the GQA scene graph; a category absent from the graph is assumed absent from the image.
- Behavior negatives are object swaps, not full action counterfactuals. They test object binding in a relation, not action recognition.
- Material questions are too few (17) for a separate conclusion.
- The blank image is a uniform canvas; a language-only baseline without any image would be cleaner but the processor requires an image input.
- The yes/no margin is a candidate readout, not free generation. The model's free-form behavior may differ.

## Reproduction

```bash
export SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct
export PYTHONPATH=.

python experiments/build_finegrained_pairs.py --max-per-task 200 --per-image 4
python experiments/finegrained_experiment.py \
  --pairs data/derived/finegrained_pairs_test.jsonl \
  --out results/finegrained/test.json
python experiments/finegrained_blank_control.py \
  --results results/finegrained/test.json \
  --out results/finegrained/test_blank.json
```

Raw outputs: `results/finegrained/test.json`, `results/finegrained/test_blank.json`.
