# Description vs cued yes/no: a pilot

This pilot asks whether open-ended description reveals failures that the yes/no candidate margin hides.

## Setup

- 60 test pairs (10 per core spatial group).
- For each source and target image, prompt: “Describe the spatial relationship between the {A} and the {B} in one short sentence.”
- Greedy generation, max 32 new tokens.
- Parse the generated sentence for left/right/above/below. A description is correct when it contains the expected relation and not its opposite.
- On the same pairs, compute the original yes/no pair accuracy from candidate margins.

## Results

| Metric | Estimate | 95% bootstrap CI |
|---|---:|---|
| yes/no pair accuracy | 0.517 | [0.383, 0.633] |
| description correct, source (yes image) | 0.117 | [0.033, 0.200] |
| description correct, target (no image) | 0.050 | [0.000, 0.117] |
| description correct on both images of a pair | 0.000 | [0.000, 0.000] |
| description with no relation word, source | 0.783 | — |
| description with no relation word, target | 0.817 | — |

Per group, yes/no pair accuracy ranges from 0.30 to 0.80, while description pair accuracy is 0.00 in every group.

Representative outputs for `woman / left / man`:

- source: “The woman is standing to the left of the man, who is seated at the table.” (correct)
- source: “The woman is standing next to the man, with the man holding a kite.” (no left/right)
- target: “The woman is standing closer to the man than the man is to the camera.” (no left/right)
- target: “The man is standing behind the woman, with his hand on her shoulder.” (wrong relation, no left/right)

## Interpretation

- The model often answers the cued yes/no question above chance, but almost never **produces** the left/right relation when asked to describe it.
- Most descriptions use coarse or different spatial language (“next to”, “in front of”, “behind”, “across from”), which may be visually reasonable but does not answer the requested relation.
- This is a stronger form of “seeing but not answering”: the evidence may be partially present, but the model does not turn it into the requested relational description.
- The effect is much larger than in the yes/no setting. This supports the intuition that description/captioning benchmarks are more revealing than candidate-margin benchmarks.

## Caveats

- The prompt is targeted (“describe the spatial relationship”), not a fully open “describe what is happening”; it is a controlled production test.
- The parser is keyword-based; “next to” is not counted as left/right even if the objects are side by side.
- n=60 pairs; per-group estimates are noisy.
- The yes/no subset is the first 10 pairs per group, not the full 40-pair test set, so its accuracy is lower than the main E1 estimate.

## Existing benchmarks to use instead of building everything

The pilot motivates using established description benchmarks, but they answer different questions:

| Goal | Benchmark | Metric |
|---|---|---|
| open image description | COCO Captions, NoCaps, Visual Genome, Flickr30k | SPICE, TIFA, CLAIR, PAC-S; CIDEr only as a secondary signal |
| object existence / hallucination | POPE, AMBER, HallusionBench, MMHal-Bench | yes/no accuracy, hallucination rate, CHAIR |
| attributes / detail / OCR | MME, MMStar, CV-Bench 2D, TextVQA, DocVQA, ChartQA | task accuracy, attribute consistency |
| action / behaviour | Kinetics, Something-Something V2, Charades, AVA, Action Genome | action accuracy, relation triplet recall |
| video “what happened” | ActivityNet Captions, YouCook2, MSR-VTT, VATEX, Video-MME, MVBench, TempCompass | captioning metrics, temporal-order QA, event recall |
| spatial relations | GQA, Visual Genome, SpatialEval, 3DSRBench, OmniSpatial, CV-Bench | relation accuracy, position-controlled splits |

**Trade-off.** Existing benchmarks provide ecological validity and comparability, but often lack same-image counterfactuals and position controls. The constructed same-image pairs remain useful as a **diagnostic**; the best design is to pair one established benchmark with the controlled pairs, not to replace one with the other.

## Reproduction

```bash
export SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct
export PYTHONPATH=.
python experiments/describe_spatial.py \
  --pairs data/derived/spatial_pairs_test.jsonl \
  --pairs-per-group 10 \
  --out results/description/spatial_test.json
```

Raw output: `results/description/spatial_test.json`.
