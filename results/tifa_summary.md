# TIFA/COCO description vs cued fact recall

This is the second external benchmark. It asks whether free descriptions are
generally worse than cued yes/no answers, using TIFA's reference facts on COCO
images.

## Setup

- 100 COCO val2014 images and their TIFA reference captions.
- 466 TIFA yes/no facts generated from those captions.
- **Important:** TIFA yes/no facts are almost all positive (9,104 yes vs 2 no in
  the full COCO subset), so the cued number is **fact recall**, not balanced
  accuracy.
- SmolVLM2 generates one free description per image.
- Cued recall: asks each TIFA question directly and takes the candidate margin.
- Caption coverage: a local Qwen2.5-Coder-3B text judge checks whether the
  generated description contains enough information to answer the fact. This is
  a CLAIR-style text entailment check, not the official CLAIR metric.
- Keyword coverage (element word in caption) is reported as a strict lower bound.

## Results

| Metric | Estimate | Image-cluster 95% CI |
|---|---:|---|
| Cued fact recall | 0.753 | [0.709, 0.793] |
| Judge caption coverage | 0.788 | [0.741, 0.833] |
| Keyword coverage (lower bound) | 0.431 | — |
| Cued minus judge coverage | -0.034 | [-0.082, 0.013] |

By fact type:

| Type | n | Cued recall | Judge coverage | Keyword coverage |
|---|---:|---:|---:|---:|
| object | 141 | 0.809 | 0.851 | 0.518 |
| animal/human | 64 | 0.812 | 0.844 | 0.547 |
| activity | 89 | 0.629 | 0.685 | 0.382 |
| attribute | 42 | 0.714 | 0.762 | 0.190 |
| location | 39 | 0.718 | 0.846 | 0.308 |
| color | 17 | 0.824 | 0.765 | 0.588 |
| counting | 15 | 0.867 | 0.800 | 0.400 |
| **spatial** | **26** | **0.692** | **0.538** | **0.231** |

## Interpretation

- **Description is not uniformly worse than cued recognition.** The long
  descriptions cover most object, animal, food, and location facts; judge
  coverage is even slightly higher than cued recall overall.
- **Spatial facts are the exception.** Spatial coverage is 0.538 vs cued 0.692;
  captions often describe the objects but omit the relation. Examples:
  - “baby elephant standing near its mother” → caption says “baby elephant is
    standing on a dirt path”, no mother;
  - “cat under the park bench” → caption says “dogs resting on a wooden bench”,
    no cat/under;
  - “man standing in front of the open door of the bus” → caption says “man
    standing next to it”;
  - “motorcycles next to the bicycle” → caption says “motorcycles parked on the
    side of a street”, no bicycle.
- This matches the targeted spatial-description pilot (yes/no 0.52 vs
  description 0.12 on 60 pairs) and supports a **spatial-specific production
  failure**, not a generic captioning failure.
- The judge is lenient in some cases (e.g., “fruit cart” judged supported by a
  fruit-market caption), so 0.788 is an upper bound; keyword 0.431 is a lower
  bound. The spatial pattern is the same under both: spatial is the lowest
  non-trivial category.

## Attempted improvement: structured prompt

We tried to improve coverage with a prompt that explicitly asks for objects,
colours/materials, actions, and spatial relationships:

> “Describe the image in detail. Mention the main objects, their colours and
> materials, what the people are doing, and the spatial relationships between
> the main objects.”

| Metric | Default prompt | Structured prompt |
|---|---:|---:|
| Overall judge coverage | 0.788 | 0.734 |
| Overall keyword coverage | 0.431 | 0.408 |
| Spatial judge coverage | 0.538 | 0.385 |
| Spatial keyword coverage | 0.231 | 0.115 |
| Attribute judge coverage | 0.762 | 0.810 |
| Colour judge coverage | 0.765 | 0.882 |

The structured prompt improves some attribute/colour facts but **reduces**
overall and spatial coverage. Prompt engineering alone does not fix spatial
relation omission; it can even distract the model from relations. This is a
negative result for the “just ask for relations” intervention.

## Caveats

- TIFA yes/no questions are almost all positive, so there is no balanced
  precision/recall or negative control.
- The judge is a text-only 3B model; it can over-credit plausible captions. It
  is not an independent VQA model.
- NoCaps was not used; this is COCO only.
- Official TIFA/CLAIR scores require their own question generation and judge
  pipelines; this is a lightweight adaptation.

## Reproduction

```bash
export SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct
export PYTHONPATH=.
python experiments/tifa_experiment.py \
  --sample data/raw/tifa/sample_100.jsonl \
  --out results/tifa/tifa_100.json
python experiments/tifa_judge.py \
  --results results/tifa/tifa_100.json \
  --out results/tifa/tifa_100_judge.json
```

Raw outputs: `results/tifa/tifa_100.json`, `results/tifa/tifa_100_judge.json`.
