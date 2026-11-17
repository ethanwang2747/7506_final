# POPE small-sample results

POPE is the standard object-existence / hallucination benchmark: “Is there a
<object> in the image?” with balanced yes/no labels. This run checks whether
the self-built existence result generalises to an external benchmark.

## Sample

- Official POPE annotations (`RUCAIBox/POPE`), three splits: random, popular, adversarial.
- 100 questions per split, stratified 50 yes / 50 no (300 total).
- 229 unique COCO val2014 images, downloaded from the COCO image server.
- SmolVLM2-2.2B-Instruct, no image splitting, candidate yes/no margin.

## Results

| Split | n | Accuracy | 95% CI | Precision | Recall | F1 | Yes rate | AUC | Flip agreement | Combined acc |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 300 | 0.783 | [0.730, 0.835] | 0.929 | 0.613 | 0.739 | 0.330 | 0.890 | 0.953 | 0.800 |
| random | 100 | 0.780 | — | 1.000 | 0.560 | 0.718 | 0.280 | 0.939 | 0.970 | 0.790 |
| popular | 100 | 0.830 | — | 0.971 | 0.680 | 0.800 | 0.350 | 0.928 | 0.940 | 0.850 |
| adversarial | 100 | 0.740 | — | 0.833 | 0.600 | 0.698 | 0.360 | 0.810 | 0.950 | 0.760 |

“Combined” uses `margin(original) + margin(horizontal flip)`, since existence is
truth-preserving under a horizontal flip.

## Interpretation

- The standard benchmark confirms the self-built result: existence questions
  are answered well above chance (0.783 overall; adversarial still 0.740).
  This is not a generic “seeing but not answering” failure.
- The model is **conservative**, not permissive: yes rate is 0.33, precision
  0.929, recall 0.613. It avoids hallucinating objects but misses some present
  ones. This is the opposite failure mode from a yes-biased captioner.
- The adversarial split is the hardest (0.740 vs 0.830 popular), as expected
  when objects are chosen to be plausible but absent.
- Flip agreement is 0.953; the combined score slightly improves accuracy
  (0.800) and AUC (0.899), consistent with the flip-consistency readout.

## Comparison with the self-built existence pairs

| Benchmark | Full accuracy | Blank-image accuracy | Visual gain |
|---|---:|---:|---:|
| self-built existence (test) | 0.865 | 0.455 | +0.410 |
| POPE overall | 0.783 | — | — |
| POPE adversarial | 0.740 | — | — |

The levels are similar; POPE is somewhat harder and has a stronger class-prior
structure. The direction is the same: existence is largely answered from the
image, unlike spatial left/right.

## Held-out flip-consistency and calibration

We split the 229 images into disjoint calibration/evaluation halves (20 random
splits) and compared three readouts on the held-out half:

| Readout | Mean accuracy | Std | Mean F1 |
|---|---:|---:|---:|
| original margin, threshold 0 | 0.779 | 0.021 | 0.732 |
| **flip-combined margin** (`m + m_flip`) | **0.798** | 0.022 | **0.757** |
| calibrated blend (fit on half) | 0.798 | 0.025 | — |

Flip-combination wins on **20/20** splits; average gain is +1.8 percentage
points accuracy and +2.5 points F1. Training a threshold/blend on the
calibration half adds nothing beyond the simple flip sum. This is a practical,
training-free accuracy improvement on an external benchmark.

## Caveats

- 300-question sample; per-split CIs are wide.
- The candidate-margin readout is used instead of free generation. POPE is
  normally evaluated with yes/no output; the margin is a reasonable analogue.
- POPE labels and COCO image coverage have their own noise.
- The blank-image control was not run on POPE; the self-built existence blank
  control (0.455) is the relevant reference.

## Reproduction

```bash
# Annotations
mkdir -p data/raw/pope/annotations data/raw/pope/images
for f in coco_pope_random.json coco_pope_popular.json coco_pope_adversarial.json; do
  curl -sSL -o data/raw/pope/annotations/$f \
    https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/$f
done

# Sample and images: see the command history / build script in this repo
python experiments/pope_experiment.py \
  --sample data/raw/pope/sample_300.jsonl \
  --out results/pope/pope_300.json
```

Raw output: `results/pope/pope_300.json`.
