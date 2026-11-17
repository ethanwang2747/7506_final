# External benchmark summary

This table consolidates the external checks. They are all small samples, intended
to test whether the self-built findings generalise, not to produce leaderboard
numbers.

## Results

| Benchmark | Task / metric | Sample | Result | Control | Interpretation |
|---|---|---|---:|---|---|
| **POPE** | existence yes/no accuracy | 300 Q, 229 COCO images | **0.783** (AUC 0.890) | — | existence is largely answered; adversarial 0.740 |
| **POPE** | flip-combined, held-out | 20 image splits | **0.798** vs 0.779 baseline; F1 0.757 vs 0.732 | flip wins 20/20 | training-free +1.8pp accuracy |
| self-built existence | same-image pair acc | 200 pairs | 0.865 | blank image 0.455 | confirms external result |
| **TIFA/COCO** | cued fact recall | 100 images, 466 facts | **0.753** | — | facts are often answerable when cued |
| **TIFA/COCO** | caption coverage | same | **0.788 judge / 0.431 keyword** | — | descriptions are not uniformly worse |
| **TIFA/COCO** | spatial fact coverage | 26 spatial facts | **0.538** | cued 0.692 | spatial facts are the exception |
| **TIFA/COCO** | structured prompt (spatial) | 100 images | **0.385** vs 0.538 default | — | prompt alone does not fix spatial omission |
| **TempCompass** | video yes/no | 30 videos, 114 Q | **0.605** | black video 0.377 | model uses video, but limited |
| **TempCompass** | video multi-choice | 58 Q | **0.552** | black video 0.431 | strong text prior; direction weak |
| **TempCompass** | open description coverage | 29 videos | **0.241** | black video 0.000 | visual, but often wrong/omitted |
| **Something-Something V2** | action pair accuracy | 30 videos | **0.667** | black video 0.600 | modest visual gain |
| **Something-Something V2** | action AUC | 30 videos | **0.729** | black video 0.601 | some signal, but weak |
| self-built spatial | yes/no pair acc | 240 pairs | 0.600 | blank text 0.500 | weak |
| self-built spatial | targeted description | 60 pairs | 0.117 source / 0.050 target | yes/no 0.517 | strong production failure |

## What generalises

1. **Existence is not a generic failure.** POPE (0.783) and the self-built
   existence pairs (0.865) agree. The model is conservative (POPE yes rate
   0.33, precision 0.93, recall 0.61), not hallucination-prone.
2. **Description is not uniformly worse.** On COCO/TIFA, long descriptions
   cover object/attribute/location facts at least as well as cued recall.
   Description hurts mainly for **spatial relations**: TIFA spatial coverage
   0.538 vs cued 0.692, and the targeted spatial pilot shows 0.05–0.12 vs 0.52.
3. **The weak axis is direction / spatial / temporal order.** TempCompass
   direction is near chance (yes/no 0.400, MC 0.200) while action is strong
   (0.833 / 0.944). This matches the image result.
4. **Video action discrimination is partial.** SSv2 pair accuracy 0.667 vs
   black-video 0.600; AUC 0.729 vs 0.601. TempCompass action MC is easier
   because options and text priors help; open templates are harder.
5. **Controls matter.** Black/blank controls often retain 0.38–0.60 accuracy,
   so external scores that do not control for language priors overstate visual
   understanding. Same-image/same-video counterfactuals remain necessary.

## Caveats

- All samples are small (30–300 items) and not intended as official benchmark
  numbers.
- TIFA yes/no facts are almost all positive; the cued number is recall, not
  balanced accuracy.
- Caption coverage judges are text-only and can be lenient; keyword coverage is
  a lower bound.
- TempCompass and SSv2 use 8 frames; fast actions and fine temporal order may be
  missed.
- SSv2 uses random false templates, not systematically hard negatives.
- Action Genome and Video-MME were not run; they are the natural remaining
  external checks.

## Recommendation for the report

Use the external table as **external validity**, and keep the same-image /
same-video controls as **mechanism**. The combined claim is:

> The model can read object existence and many attributes, and can answer some
> action questions, but its spatial/temporal-direction readout is the fragile
> part. “Seeing but not answering” is task-specific, not a general VLM failure.
