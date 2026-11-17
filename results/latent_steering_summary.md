# Latent source-target steering: no robust gain

This tests whether a simple linear “source minus target” direction in the final
token can improve spatial pair decisions.

## Setup

- Dev: 57 pairs (10 per group) to estimate the direction.
- Direction: mean final-token hidden state difference `h_source - h_target` at a
  layer input; normalized, scaled by the mean difference norm.
- Test: 57 held-out pairs.
- Conditions: clean, add direction to the target final token, subtract from the
  source, add to both, random unit direction, shuffled-pair direction.
- Layers 6 and 12; per-group directions also tested at layer 6.

## Results

Global direction:

| Layer | Condition | Pair acc | AUC | Mean delta |
|---|---|---:|---:|---:|
| 6 | clean | 0.517 | 0.549 | 0.337 |
| 6 | target + d | 0.533 | 0.544 | 0.320 |
| 6 | source − d | 0.500 | 0.544 | 0.316 |
| 6 | all + d | 0.550 | 0.549 | 0.315 |
| 6 | target + random | 0.550 | 0.558 | 0.345 |
| 12 | clean | 0.517 | 0.549 | 0.337 |
| 12 | target + d | 0.483 | 0.519 | 0.238 |
| 12 | target + random | 0.567 | 0.561 | 0.360 |

Per-group directions at layer 6 show the same pattern: `target + d` is equal or
slightly worse than clean for most groups; random directions are often as good
or better. No group has a consistent gain that survives a random-direction
control.

## Interpretation

- A single linear source-target direction at the final token does **not** carry
  usable relation evidence. Adding it to the target does not improve pair
  accuracy or AUC; subtracting it from the source does not help either.
- Random directions sometimes “improve” by chance at this sample size; the
  proper comparison is direction vs random, and the direction has no advantage.
- This is consistent with the attention result: the failure is not a single
  attention head or a single linear direction. The remaining candidates are
  **localization/region selection** and the **readout interface**.
- The one positive readout result remains training-free flip consistency:
  spatial AUC 0.600→0.690 (test), 0.633→0.813 (strict); POPE accuracy
  0.779→0.798.

## Caveats

- 57 dev/test pairs; small sample.
- Final token only; visual-token or region-level directions were not tested.
- One fixed scale (mean difference norm); a sweep might behave differently but
  would need held-out selection.
- Linear direction may mix position, object appearance, and relation; a
  nonlinear or per-relation learned controller may still work.

## Reproduction

```bash
SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct PYTHONPATH=. \
python experiments/latent_steering.py --layer 6 --out results/latent/steering_layer6.json
python experiments/latent_steering.py --layer 12 --out results/latent/steering_layer12.json
python experiments/latent_steering_per_group.py --layer 6 \
  --out results/latent/steering_per_group_layer6.json
```

Raw outputs are under `results/latent/`.
