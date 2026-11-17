# Causal attention-head intervention: no effect

This is the causal follow-up to the attention audit. We ablate or amplify the
heads with the highest target/reference attention mass, and compare with random
heads and a whole-layer ablation.

## Setup

- 57 dev pairs, six spatial groups.
- Layer 6 and 12, scaling the output of selected heads before `o_proj`.
- Conditions: clean, ablate top 2 heads per layer, amplify top heads ×2,
  ablate/amplify random heads, ablate all 32 heads of layer 6.
- Pair accuracy / AUC over source-target examples.

## Results

| Condition | Pair accuracy | AUC | Mean source-target delta |
|---|---:|---:|---:|
| clean | 0.737 | 0.710 | 1.070 |
| ablate top heads | 0.737 | 0.703 | 1.103 |
| amplify top heads | 0.737 | 0.687 | 0.823 |
| ablate random heads | 0.754 | 0.708 | 1.053 |
| amplify random heads | 0.737 | 0.712 | 0.987 |
| ablate all layer-6 heads | 0.737 | 0.746 | 0.851 |

## Interpretation

- The “top” heads from the attention audit have **no measurable causal effect**
  on pair accuracy. Ablating them is indistinguishable from ablating random
  heads; amplifying them is slightly worse.
- Even zeroing all 32 heads of layer 6 does not reduce pair accuracy (AUC is
  slightly higher, within noise). At this level, attention-head output is not
  the bottleneck for the spatial pair decision.
- This is consistent with the audit: target/reference attention mass is tiny
  and uncorrelated with correctness.
- Attention is therefore a **control row**, not the intervention target. The
  next causal tests should target latent content (steering directions) or the
  localization/readout interface.

## Caveats

- 57 dev pairs; only layers 6/12; output scaling, not attention-pattern
  patching.
- A null result at this scale does not prove attention is irrelevant for all
  tasks; it does rule out the simple “amplify a relation head” story.
- Layer 6 all-head ablation may be compensated by later layers; a full
  layer-wise sweep would be needed to rule out attention entirely.

## Reproduction

```bash
SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct PYTHONPATH=. \
python experiments/attention_head_intervention.py \
  --out results/attention/head_intervention_dev.json
```

Raw output: `results/attention/head_intervention_dev.json`.
