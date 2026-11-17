# Attention audit: no ready-made relation heads

This is the first mechanism probe for the “attention → intervention” route.
The question: does the model's answer-position attention already focus on the
target/reference object tokens, and do any heads predict correctness?

## Setup

- 57 dev pairs from six spatial groups (10 per group).
- Layers 6 and 12, all 32 heads.
- Query: the **last answer-position token**.
- Keys: the 81 visual tokens.
- Target/reference token sets come from GQA boxes (used only for analysis).
- Selectivity = mean attention to target+reference minus mean attention to the
  rest of the visual block.
- Pair correctness = source margin > target margin.

## Results

Top heads by target+reference mass:

| Head | Target mass | Reference mass | Background | Selectivity | corr with correct |
|---|---:|---:|---:|---:|---:|
| L6H03 | 0.0007 | 0.0007 | 0.0011 | -0.0005 | +0.004 |
| L6H09 | 0.0006 | 0.0006 | 0.0008 | -0.0002 | -0.082 |
| L6H04 | 0.0006 | 0.0005 | 0.0010 | -0.0004 | -0.053 |
| L12H24 | 0.0008 | 0.0008 | 0.0011 | -0.0003 | +0.031 |
| L12H17 | 0.0006 | 0.0005 | 0.0003 | +0.0003 | -0.088 |

Key facts:

- attention mass on target/reference is **tiny** (~0.0007) and usually **lower**
  than background (~0.0011);
- selectivity is near zero or negative;
- no head has a meaningful positive correlation with pair correctness or margin
  delta (the largest correlations are weak and often negative).

## Interpretation

- There is **no ready-made “relation head”** to amplify. A simple attention-head
  steering intervention is unlikely to work.
- The model's answer token largely ignores the object tokens; this explains why
  global attention readers over all visual tokens fail, even though an oracle
  region readout works.
- The bottleneck is not “the attention weights are slightly wrong”; it is that
  the readout does not route to the object evidence at all.
- The next causal test should be **oracle attention steering**: force the answer
  token to attend to target/reference tokens (boxes only at test, as a
  diagnostic). If output improves, the routing is causally sufficient and the
  next step is a learned attention controller. If it does not, the problem is
  downstream (connector/MLP/readout), and attention is the wrong lever.

## Caveats

- 57 pairs, layers 6/12, answer-position query only.
- Attention mass over 81 visual tokens is naturally small; relative comparisons
  (target vs background) are the meaningful signal.
- Bbox token sets are oracle information used only for analysis.
- Other layers, visual self-attention, and MLP/residual paths are not covered.

## Reproduction

```bash
SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct PYTHONPATH=. \
python experiments/attention_audit.py \
  --out results/attention/audit_dev_last.json
```

Raw output: `results/attention/audit_dev_last.json`.
