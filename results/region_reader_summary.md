# Learnable region reader: negative result

This tests whether a token-level region scorer can learn object localization from
bbox masks during training and then run without boxes at test.

## Setup

- Frozen S1/S6/S12 features + question embedding (`E_mean`).
- Token-level scorer produces attention over 81 visual tokens.
- Training loss: classification BCE + token-mask BCE against target+reference
  tokens + attention-mass loss.
- Test: no boxes; predicted attention pools the visual tokens.
- Global model: dev strict + dev core; 2 seeds.
- Per-group model: 14–52 dev pairs per group; 2 seeds.

## Results

Global region reader:

| Split | AUC | Pair accuracy |
|---|---:|---:|
| test pairs | 0.561 | 0.546 |
| test flips | 0.534 | 0.554 |

Per-group region reader (test / flip):

| Group | n train | Test AUC | Flip AUC |
|---|---:|---:|---:|
| chair / left / table | 14 | 0.497 | 0.491 |
| man / left / person | 40 | 0.526 | 0.497 |
| man / right / woman | 52 | 0.547 | 0.498 |
| person / right / man | 40 | 0.530 | 0.501 |
| tree / left / building | 44 | 0.514 | 0.507 |
| woman / left / man | 52 | 0.549 | 0.496 |

Both are at or near chance and below the original model margin (0.600), let alone
the oracle region probe (0.812).

## Interpretation

- A token-level mask loss on frozen features is **not enough** to learn
  transferable object localization from 14–52 examples per group. The scorer
  overfits the dev groups/images and does not generalize.
- This does not contradict the oracle region result: the information is present
  in the feature map, but the mapping from question+features to object tokens is
  not learnable with this supervision budget/architecture.
- Practical localization likely needs a **pretrained detector/segmenter** or
  much more region supervision, not just a small head on frozen features.
- The only deployable gain remains training-free flip consistency; the current
  region-reader route is a negative result.

## Caveats

- Small per-group training sets; mask BCE may need more data or a stronger
  architecture.
- Only S1/S6/S12 pooled inputs, not the full spatial feature map.
- Predicted regions were not evaluated against bbox IoU; the end metric is the
  downstream pair decision.
- A detector-based region proposal (e.g., OWLv2/GroundingDINO) is the natural
  next test.

## Reproduction

```bash
SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct PYTHONPATH=. \
python experiments/region_bce_reader.py --out results/region_reader/region_bce.json
```

Per-group results: `results/region_reader/per_group.json`.
