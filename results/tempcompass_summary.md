# TempCompass 30-video pilot

This is the video “what happened” pilot. TempCompass tests action, direction,
speed, event order, and attribute change in short videos.

## Setup

- 30 videos, 6 per temporal dimension.
- 8 evenly spaced frames per video, passed as native video input to SmolVLM2.
- Per video: 4 yes/no questions, 2 multiple-choice questions, 1 open description.
- Every question is repeated with a **black video** (8 black frames) as the
  language-prior control.
- Description coverage is judged by a local Qwen2.5-Coder-3B text judge against
  the reference option; keyword coverage is a strict lower bound.

## Results

Overall (201 task items):

| Task | n | Full video | Black video | Visual gain |
|---|---:|---:|---:|---:|
| yes/no accuracy | 114 | 0.605 | 0.377 | **+0.228** |
| multiple-choice accuracy | 58 | 0.552 | 0.431 | **+0.121** |
| description judge coverage | 29 | 0.241 | 0.000 | **+0.241** |
| description keyword coverage | 29 | 0.324 | 0.000 | +0.324 |

By dimension:

| Dimension | yes/no full / blank | MC full / blank | Description judge coverage |
|---|---|---|---|
| action | 0.833 / 0.250 | 0.944 / 0.667 | 0.444 |
| direction | 0.400 / 0.450 | 0.200 / 0.100 | 0.200 |
| speed | 0.688 / 0.500 | 0.250 / 0.375 | 0.000 |
| order | 0.611 / 0.389 | 0.500 / 0.400 | 0.000 |
| attribute_change | 0.375 / 0.417 | 0.500 / 0.417 | 0.333 |

## Interpretation

- The model **does use the video** on average: yes/no full 0.605 vs black 0.377;
  MC 0.552 vs 0.431; descriptions cover 0.241 of references vs 0 for black
  video.
- The gain is very uneven. **Action** is the strongest dimension (yes/no 0.833,
  MC 0.944). **Direction** is near chance in yes/no and MC, and description
  coverage is low. Speed and order descriptions cover none of the references in
  this small sample.
- This mirrors the image results: direction/spatial/temporal-order evidence is
  the weak part, while object/action recognition is comparatively strong.
- The black-video control is important: MC accuracy is 0.431 without any video,
  so part of the multiple-choice score is a text prior. The description prompt
  has no options, and the black-video descriptions never match the reference,
  so the description gain is more clearly visual.

Representative failures:

- reference “pushing bikes up a hill” → generated “a group of mountain bikers
  ride down a dirt trail”;
- reference “eating food” → generated “a small brown squirrel is climbing up a
  rock”;
- direction/order questions often answered from a prior rather than the frames.

## Caveats

- 30 videos only; per-dimension n is 4–9 descriptions and 16–36 yes/no items.
- 8 frames may miss fast direction changes and fine order; native video was fed
  with default fps because frame timestamps were unavailable.
- No audio or subtitles; TempCompass videos are short clips.
- The text judge can be lenient or strict; keyword coverage is a lower bound.
- MC questions include options and therefore have a strong text-only prior.

## Reproduction

```bash
export SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct
export PYTHONPATH=.
python experiments/tempcompass_experiment.py \
  --out results/tempcompass/tempcompass_30.json
python experiments/tempcompass_judge.py \
  --results results/tempcompass/tempcompass_30.json \
  --out results/tempcompass/tempcompass_30_judge.json
```

Raw outputs: `results/tempcompass/tempcompass_30.json`,
`results/tempcompass/tempcompass_30_judge.json`.
