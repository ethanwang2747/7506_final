# Something-Something V2 small-sample action discrimination

This is the real-action external check, complementing the object-swap behavior
pairs.

## Setup

- 30 test videos from the 20bn Something-Something V2 shard `00`.
- 30 distinct action templates.
- For each video: one **true** template and one **false** template randomly
  chosen from the other videos.
- Both questions use the same video; a black video is the language-prior control.
- 8 evenly spaced frames, native video input to SmolVLM2.
- Pair accuracy: `margin(true question) > margin(false question)`.

## Results

| Metric | Full video | Black video |
|---|---:|---:|
| Pair accuracy | 0.667 [0.500, 0.833] | 0.600 [0.433, 0.767] |
| AUC over true/false questions | 0.729 | 0.601 |
| Yes rate on true question | 0.267 | — |
| Yes rate on false question | 0.133 | — |

Visual gain: +0.067 pair accuracy, +0.128 AUC.

## Interpretation

- The model uses the video, but the gain is modest on this open-template action
  task. It is much weaker than TempCompass multiple-choice action accuracy.
- The model is strongly conservative: it says yes to only 27% of true actions
  and 13% of false actions. The AUC gap (0.729 vs 0.601) is larger than the
  thresholded pair-accuracy gap, so part of the issue is calibration.
- This is consistent with the overall picture: the model handles object
  existence and some actions, but fine-grained action/relation discrimination
  is fragile, especially without answer options.
- TempCompass action MC was 0.944 full vs 0.667 blank; the options and text
  prior make that task easier. SSv2 with free-form templates is a harder test.

Representative examples:

- true “twisting (wringing) something wet until water comes out” vs false
  “digging something out of something”: full delta +1.19, blank +0.19;
- true “lifting something up completely, then letting it drop down” vs false
  “stacking number of something”: full delta −0.12, blank +0.33 (failure).

## Caveats

- 30 videos only; bootstrap CIs are wide.
- False templates are random, not systematically hard negatives or true
  opposites.
- The template wording is long and awkward; this is a hard prompt format.
- 8 frames may miss fast actions; no audio.
- Only one 1GB shard was downloaded; the sample is not a random sample of the
  full test set.

## Reproduction

```bash
export SMOLVLM_MODEL=$PWD/models/SmolVLM2-2.2B-Instruct
export PYTHONPATH=.
python experiments/ssv2_experiment.py \
  --sample data/raw/ssv2/sample_30.tsv \
  --out results/ssv2/ssv2_30.json
```

Raw output: `results/ssv2/ssv2_30.json`.
