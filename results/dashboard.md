# Experiment dashboard (working notes)

This is the lab notebook for the v5 experiment set. It is not the report.

## 1. Experiment inventory

| # | Script | Output | n | Purpose |
|---|---|---|---|---|
| A | `audit_gqa_pairs.py` | `results/audit/gqa_pairing_audit.json` | all GQA pools | object-presence confound + scene-graph coverage |
| E0 | `e0_validate.py` | `results/e0/` | 1 pair per task | hook, shape, identity patch, reproducibility |
| E1 | `e1_behavior_map.py` | `results/e1/` | 6 spatial + 5 semantic groups | clean behavior, image/question/gray controls, generation vs margin |
| E2 | `e2_ablation.py` | `results/e2/` | 20 test pairs/group | target/reference/irrelevant/background/full mean replacement |
| E2b | `e2_bilateral.py` | `results/e2/bilateral*.json` | 10 test pairs/group | source+target ablation, AUC after ablation |
| E2s | `e2_splitting.py` | `results/e2/spatial_test_splitting.json` | 5 pairs/group | splitting global-block vs full-block |
| E3 | `e3_patching.py` | `results/e3/` | 20 test pairs/group | full visual-block patching S0–S18 |
| E3r | `e3_regional.py` | `results/e3/*_regional.json` | 20 pairs/group | regional target/reference/irrelevant/random patching |
| E3t | `e3_text_position_patch.py` | `results/e3/text_position_patch.json` | 5 pairs/group | S12/S18 visual vs text vs both position patch |
| E3c | `e3_crop_patch.py` | `results/e3/crop_patch*.json` | 5 pairs/group | crop/global target activation transfer |
| E4p | `e4_stage_probe.py` | `results/e4/stage_probe*.json` | 6 groups | S0/S1/S6/S12 probe, position-only, residual, matched subset |
| E4q | `e4_question_context.py` | `results/e4/question_context.json` | 5 pairs/group | own vs control question target ablation |
| E4j | `e4_joint.py` | `results/e4/joint_ablation.json` | 10 pairs/group | target+reference joint mean replacement |
| E4c | `e4_crop_cumulative.py` | `results/e4/crop_cumulative.json` | 5 pairs/group | crop cumulative mean replacement |
| F | `flip_counterfactual.py` | `results/flip/` | 10–20 pairs/group | same-image horizontal flip, bilateral ablation/patch |
| V | `make_representation_figure.py` | `results/figures/representation_3d.png` | 1 pair | layer × token representation view |
| V | `make_data_figure.py` | `results/figures/data_validity.png` | 1 pair | image/flip boxes + 9×9/27×27 grids |

## 2. Self-check: issues found and actions

| Issue | Evidence | Action |
|---|---|---|
| Splitting block order: crops first, global last | `split_images` source; offline processor rows=4, cols=3, 13 blocks | Fixed `e2_splitting.py`; reran. Old “global = block 0” results are invalid. |
| Global block target effect | corrected global target Δm: woman −1.35, man/woman −1.67, tree −1.03, chair −0.48 | Global block carries substantial target evidence; “crops only” claim withdrawn. |
| Region probe position confound | position-only AUC 0.86–0.99 vs region 0.66–0.89 | Added residualized probe and position-matched subset; raw probe claim withdrawn. |
| Source-only E2 | target ablation only on source | Added bilateral E2; AUC after ablation reported. |
| Cross-scene pairing not minimal | random T/F images differ in scene | Added same-image horizontal flip. |
| Regional random control | source random could include target/reference | Fixed to exclude target/reference/irrelevant in both images; reran. |
| Late-layer null | visual-only patch at S18 = 0 | Added text/visual/both patch; S18 text patch 0.70–0.94, both = 1.00. |
| Crop-level distribution | global vs full block | Added crop cumulative/crop patch; global dominates, crops add little. |
| Multiple comparisons | many group × stage × region CIs | Not corrected; treat patterns descriptive, CIs per comparison. |
| Mirror groups | woman/man vs man/woman, etc. | Not independent; note in limitations. |

## 3. Corrected key numbers

### 3.1 Clean behavior (test)

Spatial: no-split only chair/table has AUC CI excluding 0.5 (0.72 [0.55,0.88]); splitting gives woman/man 0.76, man/woman 0.73, tree/building 0.76, chair/table 0.79, man/person 0.48, person/man 0.50.
Semantic: fence/wood 0.89, table/brown 0.81, door/white 0.81, building/white 0.75, jacket/black 0.69.

### 3.2 Bilateral target ablation (AUC before → after, S1/S6)

Full 40-pair test set, paired source–target accuracy before → after target ablation (S1/S6):

| group | clean | after S1 | after S6 |
|---|---:|---:|---:|
| chair/table | 0.70 | 0.55 [0.40,0.70] | 0.50 [0.35,0.65] |
| woman/man | 0.68 | 0.53 [0.38,0.68] | 0.62 [0.47,0.78] |
| man/woman | 0.70 | 0.53 [0.38,0.68] | 0.47 [0.33,0.62] |
| man/person | 0.57 | 0.50 | 0.57 |
| person/man | 0.50 | 0.53 | 0.53 |
| tree/building | 0.47 | 0.47 | 0.50 |

The three groups with clean discrimination above chance show a moderate drop (roughly 0.1–0.2); the three at/below chance do not. The after-ablation CIs include 0.5, so this is “moderate contribution,” not a decisive knockout. The earlier “drops to chance” phrasing was too strong.

Semantic 40-pair run: fence/wood 0.88→0.78/0.78; table/brown 0.85→0.55/0.60; door/white 0.82→0.75/0.72; building/white 0.75→0.65/0.57; jacket/black 0.72→0.60/0.60.

### 3.3 Same-image flip (stable pairs, |Δm|≥0.5)

40 pairs/group; R uses stable pairs with |Δm|≥0.5:

| group | clean AUC | stable n | target R S1 | target R S6 |
|---|---:|---:|---:|---:|
| chair/table | 0.75 [0.60,0.88] | 11 | 0.45 [0.02,0.87] | 0.43 [0.20,0.87] |
| woman/man | 0.75 [0.60,0.88] | 20 | 0.56 [0.40,0.67] | 0.58 [0.44,0.78] |
| man/person | 0.62 [0.47,0.78] | 9 | 0.29 [0.13,0.75] | 0.33 [0.00,1.00] |
| man/woman | 0.60 [0.45,0.75] | 18 | 0.63 [0.29,0.88] | 0.68 [0.29,0.88] |
| person/man | 0.57 [0.42,0.72] | 17 | 0.32 [0.12,0.63] | 0.25 [0.13,0.62] |
| tree/building | 0.55 [0.40,0.70] | 7 | 0.81 [0.62,1.00] | 0.74 [0.38,0.86] |

With 40 pairs, every group has a positive target-transfer R, but only chair/table and woman/man have clean AUC CIs excluding 0.5. `tree/building` has the largest R (0.81) with only 0.55 clean AUC; `man/woman` has R=0.63 with AUC 0.60 (CI includes 0.5). This is the key dissociation: positive regional transfer does not imply above-chance source–target ranking.

Same-scene clean AUC with 95% bootstrap CIs (40 pairs): chair/table 0.75 [0.60,0.88]; woman/man 0.75 [0.60,0.88]; man/person 0.62 [0.47,0.78]; man/woman 0.60 [0.45,0.75]; person/man 0.57 [0.42,0.72]; tree/building 0.55 [0.40,0.70]. Only chair/table and woman/man have CIs excluding 0.5. The 20-pair subset gave a different ranking (e.g., man/woman 0.40, tree 0.70), so small-n flip experiments are unstable; the 40-pair results are the ones to report.

A 2×2 view of same-scene behavior vs regional target transfer (40 pairs; behavior CI vs 0.5):

| | transfer positive | transfer null |
|---|---|---|
| clean AUC CI > 0.5 | chair/table (0.75, R=0.45); woman/man (0.75, R=0.56) | — |
| clean AUC CI includes 0.5 | man/person (0.62, R=0.29); man/woman (0.60, R=0.63); person/man (0.57, R=0.32); tree/building (0.55, R=0.81) | — |

No group has a null transfer. The dissociation is therefore not “behavior vs no transfer” but “positive transfer with or without clear behavioral discrimination.” Regional transfer alone can move the margin without correct source–target ranking; clean behavior can be clear without a large target-region transfer (chair/table). Neither is sufficient for a spatial-mechanism claim.

### 3.4 Relation-type behavior (test)

Spatial left/right (no-split / splitting): chair 0.72/0.79, woman 0.61/0.76, man/woman 0.62/0.73, tree 0.46/0.76; man/person 0.41/0.48, person/man 0.40/0.50.

Above/below (10 pairs/group, 8–12 groups screened):
- no-split top: tree/above/building 0.68, woman/below/man 0.64, tree/below/building 0.62; most others 0.25–0.56.
- splitting top: tree/above/building 0.65, tree/below/building 0.59, fence/below/man 0.56; others ≤0.51.
- same-image vertical flip (20 pairs/group): tree/above/building 0.50, tree/below/building 0.20, woman/below/man 0.40, man/below/woman 0.35. No above/below group is above chance in the minimal counterfactual; this family is not usable for mechanism experiments.

Wearing/on (canonical scene-graph pairs, 20 pairs/group, splitting):
- man/wearing/shirt 0.77, man/wearing/helmet 0.75, woman/wearing/jacket 0.74, sign/on/pole 0.73, fork/on/plate 0.71; man/wearing/pants 0.59.
- This relation family is clearly more separable than above/below.
- Bilateral target ablation (20 pairs): woman/jacket 0.70→0.65/0.60; fork/plate 0.70→0.50/0.45; sign/pole 0.65→0.75/0.75; man/shirt 0.85→0.60/0.55.

### 3.5 Pair-validity control (major)

The loose pair filter used area ≥0.5%, bbox-center margin ≥0.02, and allowed overlapping/tiny objects. This produced many human-ambiguous above/below pairs (e.g., a woman “below” a man when they are side by side; tiny tree/building boxes). Strict filter: area ≥1.5%, IoU ≤0.10, center gap ≥0.12, both source and target.

Strict test pairs: 5,348 (from 33,168), including 436 above and 436 below.

Strict behavior (test, splitting):
- left/right: bus/building 0.88, fence/tree 0.81, man/woman 0.75, man/table 0.72, table/window 0.65, chair/window 0.59.
- above/below: sign/above/building 0.92, tree/above/building 0.92, building/below/tree 0.84, building/below/sign 0.76; chair/table still 0.40.
- strict vertical flip (5 pairs): sign/above/building 0.80, tree/above/building 0.80, building/below/tree 0.80; building/below/sign 0.40.
- strict horizontal flip (10 pairs): man/woman 0.80, bus/building 0.70, man/table 0.70, fence/tree 0.60.

This is a direct cause finding: **pair/label validity explains a substantial part of the earlier weak spatial behavior.** Loose above/below pairs were often not human-valid; after strict filtering, several groups reach 0.76–0.92. The earlier “model cannot do above/below” conclusion is therefore too strong; it is partly a task-construction artifact.

Strict bilateral target ablation (10 pairs): bus/building 0.80→0.80/0.80; sign/above/building 0.86→0.86/0.86; tree/above/building 0.60→0.40/0.40; man/woman 0.40→0.40/0.30. Even on valid pairs, target-region ablation does not consistently reduce discrimination; `sign/above/building` has strong behavior but no target-ablation effect. This means pair validity explains behavior, but not the causal role of the target region.

### 3.6 Probes

Whole-image probe: 0.44–0.55 (near chance).
Raw region probe: 0.66–0.89; position-only 0.86–0.99.
Position-residual probe: woman/man 0.72, man/woman 0.72, tree 0.68, chair 0.65, man/person 0.58 (CI includes 0.5).
Position-matched (threshold 0.3): woman/man 0.55–0.64, man/woman 0.61–0.74, others too few samples.
Interpretation: position explains most of the raw region-probe signal; a weak residual signal remains for some groups, but it is not clean evidence of relation encoding.

### 3.7 Crop/global

Global-block mean replacement: strong target effect (woman −1.35, man/woman −1.67, tree −1.03, chair −0.48).
Crop mean replacement: all crops small (woman −0.35, chair −0.06); global+crops stronger than global alone for woman/man.
Crop activation patch (target is the same-scene horizontal flip): global R 0.2–0.4, crops ≈0, global+crops ≈global; full = 1 by construction.
Conclusion: global block carries most target evidence; crops add modestly; M6 not directly supported.

## 4. Current working conclusion

Semantic attributes: stable margin separation; target-region ablation reduces paired accuracy by ~0.1–0.2, with after-ablation CIs often including 0.5. It contributes, but the evidence is moderate, not a clean knockout. Relation type matters: wearing/on is clearly separable (0.71–0.77 with 20 pairs), left/right is mixed, and loose above/below pairs are at/below chance. However, strict pair filtering recovers 0.76–0.92 for several above/below groups and 0.72–0.88 for left/right groups, so **pair/label validity is a major cause of the earlier weak results**. The earlier “model cannot do above/below” conclusion is withdrawn; the model can do some clean above/below pairs, but the task is fragile and pair-dependent.
Spatial left/right: same-scene flip clean AUC is clearly above chance only for chair/table and woman/man; other groups' CIs include 0.5. Every group shows positive regional target transfer (stable R 0.29–0.81), but transfer is not correlated with clean behavior (r≈−0.17). Regional transfer can move the margin without correct source–target ranking.
Early relation encoding: not established; region probes are largely position-confounded, with weak residual signal for some groups.
Distributed across crops: not supported; the global block carries most target evidence, crops add little.
Failure localization: still open among projection, binding, late composition, readout/calibration. The current result is a failure-factor screen, not a stage localization.
