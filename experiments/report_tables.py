import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
STAGES = ["S0", "S1", "S6", "S12", "S18"]


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    return "\n".join(out)


def main():
    lines = ["# Experiment summary", ""]
    e1 = {}
    for task in ["spatial", "semantic"]:
        for split in ["dev", "test"]:
            d = load(RES / f"e1/{task}_{split}_split0.json")
            if d:
                e1[(task, split)] = d["summary"]
    rows = []
    for (task, split), summary in e1.items():
        for s in summary:
            rows.append([task, split, s["group"], f"{s['auc']:.3f}", f"{s['mean_delta']:+.2f}",
                         f"{s['mean_delta_image']:+.2f}", f"{s['mean_delta_question']:+.2f}",
                         f"{s['mean_delta_gray']:+.2f}", f"{s['gen_source_acc']:.2f}", f"{s['gen_target_acc']:.2f}",
                         f"{s['margin_source_acc']:.2f}", f"{s['margin_target_acc']:.2f}"])
    lines += ["## E1 clean behavior and input dependence", "",
              table(["task", "split", "group", "AUC", "delta", "img", "question", "gray", "gen yes", "gen no", "margin yes", "margin no"], rows), ""]

    e1ci = {}
    for task in ["spatial", "semantic"]:
        for split in ["dev", "test"]:
            d = load(RES / f"stats/{task}_{split}_e1_summary.json")
            if d:
                e1ci[(task, split)] = d["e1"]
    rows = []
    for (task, split), summary in e1ci.items():
        for r in summary:
            rows.append([task, split, r["group"], f"{r['auc']:.3f}", f"[{r['auc_ci95'][0]:.3f}, {r['auc_ci95'][1]:.3f}]",
                         f"{r['mean_delta']:+.2f}", f"[{r['delta_ci95'][0]:+.2f}, {r['delta_ci95'][1]:+.2f}]", r["n_pairs"]])
    lines += ["## E1 bootstrap confidence intervals", "",
              table(["task", "split", "group", "AUC", "AUC 95% CI", "delta", "delta 95% CI", "n"], rows), ""]

    s0 = load(RES / "e1/spatial_test_split0.json")
    s1 = load(RES / "e1/spatial_test_split1.json")
    if s0 and s1:
        m0 = {r["group"]: r for r in s0["summary"]}
        m1 = {r["group"]: r for r in s1["summary"]}
        rows = []
        for g in sorted(set(m0) | set(m1)):
            rows.append([g, f"{m0[g]['auc']:.3f}", f"{m1[g]['auc']:.3f}",
                         f"{m0[g]['mean_delta']:+.2f}", f"{m1[g]['mean_delta']:+.2f}"])
        lines += ["## E1 resolution ablation (spatial test)", "",
                  table(["group", "AUC no-split", "AUC splitting", "delta no-split", "delta splitting"], rows), ""]

    e2 = {}
    for task in ["spatial", "semantic"]:
        for split in ["dev", "test"]:
            d = load(RES / f"stats/{task}_{split}_e2_summary.json")
            if d:
                e2[(task, split)] = d["e2"]
    rows = []
    for (task, split), summary in e2.items():
        for r in summary:
            if r["region"] != "target_minus_controls":
                continue
            rows.append([task, split, r["group"], r["stage"], f"{r['mean']:+.2f}",
                         f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]", r["n"]])
    lines += ["## E2 target region minus matched controls", "",
              table(["task", "split", "group", "stage", "delta m", "95% CI", "n"], rows), ""]

    es = load(RES / "e2/spatial_test_splitting.json")
    if es:
        rows = []
        for r in es["summary"]:
            rows.append([r["group"], r["n_pairs"], f"{r['n_blocks_mean']:.1f}",
                         f"{r['S1:target']['mean']:+.2f}" if "S1:target" in r else "-",
                         f"{r['S1:reference']['mean']:+.2f}" if "S1:reference" in r else "-",
                         f"{r['S1:background']['mean']:+.2f}" if "S1:background" in r else "-",
                         f"{r['S1:full']['mean']:+.2f}" if "S1:full" in r else "-"])
        lines += ["## E2 splitting-resolution global-block ablation (S1, mean delta m)", "",
                  table(["group", "n", "blocks", "global target", "global reference", "global background", "full block"], rows), ""]

    e3 = {}
    for task in ["spatial", "semantic"]:
        for split in ["dev", "test"]:
            d = load(RES / f"stats/{task}_{split}_summary.json")
            if d and "e3" in d:
                e3[(task, split)] = d["e3"]
    rows = []
    for (task, split), summary in e3.items():
        for r in summary:
            rows.append([task, split, r["group"], r["stage"], f"{r['median_R']:+.2f}",
                         f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]", f"{r['frac_gt_half']:.2f}", r["n"]])
    lines += ["## E3 full visual block recovery", "",
              table(["task", "split", "group", "stage", "median R", "95% CI", "frac R>0.5", "n"], rows), ""]

    reg = load(RES / "e3/spatial_test_regional.json")
    if reg:
        rows = []
        for r in reg["summary"]:
            rows.append([r["group"], r["n_pairs"],
                         f"{r['S1:target']['median']:+.2f}" if r['S1:target']['median'] is not None else "-",
                         f"{r['S1:reference']['median']:+.2f}" if r['S1:reference']['median'] is not None else "-",
                         f"{r['S1:irrelevant']['median']:+.2f}" if r['S1:irrelevant']['median'] is not None else "-",
                         f"{r['S1:random']['median']:+.2f}" if r['S1:random']['median'] is not None else "-",
                         f"{r['S6:target']['median']:+.2f}" if r['S6:target']['median'] is not None else "-",
                         f"{r['S6:reference']['median']:+.2f}" if r['S6:reference']['median'] is not None else "-",
                         f"{r['S6:irrelevant']['median']:+.2f}" if r['S6:irrelevant']['median'] is not None else "-",
                         f"{r['S6:random']['median']:+.2f}" if r['S6:random']['median'] is not None else "-"])
        lines += ["## E3 exploratory regional patching (median R)", "",
                  table(["group", "n", "S1 target", "S1 ref", "S1 irr", "S1 random", "S6 target", "S6 ref", "S6 irr", "S6 random"], rows), ""]

    cov_files = sorted(RES.glob("e2/coverage_*.json"))
    if cov_files:
        rows = []
        for f in cov_files:
            d = json.loads(f.read_text())
            for r in d["summary"]:
                rows.append([f.stem.replace("coverage_", ""), r["region"],
                             f"{r['S1']['mean']:+.2f}", f"{r['S6']['mean']:+.2f}"])
        lines += ["## E2 coverage ablation (mean delta m)", "",
                  table(["case", "region", "S1", "S6"], rows), ""]

    e3s = load(RES / "e3/spatial_test_splitting.json")
    if e3s:
        rows = []
        for r in e3s["summary"]:
            rows.append([r["group"], r["n_pairs"], f"{r['blocks']:.1f}",
                         f"{r['S1']['median']:+.2f}" if r["S1"]["median"] is not None else "-",
                         f"{r['S6']['median']:+.2f}" if r["S6"]["median"] is not None else "-"])
        lines += ["## E3 splitting full-block recovery (median R)", "",
                  table(["group", "n", "blocks", "S1", "S6"], rows), ""]

    probe = load(RES / "e4/probe.json")
    if probe:
        rows = [[r["group"], r["stage"], f"{r['auc_test']:.3f}", f"{r['acc_test']:.3f}", f"{r['majority_test']:.3f}", r["n_train"], r["n_test"]] for r in probe]
        lines += ["## E4 linear probe (source vs target)", "",
                  table(["group", "stage", "AUC", "accuracy", "majority", "n train", "n test"], rows), ""]

    sp = load(RES / "e4/stage_probe.json")
    if sp:
        rows = [[r["group"], r["stage"], f"{r['auc']:.2f}",
                 f"[{r['auc_ci95'][0]:.2f},{r['auc_ci95'][1]:.2f}]" if r["auc_ci95"] else "-",
                 f"{r['random_label_auc']:.2f}", r["n_train"], r["n_test"]] for r in sp]
        lines += ["## E4 stage-wise region probe with position baseline", "",
                  table(["group", "stage", "AUC", "95% CI", "random-label AUC", "n train", "n test"], rows), ""]

    qc = load(RES / "e4/question_context.json")
    if qc:
        rows = []
        for r in qc["summary"]:
            for s in ["S0", "S1", "S6"]:
                rows.append([r["group"], s, f"{r.get(f'own:{s}:target_minus_controls', float('nan')):+.2f}",
                             f"{r.get(f'control:{s}:target_minus_controls', float('nan')):+.2f}"])
        lines += ["## E4 question-context target specificity", "",
                  table(["group", "stage", "own question", "control question"], rows), ""]

    flip = load(RES / "flip/flip_counterfactual.json")
    if flip:
        rows = []
        for r in flip["summary"]:
            rows.append([r["group"], r["n_pairs"], f"{r['auc_clean']:.2f}", f"{r['mean_delta']:+.2f}",
                         f"{r['S1:auc_after_target_ablation']:.2f}", f"{r['S6:auc_after_target_ablation']:.2f}",
                         f"{r['S1:patch_target']['median_R']:+.2f}", f"{r['S6:patch_target']['median_R']:+.2f}",
                         f"{r['S1:patch_random']['median_R']:+.2f}", f"{r['S6:patch_random']['median_R']:+.2f}"])
        lines += ["## Minimal same-image flip counterfactual", "",
                  table(["group", "n", "clean AUC", "delta", "AUC after target abl S1", "S6",
                         "target patch R S1", "S6", "random patch R S1", "S6"], rows), ""]

    joint = load(RES / "e4/joint_ablation.json")
    if joint:
        rows = []
        for r in joint["summary"]:
            for stage in ["S1", "S6", "S12"]:
                if f"{stage}:target" in r:
                    rows.append([r["group"], stage,
                                 f"{r[f'{stage}:target']['mean']:+.2f}",
                                 f"{r[f'{stage}:reference']['mean']:+.2f}",
                                 f"{r[f'{stage}:joint']['mean']:+.2f}",
                                 f"{r[f'{stage}:full']['mean']:+.2f}"])
        lines += ["## E4 joint target+reference ablation (mean delta m)", "",
                  table(["group", "stage", "target", "reference", "joint", "full"], rows), ""]

    sr = load(RES / "e3/semantic_test_regional.json")
    if sr:
        rows = []
        for r in sr["summary"]:
            rows.append([r["group"], r["n_pairs"],
                         f"{r['S1:target']['median']:+.2f}" if r['S1:target']['median'] is not None else "-",
                         f"{r['S1:irrelevant']['median']:+.2f}" if r['S1:irrelevant']['median'] is not None else "-",
                         f"{r['S1:random']['median']:+.2f}" if r['S1:random']['median'] is not None else "-",
                         f"{r['S6:target']['median']:+.2f}" if r['S6:target']['median'] is not None else "-",
                         f"{r['S6:irrelevant']['median']:+.2f}" if r['S6:irrelevant']['median'] is not None else "-",
                         f"{r['S6:random']['median']:+.2f}" if r['S6:random']['median'] is not None else "-"])
        lines += ["## E3 semantic exploratory regional patching (median R)", "",
                  table(["group", "n", "S1 target", "S1 irr", "S1 random", "S6 target", "S6 irr", "S6 random"], rows), ""]

    out = RES / "summary.md"
    out.write_text("\n".join(lines))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
