import argparse, json, math
from pathlib import Path

import torch

from experiments.common import ROOT, build_prompt, load_images, load_model, load_processor, yes_no_ids

CONFIG = ROOT / "data/derived/experiment_config.json"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def margin(model, inputs, yes_ids, no_ids):
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits[:, -1, :]
    lp = torch.log_softmax(logits.float(), dim=-1)
    return (torch.logsumexp(lp[:, yes_ids], dim=-1) - torch.logsumexp(lp[:, no_ids], dim=-1)).item()


def prepare(proc, question, image):
    prompt = build_prompt(proc, question)
    inp = proc(text=prompt, images=[[image]], return_tensors="pt")
    return {k: v.to("cuda") for k, v in inp.items()}


def capture(model, inputs, layer_indices):
    cap = {}
    handles = []

    def out_hook(name):
        def f(mod, inp, out):
            t = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, (tuple, list)) else out)
            cap[name] = t.detach()
        return f

    handles.append(model.model.vision_model.post_layernorm.register_forward_hook(out_hook("S0")))
    handles.append(model.model.connector.register_forward_hook(out_hook("S1")))
    for i in layer_indices:
        def pre(i):
            def f(mod, args, kwargs):
                x = kwargs.get("hidden_states", args[0] if args else None)
                cap[f"S{i}"] = x.detach()
            return f
        handles.append(model.model.text_model.layers[i].register_forward_pre_hook(pre(i), with_kwargs=True))
    with torch.no_grad():
        model(**inputs, use_cache=False)
    for h in handles:
        h.remove()
    return cap


def patch_output(model, inputs, module, tensor, yes_ids, no_ids):
    handle = module.register_forward_hook(lambda mod, inp, out: tensor)
    try:
        return margin(model, inputs, yes_ids, no_ids)
    finally:
        handle.remove()


def patch_lm(model, inputs, layer_idx, v0, v1, src_hs, yes_ids, no_ids):
    layer = model.model.text_model.layers[layer_idx]

    def f(mod, args, kwargs):
        x = kwargs.get("hidden_states", args[0] if args else None)
        x[:, v0:v1, :] = src_hs[:, v0:v1, :]

    handle = layer.register_forward_pre_hook(f, with_kwargs=True)
    try:
        return margin(model, inputs, yes_ids, no_ids)
    finally:
        handle.remove()


def token_box(o, W, H, grid=9):
    x0 = max(0, min(grid - 1, math.floor(o["bbox"][0] / W * grid)))
    x1 = max(0, min(grid - 1, math.ceil((o["bbox"][0] + o["bbox"][2]) / W * grid) - 1))
    y0 = max(0, min(grid - 1, math.floor(o["bbox"][1] / H * grid)))
    y1 = max(0, min(grid - 1, math.ceil((o["bbox"][1] + o["bbox"][3]) / H * grid) - 1))
    return [r * grid + c for r in range(y0, y1 + 1) for c in range(x0, x1 + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["spatial", "semantic"], default="spatial")
    ap.add_argument("--group", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=None)
    ap.add_argument("--splitting", type=int, default=0)
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    task = args.task
    group = args.group or cfg["tasks"][task]["core_groups"][0]
    path = ROOT / cfg["data"][f"{task}_pairs_{args.split}"]
    rows = [r for r in read_jsonl(path) if r["group"] == group]
    if not rows:
        raise SystemExit(f"no pairs for {group}")
    pair = rows[0]

    image_ids = [pair["source_image_id"], pair["target_image_id"]]
    images = load_images(image_ids)
    proc = load_processor(splitting=bool(args.splitting))
    model = load_model(attn="sdpa")
    yes_ids, no_ids = yes_no_ids(proc)

    src_in = prepare(proc, pair["question"], images[pair["source_image_id"]])
    tgt_in = prepare(proc, pair["question"], images[pair["target_image_id"]])
    pos = (src_in["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
    v0, v1 = int(pos[0]), int(pos[-1]) + 1
    n_visual = v1 - v0

    m_s = margin(model, src_in, yes_ids, no_ids)
    m_s2 = margin(model, src_in, yes_ids, no_ids)
    m_t = margin(model, tgt_in, yes_ids, no_ids)

    report = {
        "config_version": cfg["version"],
        "task": task,
        "group": group,
        "pair_id": pair["pair_id"],
        "splitting": bool(args.splitting),
        "input_len": int(src_in["input_ids"].shape[1]),
        "visual_tokens": n_visual,
        "image_positions": [v0, v1 - 1],
        "clean_margin_source": m_s,
        "clean_margin_source_repeat": m_s2,
        "clean_margin_target": m_t,
        "reproducibility_error": abs(m_s - m_s2),
    }

    layer_indices = [6, 12, 18]
    src_cap = capture(model, src_in, layer_indices)
    tgt_cap = capture(model, tgt_in, layer_indices)

    report["shapes"] = {k: list(v.shape) for k, v in src_cap.items()}
    if n_visual == 81:
        W = H = 384
        report["token_regions_9x9"] = {
            "target": token_box(pair["source_target"], W, H, 9),
            "reference": token_box(pair["source_reference"], W, H, 9) if "source_reference" in pair else None,
        }

    # identity patches on source
    ident = {}
    ident["S0"] = patch_output(model, src_in, model.model.vision_model.post_layernorm, src_cap["S0"], yes_ids, no_ids)
    ident["S1"] = patch_output(model, src_in, model.model.connector, src_cap["S1"], yes_ids, no_ids)
    for i in layer_indices:
        ident[f"S{i}"] = patch_lm(model, src_in, i, v0, v1, src_cap[f"S{i}"], yes_ids, no_ids)
    report["identity_patch_margin"] = ident
    report["identity_patch_error"] = {k: abs(v - m_s) for k, v in ident.items()}

    # source -> target full visual block
    full = {}
    full["S0"] = patch_output(model, tgt_in, model.model.vision_model.post_layernorm, src_cap["S0"], yes_ids, no_ids)
    full["S1"] = patch_output(model, tgt_in, model.model.connector, src_cap["S1"], yes_ids, no_ids)
    for i in layer_indices:
        full[f"S{i}"] = patch_lm(model, tgt_in, i, v0, v1, src_cap[f"S{i}"], yes_ids, no_ids)
    report["source_to_target_patch_margin"] = full
    denom = m_s - m_t
    report["recovery_R"] = {k: (v - m_t) / denom if abs(denom) > 1e-8 else None for k, v in full.items()}

    out = Path(args.out) if args.out else ROOT / f"results/e0/{task}_{group.replace('|', '_').replace(' ', '-')}_split{args.splitting}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
