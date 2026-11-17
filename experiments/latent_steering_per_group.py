"""Per-group source-target latent steering."""

import argparse, collections, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, load_images, load_model, load_processor, yes_no_ids
from experiments.e0_validate import prepare
from experiments.latent_steering import GROUPS, margin_from_logits

def read_jsonl(path):
    with open(path) as f: return [json.loads(l) for l in f if l.strip()]

def capture(model, inp, layer, yes_ids, no_ids):
    box={}
    def f(mod,args,kwargs):
        x=kwargs.get("hidden_states",args[0] if args else None); box["h"]=x[:,-1,:].detach().clone()
    h=model.model.text_model.layers[layer].register_forward_pre_hook(f,with_kwargs=True)
    try:
        with torch.no_grad(): out=model(**inp,use_cache=False)
    finally: h.remove()
    return box["h"], margin_from_logits(out.logits[:,-1,:],yes_ids,no_ids)

def steer(model, inp, layer, d, alpha, yes_ids, no_ids):
    def f(mod,args,kwargs):
        x=kwargs.get("hidden_states",args[0] if args else None); x[:,-1,:]+=alpha*d
    h=model.model.text_model.layers[layer].register_forward_pre_hook(f,with_kwargs=True)
    try:
        with torch.no_grad(): out=model(**inp,use_cache=False)
        return margin_from_logits(out.logits[:,-1,:],yes_ids,no_ids)
    finally: h.remove()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--layer",type=int,default=6); ap.add_argument("--out",required=True); args=ap.parse_args()
    dev=[r for r in read_jsonl(ROOT/"data/derived/spatial_pairs_dev.jsonl") if r["group"] in GROUPS]
    test=[r for r in read_jsonl(ROOT/"data/derived/spatial_pairs_test.jsonl") if r["group"] in GROUPS]
    bydev=collections.defaultdict(list); bytest=collections.defaultdict(list)
    for r in dev: bydev[r["group"]].append(r)
    for r in test: bytest[r["group"]].append(r)
    dev={g:v[:10] for g,v in bydev.items()}; test={g:v[:10] for g,v in bytest.items()}
    images=load_images(sorted({r[k] for g in dev for r in dev[g] for k in ('source_image_id','target_image_id')} | {r[k] for g in test for r in test[g] for k in ('source_image_id','target_image_id')}))
    proc=load_processor(splitting=False); model=load_model(attn="sdpa"); yes,no=yes_no_ids(proc)
    out={}
    for g in GROUPS:
        diffs=[]
        for r in dev[g]:
            hs,_=capture(model,prepare(proc,r["question"],images[r["source_image_id"]]),args.layer,yes,no)
            ht,_=capture(model,prepare(proc,r["question"],images[r["target_image_id"]]),args.layer,yes,no)
            diffs.append((hs-ht).squeeze(0))
        norm=float(np.mean([d.norm().item() for d in diffs])); d=torch.stack(diffs).mean(0); d=d/d.norm()
        rng=np.random.default_rng(0); dr=torch.tensor(rng.normal(size=d.shape),dtype=d.dtype,device=d.device); dr=dr/dr.norm()
        perm=torch.tensor(rng.permutation(len(diffs))); ds=torch.stack(diffs)[perm].mean(0); ds=ds/ds.norm()
        rows=[]
        for r in test[g]:
            ins=prepare(proc,r["question"],images[r["source_image_id"]]); int_=prepare(proc,r["question"],images[r["target_image_id"]])
            ms=margin_from_logits(model(**ins,use_cache=False).logits[:,-1,:],yes,no)
            mt=margin_from_logits(model(**int_,use_cache=False).logits[:,-1,:],yes,no)
            mt_d=steer(model,int_,args.layer,d,norm,yes,no)
            mt_r=steer(model,int_,args.layer,dr,norm,yes,no)
            mt_s=steer(model,int_,args.layer,ds,norm,yes,no)
            rows.append({"clean":(ms,mt),"target_plus":(ms,mt_d),"random":(ms,mt_r),"shuffled":(ms,mt_s)})
        def acc(key):
            from sklearn.metrics import roc_auc_score
            y=[];s=[]
            for x in rows:
                a,b=x[key]; y += [1,0]; s += [a,b]
            return {"pair_acc":float(np.mean([x[key][0]>x[key][1] for x in rows])),"auc":float(roc_auc_score(y,s)),"mean_delta":float(np.mean([x[key][0]-x[key][1] for x in rows]))}
        out[g]={k:acc(k) for k in ["clean","target_plus","random","shuffled"]}
        out[g]["norm"]=norm
        print(g,out[g])
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(out,indent=2)); print('saved',args.out)

if __name__=="__main__": main()
