"""Cross-dataset summary from per-dataset aggregate tables.  python -m cr.compare_datasets --tables main=results/v3_main_tables xdomain_footwear=results/v3_xd_footwear_tables ... --out results/cross_dataset"""
import argparse, os, pandas as pd
from .aggregate import latex
ap = argparse.ArgumentParser(); ap.add_argument("--tables", nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
SHORT = {"openai_clip-vit-base-patch32": "CLIP", "facebook_dinov2-base": "DINOv2"}; rows = []
KEEP = ["Semantic only", "Colour only (HSV)", "Fixed, val-best (mAP)", "Adaptive, val-best (mAP)", "Learned fusion (logistic, fixed form)", "Learned fusion (logistic, query-conditioned)", "Fixed, val-best, CLIP colour-text"]
for kv in a.tables:
    lab, d = kv.split("=", 1); m = pd.read_csv(f"{d}/table_main.csv")
    for _, r in m.iterrows():
        meth = str(r["method"]).replace("$\\lambda$", "l")
        if meth in KEEP: rows.append({"dataset": lab, "encoder": SHORT.get(r["encoder"], r["encoder"]), "method": meth, **{k: r[k] for k in ("mAP", "P@10", "Sem@10", "ColLab@10") if k in m.columns}})
t = pd.DataFrame(rows); t.to_csv(f"{a.out}/table_cross_dataset.csv", index=False); latex(t, f"{a.out}/table_cross_dataset.tex", "Main methods across datasets (test split; mean$\\pm$SD over seeds).", "tab:cross"); print(t.to_string(index=False))
