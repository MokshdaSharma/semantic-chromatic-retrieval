"""Extra paper tables built ONLY from run files / aggregated tables.
  python -m cr.paper_tables --runs results/v2/fs_* --tables results/v2_tables --out paper/tables"""
import argparse, glob, json, os, collections
import numpy as np, pandas as pd
from .aggregate import latex, mclass, NAMES
from .metrics import boot_ci
SHORT = {"openai_clip-vit-base-patch32": "CLIP ViT-B/32", "facebook_dinov2-base": "DINOv2-B"}
PAIRS = [("fixed_best_mAP", "semantic_only", "Fixed(val-best) $-$ Semantic only"), ("fixed_best_mAP", "colour_only", "Fixed(val-best) $-$ Colour only")]
METS = ["AP", "R@1", "P@10", "Sem@10", "ColLab@10"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--tables", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    # 1) paired fusion vs single modality (per query, per seed)
    acc = collections.defaultdict(list)
    for d in a.runs:
        if not os.path.exists(f"{d}/summary.json"): continue
        s = json.load(open(f"{d}/summary.json")); enc = SHORT.get(s["encoder"], s["encoder"]); pq = pd.read_csv(f"{d}/per_query.csv"); pq["cls"] = pq["method"].map(mclass)
        for x, y, lab in PAIRS:
            X, Y = pq[pq.cls == x].set_index("qi"), pq[pq.cls == y].set_index("qi")
            for m in METS:
                col = m if m in X.columns else None
                if col is None: continue
                mu, lo, hi = boot_ci((X[col] - Y[col].reindex(X.index)).values, n=500, seed=s["seed"]); acc[(enc, lab, m)].append((mu, lo > 0 or hi < 0))
    rows = [{"encoder": e, "comparison": l, "metric": "mAP" if m == "AP" else m, "mean_diff": f"{np.mean([v[0] for v in vs]):+.3f}", "seeds_CI_excl_0": f"{sum(v[1] for v in vs)}/{len(vs)}"} for (e, l, m), vs in acc.items()]
    fv = pd.DataFrame(rows); fv.to_csv(f"{a.tables}/table_fusion_vs_single.csv", index=False); latex(fv, f"{a.out}/table_fusion_vs_single.tex", "Fusion minus single modality (paired over test queries; last column = seeds with 95\\% bootstrap CI excluding 0).", "tab:fusion")
    # 2) robustness: worst-case retained share of clean P@10 and absolute value at that level
    r = pd.read_csv(f"{a.tables}/table_robustness.csv"); m = pd.read_csv(f"{a.tables}/table_main.csv")
    clean = {}
    for _, x in m.iterrows():
        inv = {v: k for k, v in NAMES.items()}; cls = inv.get(x["method"]);  clean[(x["encoder"], cls)] = float(str(x["P@10"]).split("$")[0])
    r = r[r.cls.isin(["colour_only", "semantic_only", "fixed_best_mAP", "adaptive_mAP"])].copy(); met = "JointLab@10"
    r["clean"] = [clean[(e, c)] for e, c in zip(r["enc"], r.cls)]; r["retained"] = r[met] / r["clean"]
    w = r.loc[r.groupby(["enc", "transform", "cls"])["retained"].idxmin()]
    out = w.assign(encoder=w["enc"].map(SHORT), method=w.cls.map(NAMES), worst_level=w.level, retained=w.retained.round(2), abs_P10=w[met].round(3))[["encoder", "transform", "method", "worst_level", "retained", "abs_P10"]]
    out.sort_values(["encoder", "transform", "method"]).to_csv(f"{a.tables}/table_robustness_worst.csv", index=False)
    piv = out.pivot_table(index=["encoder", "transform"], columns="method", values="retained").round(2).reset_index(); piv.columns = [str(c) for c in piv.columns]
    latex(piv, f"{a.out}/table_robustness_worst.tex", "Worst-case share of clean P@10 retained over four severity levels (query-side transforms; hue shift is an intentional colour change, not a nuisance).", "tab:robust")
    # 3) colour group
    g = pd.read_csv(f"{a.tables}/table_colour_group.csv", header=[0, 1], index_col=[0, 1])
    cg = pd.DataFrame({"ColLab@10 semantic only": g[("ColLab@10", "semantic_only")], "ColLab@10 fixed (val-best)": g[("ColLab@10", "fixed_best_mAP")]}); cg["gain"] = (cg.iloc[:, 1] - cg.iloc[:, 0]).round(3)
    cg = cg.reset_index().rename(columns={"enc": "encoder"}); cg["encoder"] = cg["encoder"].map(SHORT); cg.round(3).to_csv(f"{a.tables}/table_colour_group_gain.csv", index=False)
    latex(cg.round(3), f"{a.out}/table_colour_group_gain.tex", "Human-label colour match@10 by query colour group (achromatic: Black, White, Grey, Silver, Off White, Charcoal, Grey Melange, Cream; grouping fixed in advance).", "tab:group")
    # 4) tau sensitivity, 5) latency/storage
    t = pd.read_csv(f"{a.tables}/table_tau_sensitivity.csv"); t["encoder"] = t["enc"].map(SHORT); t["method"] = t.cls.map(NAMES)
    tp = t[t.encoder == "CLIP ViT-B/32"].pivot_table(index="method", columns="tau_pct", values="Col@10").round(3).reset_index(); tp.columns = ["method"] + [f"tau@{c}th pct" for c in tp.columns[1:]]
    tp.insert(0, "encoder", "CLIP ViT-B/32"); latex(tp, f"{a.out}/table_tau_sensitivity.tex", "Pixel-based colour consistency@10 versus threshold percentile (CLIP runs; $\\tau$ = percentile of $\\Delta E_{00}$ among same-class validation pairs).", "tab:tau")
    L = pd.read_csv(f"{a.tables}/table_latency.csv"); L["encoder"] = L["encoder"].map(SHORT); L = L.rename(columns=lambda c: c.replace("_per_query", "").replace("_ms", " (ms/query)")); latex(L.round(4), f"{a.out}/table_latency.tex", "CPU brute-force scoring latency (excludes encoder inference) and storage per gallery image (bytes, fp32).", "tab:latency")
    for f in ("table_main.tex", "table_paired.tex", "table_metric_agreement.tex"): os.system(f"cp {a.tables}/{f} {a.out}/{f}")
    os.system(f"cp {a.tables}/figures/*.png {a.out}/../figures/ 2>/dev/null")
    print(fv.to_string(index=False)); print(piv.to_string(index=False))
if __name__ == "__main__": main()
