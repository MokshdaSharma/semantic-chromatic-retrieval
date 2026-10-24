"""Extra paper tables built ONLY from run files / aggregated tables. Sections whose inputs are absent are skipped
(e.g. robustness for --no_robustness runs).
  python -m cr.paper_tables --runs results/v3/main_* --tables results/v3_tables/main --out paper/tables"""
import argparse, collections, json, os
import numpy as np, pandas as pd
from .aggregate import latex, mclass, NAMES
from .metrics import boot_ci
SHORT = {"openai_clip-vit-base-patch32": "CLIP ViT-B/32", "facebook_dinov2-base": "DINOv2-B"}
PAIRS = [("fixed_best_mAP", "semantic_only", "Fixed(val-best) $-$ Semantic only"), ("fixed_best_mAP", "colour_only", "Fixed(val-best) $-$ Colour only")]
METS = ["AP", "R@1", "P@10", "Sem@10", "ColLab@10"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--tables", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T = a.tables; has = lambda f: os.path.exists(f"{T}/{f}")
    # 1) paired fusion vs single modality (per query, per seed)
    acc = collections.defaultdict(list)
    for d in a.runs:
        if not os.path.exists(f"{d}/summary.json"): continue
        s = json.load(open(f"{d}/summary.json")); enc = SHORT.get(s["encoder"], s["encoder"]); pq = pd.read_csv(f"{d}/per_query.csv"); pq["cls"] = pq["method"].map(mclass)
        for x, y, lab in PAIRS:
            X, Y = pq[pq.cls == x].set_index("qi"), pq[pq.cls == y].set_index("qi")
            for m in METS:
                if m not in X.columns: continue
                mu, lo, hi = boot_ci((X[m] - Y[m].reindex(X.index)).values, n=500, seed=s["seed"]); acc[(enc, lab, m)].append((mu, lo > 0 or hi < 0))
    rows = [{"encoder": e, "comparison": l, "metric": "mAP" if m == "AP" else m, "mean_diff": f"{np.mean([v[0] for v in vs]):+.3f}", "seeds_CI_excl_0": f"{sum(v[1] for v in vs)}/{len(vs)}"} for (e, l, m), vs in acc.items()]
    fv = pd.DataFrame(rows); fv.to_csv(f"{T}/table_fusion_vs_single.csv", index=False)
    latex(fv, f"{a.out}/table_fusion_vs_single.tex", "Fusion minus single modality (paired over test queries; last column = seeds with 95\\% bootstrap CI excluding 0).", "tab:fusion")
    # 2) robustness
    if has("table_robustness.csv"):
        r = pd.read_csv(f"{T}/table_robustness.csv"); m = pd.read_csv(f"{T}/table_main.csv"); inv = {v: k for k, v in NAMES.items()}
        clean = {(x["encoder"], inv.get(x["method"])): float(str(x["P@10"]).split("$")[0]) for _, x in m.iterrows()}
        r = r[r.cls.isin(["colour_only", "semantic_only", "fixed_best_mAP", "adaptive_mAP"])].copy(); met = "JointLab@10"
        r["clean"] = [clean[(e, c)] for e, c in zip(r["enc"], r.cls)]; r["retained"] = r[met] / r["clean"]
        w = r.loc[r.groupby(["enc", "transform", "cls"])["retained"].idxmin()]
        out = w.assign(encoder=w["enc"].map(SHORT), method=w.cls.map(NAMES), worst_level=w.level, retained=w.retained.round(2), abs_P10=w[met].round(3))[["encoder", "transform", "method", "worst_level", "retained", "abs_P10"]]
        out.sort_values(["encoder", "transform", "method"]).to_csv(f"{T}/table_robustness_worst.csv", index=False)
        piv = out.pivot_table(index=["encoder", "transform"], columns="method", values="retained").round(2).reset_index(); piv.columns = [str(c) for c in piv.columns]
        latex(piv, f"{a.out}/table_robustness_worst.tex", "Worst-case share of clean P@10 retained over four severity levels (query-side transforms; hue shift is an intentional colour change, not a nuisance).", "tab:robust")
    # 3) colour group
    if has("table_colour_group.csv"):
        g = pd.read_csv(f"{T}/table_colour_group.csv", header=[0, 1], index_col=[0, 1])
        cg = pd.DataFrame({"ColLab@10 semantic only": g[("ColLab@10", "semantic_only")], "ColLab@10 fixed (val-best)": g[("ColLab@10", "fixed_best_mAP")]}); cg["gain"] = (cg.iloc[:, 1] - cg.iloc[:, 0]).round(3)
        cg = cg.reset_index().rename(columns={"enc": "encoder"}); cg["encoder"] = cg["encoder"].map(SHORT); cg.round(3).to_csv(f"{T}/table_colour_group_gain.csv", index=False)
        latex(cg.round(3), f"{a.out}/table_colour_group_gain.tex", "Human-label colour match@10 by query colour group (achromatic: Black, White, Grey, Silver, Off White, Charcoal, Grey Melange, Cream; grouping fixed in advance).", "tab:group")
    # 4) tau sensitivity
    if has("table_tau_sensitivity.csv"):
        t = pd.read_csv(f"{T}/table_tau_sensitivity.csv"); t["encoder"] = t["enc"].map(SHORT); t["method"] = t.cls.map(NAMES)
        e0 = "CLIP ViT-B/32" if (t.encoder == "CLIP ViT-B/32").any() else t.encoder.iloc[0]; tp = t[t.encoder == e0].pivot_table(index="method", columns="tau_pct", values="Col@10").round(3).reset_index()
        tp.columns = ["method"] + [f"tau@{c}th pct" for c in tp.columns[1:]]; tp.insert(0, "encoder", e0)
        latex(tp, f"{a.out}/table_tau_sensitivity.tex", f"Pixel-based colour consistency@10 versus threshold percentile ({e0} runs; $\\tau$ = percentile of $\\Delta E_{{00}}$ among same-class validation pairs).", "tab:tau")
    # 5) latency / storage
    if has("table_latency.csv"):
        L = pd.read_csv(f"{T}/table_latency.csv"); L["encoder"] = L["encoder"].map(SHORT); L = L.rename(columns=lambda c: c.replace("_per_query", "").replace("_ms", " (ms/query)"))
        latex(L.round(4), f"{a.out}/table_latency.tex", "CPU brute-force scoring latency (excludes encoder inference) and storage per gallery image (bytes, fp32).", "tab:latency")
    for f in ("table_main.tex", "table_paired.tex", "table_metric_agreement.tex"):
        if has(f): os.system(f"cp {T}/{f} {a.out}/{f}")
    os.makedirs(f"{a.out}/../figures", exist_ok=True); os.system(f"cp {T}/figures/*.png {a.out}/../figures/ 2>/dev/null")
    print(fv.to_string(index=False))
if __name__ == "__main__": main()
