"""Aggregate any set of runs (different encoders and/or seeds) into paper tables + figures.
Every number comes from the saved run files; nothing is typed by hand.
  python -m cr.aggregate --runs results/v2/fs_clip_s* results/v2/fs_dinov2_s* --out results/v2_tables
Tables: main, paired (adaptive vs fixed), robustness, metric_agreement, per_colour, tau_sensitivity, latency (CSV + LaTeX)."""
import argparse, json, os, collections
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

MAIN = ["mAP", "R@1", "P@10", "Sem@10", "ColLab@10", "JointLab@10", "Col@10", "Joint@10"]
ORDER = ["colour_only", "semantic_only", "fixed_0.5", "fixed_best_mAP", "adaptive_mAP", "fixed_best_Joint", "adaptive_Joint", "fixed_best_alt_colour"]
NAMES = {"colour_only": "Colour only (HSV)", "semantic_only": "Semantic only", "fixed_0.5": "Fixed, $\\lambda$=0.5",
         "fixed_best_mAP": "Fixed, val-best (mAP)", "adaptive_mAP": "Adaptive, val-best (mAP)",
         "fixed_best_Joint": "Fixed, val-best (Joint)", "adaptive_Joint": "Adaptive, val-best (Joint)",
         "fixed_best_alt_colour": "Fixed, val-best (mAP), Lab hist."}
ACHROMATIC = {"Black", "White", "Grey", "Silver", "Off White", "Charcoal", "Grey Melange", "Cream"}  # grouping fixed before per-colour analysis

def mclass(n):
    if n.startswith("colour_only"): return "colour_only"
    if n == "semantic_only": return "semantic_only"
    if n == "fixed_lambda0.5": return "fixed_0.5"
    if n.startswith("fixed_best_with_"): return "fixed_best_alt_colour"
    if n.startswith("fixed_bestJ("): return "fixed_best_Joint"
    if n.startswith("fixed_best("): return "fixed_best_mAP"
    if n.startswith("adaptiveJ("): return "adaptive_Joint"
    if n.startswith("adaptive("): return "adaptive_mAP"
    return n
num = lambda x: float(str(x).split(" ")[0])
def fmt(v, d=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f}" if len(v) == 1 else f"{v.mean():.{d}f}$\\pm${v.std(ddof=1):.{d}f}"

def latex(df, path, caption, label):
    cols = list(df.columns); esc = lambda t: str(t).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&") if "$" not in str(t) else str(t)
    lines = ["\\begin{table}[t]\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}", "\\resizebox{\\linewidth}{!}{\\begin{tabular}{" + "l" * 2 + "r" * (len(cols) - 2) + "}\\toprule",
             " & ".join(esc(c) for c in cols) + " \\\\ \\midrule"]
    lines += [" & ".join(esc(x) for x in r) + " \\\\" for r in df.astype(str).values.tolist()]
    lines += ["\\bottomrule\\end{tabular}}", "\\end{table}"]; open(path, "w").write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out + "/figures", exist_ok=True)
    runs = []
    for d in a.runs:
        if os.path.exists(f"{d}/summary.json"): s = json.load(open(f"{d}/summary.json")); runs.append({"dir": d, "enc": s["encoder"], "seed": s["seed"], "s": s})
    by = collections.defaultdict(list)
    for r in runs: by[r["enc"]].append(r)
    meta = {e: {"n_runs": len(v), "seeds": sorted(x["seed"] for x in v), "dataset": v[0]["s"]["dataset"], "n_test_queries": [x["s"]["n_test_queries"] for x in v],
                "n_test_gallery": [x["s"]["n_test_gallery"] for x in v]} for e, v in by.items()}
    json.dump(meta, open(f"{a.out}/run_inventory.json", "w"), indent=1)

    # ---- main table
    rows = []
    for e, rs in by.items():
        vals = collections.defaultdict(list)
        for r in rs:
            m = pd.read_csv(f"{r['dir']}/metrics_main.csv")
            for _, row in m.iterrows():
                for k in MAIN:
                    if k in m.columns and pd.notna(row[k]): vals[(mclass(row["method"]), k)].append(num(row[k]))
        for c in ORDER:
            if (c, "mAP") in vals: rows.append({"encoder": e, "method": NAMES[c], **{k: fmt(vals[(c, k)]) for k in MAIN if (c, k) in vals}})
    main = pd.DataFrame(rows); main.to_csv(f"{a.out}/table_main.csv", index=False)
    latex(main[["encoder", "method"] + [k for k in MAIN[:7] if k in main.columns]], f"{a.out}/table_main.tex", "Main comparison on the test split (mean$\\pm$SD over seeds where more than one run exists).", "tab:main")

    # ---- paired adaptive vs fixed (mean diff over seeds, share of seeds whose CI excludes 0)
    prow = []
    for e, rs in by.items():
        for key in [k for k in rs[0]["s"] if k.startswith("adaptive") and "_minus_" in k]:
            acc = collections.defaultdict(list)
            for r in rs:
                for m, (mu, lo, hi) in r["s"].get(key, {}).items(): acc[m].append((mu, lo > 0 or hi < 0))
            for m, l in acc.items():
                prow.append({"encoder": e, "comparison": key.split(" (")[0], "metric": m, "mean_diff": f"{np.mean([x[0] for x in l]):+.4f}", "seeds_CI_excl_0": f"{sum(x[1] for x in l)}/{len(l)}"})
    paired = pd.DataFrame(prow); paired.to_csv(f"{a.out}/table_paired.csv", index=False)
    if len(paired): latex(paired, f"{a.out}/table_paired.tex", "Adaptive minus fixed-weight fusion (paired over test queries; last column = seeds whose 95\\% bootstrap CI excludes 0). Seven metrics, no multiplicity correction.", "tab:paired")

    # ---- robustness
    rb = []
    for e, rs in by.items():
        for r in rs:
            f = f"{r['dir']}/robustness.csv"
            if os.path.exists(f): x = pd.read_csv(f); x["enc"], x["seed"], x["cls"] = e, r["seed"], x["method"].map(mclass); rb.append(x)
    if rb:
        rb = pd.concat(rb); metric = "JointLab@10" if "JointLab@10" in rb.columns else "Joint@10"
        g = rb.groupby(["enc", "transform", "level", "cls"])[["mAP", metric]].mean().reset_index(); g.to_csv(f"{a.out}/table_robustness.csv", index=False)
        for e in g["enc"].unique():
            fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
            for axx, k in zip(ax.ravel(), g["transform"].unique()):
                for c, gg in g[(g["enc"] == e) & (g["transform"] == k)].groupby("cls"):
                    if c in ("fixed_best_mAP", "adaptive_mAP", "semantic_only", "colour_only"): axx.plot(gg.level, gg[metric], "o-", label=NAMES[c].replace("$\\lambda$", "l"))
                axx.set_title(k); axx.set_xlabel("severity"); axx.set_ylabel(metric)
            ax[0, 0].legend(fontsize=6); plt.suptitle(f"Robustness (query-side), {e}"); plt.tight_layout(); plt.savefig(f"{a.out}/figures/fig3_robustness_{e}.png", dpi=200); plt.close()

    # ---- trade-off curves
    fig, ax = plt.subplots(figsize=(5.5, 4.2)); ylab = set()
    pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ci, (e, rs) in enumerate(by.items()):
        for i, r in enumerate(rs):
            t = pd.read_csv(f"{r['dir']}/tradeoff_sweep.csv"); y = "ColLab@10" if "ColLab@10" in t else "Col@10"; ylab.add(y)
            ax.plot(t["Sem@10"], t[y], "o-", ms=2.5, lw=1, alpha=.55, color=pal[ci], label=f"{e} ({len(rs)} seeds)" if i == 0 else None)
    ax.set_xlabel("Semantic relevance@10 (category)"); ax.set_ylabel("Colour match@10 (human label)" if ylab == {"ColLab@10"} else "Colour consistency@10 (pixel dE00 <= tau)" if ylab == {"Col@10"} else "Colour metric@10 (mixed runs: check inputs)"); ax.legend(fontsize=7); ax.set_title("Fixed fusion: lambda 0 -> 1")
    plt.tight_layout(); plt.savefig(f"{a.out}/figures/fig4_tradeoff.png", dpi=200); plt.close()

    # ---- metric validity (AUC of each score for predicting colour-label agreement)
    ag = [pd.read_csv(f"{r['dir']}/metric_agreement.csv").assign(enc=e) for e, rs in by.items() for r in rs if os.path.exists(f"{r['dir']}/metric_agreement.csv")]
    if ag:
        ag = pd.concat(ag); t = ag.groupby(["enc", "scope", "score"])["auc"].agg(lambda x: fmt(x)).reset_index(); t.to_csv(f"{a.out}/table_metric_agreement.csv", index=False)
        latex(t, f"{a.out}/table_metric_agreement.tex", "ROC-AUC of each score for predicting agreement of the human colour label on random query--gallery pairs.", "tab:agreement")

    # ---- per-colour analysis (pooled over seeds)
    pcq = []
    for e, rs in by.items():
        for r in rs:
            f = f"{r['dir']}/per_query.csv"
            if os.path.exists(f): x = pd.read_csv(f); x["enc"], x["cls"] = e, x["method"].map(mclass); pcq.append(x)
    if pcq:
        pq = pd.concat(pcq); pq = pq[pq.cls.isin(["semantic_only", "fixed_0.5", "fixed_best_mAP", "adaptive_mAP"]) & pq.colour_label.notna() & (pq.colour_label != "")]
        pq["colour_group"] = np.where(pq.colour_label.isin(ACHROMATIC), "achromatic", "chromatic")
        pc = pq.pivot_table(index=["enc", "colour_label"], columns="cls", values="ColLab@10", aggfunc="mean"); pc["n_queries"] = pq[pq.cls == "semantic_only"].groupby(["enc", "colour_label"]).size()
        pc[pc.n_queries >= 15].round(3).reset_index().to_csv(f"{a.out}/table_per_colour.csv", index=False)
        pg = pq.pivot_table(index=["enc", "colour_group"], columns="cls", values=["ColLab@10", "AP"], aggfunc="mean").round(3); pg.to_csv(f"{a.out}/table_colour_group.csv")

    # ---- tau sensitivity + latency
    ts = [pd.read_csv(f"{r['dir']}/tau_sensitivity.csv").assign(enc=e) for e, rs in by.items() for r in rs if os.path.exists(f"{r['dir']}/tau_sensitivity.csv")]
    if ts:
        ts = pd.concat(ts); ts["cls"] = ts.method.map(mclass)
        ts.groupby(["enc", "cls", "tau_pct"])[["tau_dE00", "Col@10", "Joint@10"]].mean().round(3).reset_index().to_csv(f"{a.out}/table_tau_sensitivity.csv", index=False)
    lat = []
    for e, rs in by.items():
        for r in rs:
            l = json.load(open(f"{r['dir']}/latency.json")); lat.append({"encoder": e, **{k: v for k, v in l.items() if k.endswith("per_query")}, **{k: v for k, v in l["storage_bytes_per_gallery_image"].items()}})
    L = pd.DataFrame(lat).groupby("encoder").mean().round(4).reset_index(); L.to_csv(f"{a.out}/table_latency.csv", index=False)
    print("runs:", json.dumps(meta)); print(main.to_string(index=False))

if __name__ == "__main__": main()
