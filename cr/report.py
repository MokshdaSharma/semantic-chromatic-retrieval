"""Figures from saved CSVs only (never hand-entered numbers)."""
import sys, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, os
d = sys.argv[1]; os.makedirs(f"{d}/figures", exist_ok=True)
t = pd.read_csv(f"{d}/tradeoff_sweep.csv")
plt.figure(figsize=(5, 4)); plt.plot(t["Sem@10"], t["Col@10"], "o-")
for _, r in t.iterrows(): plt.annotate(f"{r.lam:.1f}", (r["Sem@10"], r["Col@10"]), fontsize=7)
plt.xlabel("Semantic relevance@10"); plt.ylabel("Colour consistency@10"); plt.title("Fixed-fusion trade-off (labels = lambda)")
plt.tight_layout(); plt.savefig(f"{d}/figures/fig4_tradeoff.png", dpi=200); plt.close()
r = pd.read_csv(f"{d}/robustness.csv"); kinds = r["transform"].unique()
fig, ax = plt.subplots(2, 3, figsize=(13, 6.5))
for a, k in zip(ax.ravel(), kinds):
    for m, g in r[r["transform"] == k].groupby("method"): a.plot(g.level, g["Joint@10"], "o-", label=m[:22])
    a.set_title(k); a.set_xlabel("severity"); a.set_ylabel("Joint@10")
ax[0, 0].legend(fontsize=6); plt.tight_layout(); plt.savefig(f"{d}/figures/fig3_robustness.png", dpi=200)
