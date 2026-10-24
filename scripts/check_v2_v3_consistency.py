"""After re-running the main dataset (v3), verify unchanged methods reproduce v2 for the same encoder+seed.
Expected: HSV-based methods identical up to GPU nondeterminism (~1e-3); the Lab-ablation row may differ (v3 fixed its normalisation).
  python scripts/check_v2_v3_consistency.py results/v2 results/v3"""
import sys, glob, json, pandas as pd, numpy as np
v2, v3 = sys.argv[1], sys.argv[2]; num = lambda x: float(str(x).split(" ")[0]); worst = {}
for f2 in sorted(glob.glob(f"{v2}/fs_*/summary.json")):
    s2 = json.load(open(f2)); d2 = f2.rsplit("/", 1)[0]
    for f3 in glob.glob(f"{v3}/main_*/summary.json"):
        s3 = json.load(open(f3)); d3 = f3.rsplit("/", 1)[0]
        if (s2["encoder"], s2["seed"]) != (s3["encoder"], s3["seed"]): continue
        m2, m3 = pd.read_csv(f"{d2}/metrics_main.csv").set_index("method"), pd.read_csv(f"{d3}/metrics_main.csv").set_index("method")
        for meth in m2.index.intersection(m3.index):
            for c in ("mAP", "R@1", "P@10", "Sem@10", "ColLab@10"):
                d = abs(num(m2.loc[meth, c]) - num(m3.loc[meth, c])); worst[(meth.split("(")[0], c)] = max(worst.get(("%s" % meth.split("(")[0], c), 0), d)
        print(f"{s2['encoder'][:12]} seed {s2['seed']}: n_test_queries v2={s2['n_test_queries']} v3={s3['n_test_queries']} (must match)")
t = pd.Series(worst).groupby(level=0).max().round(5); print("\nmax |v2-v3| per method (all metrics):"); print(t.to_string())
