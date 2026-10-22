import numpy as np
from .colour import delta_e00

def average_precision(rel_sorted):
    r = rel_sorted.astype(np.float64); n = r.sum()
    if n == 0: return 0.0
    prec = np.cumsum(r) / (np.arange(len(r)) + 1)
    return float((prec * r).sum() / n)

def evaluate(score_fn, nq, q_id, g_id, q_cat, g_cat, q_lab, g_lab, tau, Ks=(1, 5, 10, 20), chunk=256, q_cl=None, g_cl=None):
    """Per-query metrics. score_fn(slice)->(chunk,G) scores. Identity relevance = same item id.
    Semantic relevance@K = share of top-K with the query's category (independent of colour).
    Colour consistency@K = share of top-K with CIEDE2000(query, result) <= tau (evaluator colour).
    Joint@K = share that is BOTH category-correct and colour-consistent.
    If human colour labels are given: ColLab@K / JointLab@K use label equality instead of pixel dE00."""
    out = {"AP": np.zeros(nq)}
    for K in Ks:
        for n in ("R", "P", "Sem", "Col", "Joint"): out[f"{n}@{K}"] = np.zeros(nq)
        if q_cl is not None:
            out[f"ColLab@{K}"] = np.zeros(nq); out[f"JointLab@{K}"] = np.zeros(nq)
    for s in range(0, nq, chunk):
        sl = slice(s, min(s + chunk, nq)); S = score_fn(sl)
        order = np.argsort(-S, axis=1)
        for j, i in enumerate(range(sl.start, sl.stop)):
            o = order[j]; rel = (g_id[o] == q_id[i])
            out["AP"][i] = average_precision(rel)
            top = o[:max(Ks)]
            de = delta_e00(np.repeat(q_lab[i][None], len(top), 0), g_lab[top])
            colok = de <= tau; semok = (g_cat[top] == q_cat[i])
            for K in Ks:
                out[f"R@{K}"][i] = rel[:K].any(); out[f"P@{K}"][i] = rel[:K].mean()
                out[f"Sem@{K}"][i] = semok[:K].mean(); out[f"Col@{K}"][i] = colok[:K].mean()
                out[f"Joint@{K}"][i] = (semok & colok)[:K].mean()
                if q_cl is not None:
                    cl = (g_cl[top] == q_cl[i]); out[f"ColLab@{K}"][i] = cl[:K].mean(); out[f"JointLab@{K}"][i] = (semok & cl)[:K].mean()
    return out

def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
