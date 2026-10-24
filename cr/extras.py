"""Extra baselines + qualitative/failure analysis.
 (a) CLIP colour-text prompts  : zero-shot colour distribution per image from a FIXED, dataset-independent colour vocabulary.
 (b) Learned fusion            : tiny logistic model trained on VALIDATION pairs only (fixed-weight form and query-conditioned form).
 (c) qualitative() / failure_analysis(): example grids and error taxonomy (needs images -> run where the dataset is available)."""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from . import fusion as U

# Generic basic-colour vocabulary, chosen BEFORE looking at any dataset labels (no leakage of dataset colour classes).
COLOUR_VOCAB = ["black", "white", "grey", "red", "orange", "yellow", "green", "blue", "navy blue", "purple", "pink", "brown",
                "beige", "maroon", "teal", "gold", "silver", "cream"]
PROMPT = "a photo of a {} product"

def clip_colour_probs(sem, text_emb, logit_scale=100.0):
    """Zero-shot colour distribution p(c|x)=softmax(scale*cos(img,text_c)); returns sqrt(p) so a dot product is the
    Bhattacharyya coefficient (same convention as the histogram descriptors). sem, text_emb are L2-normalised."""
    z = logit_scale * sem @ text_emb.T; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    return np.sqrt(p).astype(np.float32)

def query_ctx(zs, zc, eval_lab, pool=50):
    """Query-level context features (computable at query time): within-pool score dispersions and query chroma."""
    ss, sc = U.pool_stats(zs, zc, pool)
    return {"ss": ss, "sc": sc, "ch": np.hypot(eval_lab[:, 1], eval_lab[:, 2])}

class LearnedFusion:
    """Logistic pairwise relevance model on z-normalised similarities.
    learned_fixed : terms [zs, zc, zs*zc]                      (query-independent weights, learned)
    learned_qcond : + zs*sigma_s(q), zc*sigma_c(q), zs*chroma(q), zc*chroma(q)   (weights depend on the query)
    Trained on validation query-gallery pairs only (hard + random candidates); C fixed in advance."""
    def __init__(self, qcond, C=1.0, n_top=50, n_rand=50, seed=0): self.qcond, self.C, self.n_top, self.n_rand, self.seed = qcond, C, n_top, n_rand, seed
    def _terms(self, zs, zc, ss, sc, ch):
        t = [zs, zc, zs * zc]
        if self.qcond: t += [zs * ss, zc * sc, zs * ch, zc * ch]
        return t
    def _std_ctx(self, ctx):
        return {k: (v - self.mu[k]) / self.sd[k] for k, v in ctx.items()}
    def fit(self, zs, zc, ctx, rel):
        rng = np.random.default_rng(self.seed); nq, G = zs.shape
        self.mu = {k: float(v.mean()) for k, v in ctx.items()}; self.sd = {k: float(v.std() + 1e-8) for k, v in ctx.items()}
        c = self._std_ctx(ctx); Xs, ys = [], []
        for i in range(nq):
            cand = np.unique(np.concatenate([np.argpartition(-zs[i], min(self.n_top, G - 1))[: self.n_top], np.argpartition(-zc[i], min(self.n_top, G - 1))[: self.n_top],
                                             rng.integers(0, G, self.n_rand)]))
            t = self._terms(zs[i, cand], zc[i, cand], c["ss"][i], c["sc"][i], c["ch"][i]); Xs.append(np.stack(t, 1)); ys.append(rel[i, cand])
        X, y = np.concatenate(Xs), np.concatenate(ys)
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-8
        self.clf = LogisticRegression(C=self.C, max_iter=1000).fit((X - self.xm) / self.xs, y)
        self.coef_ = self.clf.coef_[0] / self.xs; self.n_pairs, self.pos_rate = len(y), float(y.mean()); return self
    def score(self, zs, zc, ctx):
        c = self._std_ctx(ctx); col = lambda v: v[:, None]
        t = self._terms(zs, zc, col(c["ss"]), col(c["sc"]), col(c["ch"]))
        return sum(w * x for w, x in zip(self.coef_, t))
    def describe(self):
        names = ["z_sem", "z_chr", "z_sem*z_chr"] + (["z_sem*sigma_s", "z_chr*sigma_c", "z_sem*chroma", "z_chr*chroma"] if self.qcond else [])
        return {n: float(w) for n, w in zip(names, self.coef_)}

def topk_all(fn, nq, k=10, chunk=256):
    out = np.zeros((nq, k), int)
    for s in range(0, nq, chunk):
        S = fn(slice(s, min(s + chunk, nq))); idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        o = np.take_along_axis(S, idx, 1).argsort(1)[:, ::-1]; out[s:s + len(S)] = np.take_along_axis(idx, o, 1)
    return out

def failure_analysis(ds, tq, tg, fns, per_q, ref, base, out_dir, k=10):
    """Error taxonomy of top-k slots, colour-label confusions, and per-category P@10. Saves CSVs."""
    os.makedirs(out_dir, exist_ok=True); nq = len(tq); rows = []; conf = None
    qcat, gcat, qid, gid = ds.cats[tq], ds.cats[tg], ds.ids[tq], ds.ids[tg]
    qcol = ds.clabels[tq] if ds.clabels is not None else None; gcol = ds.clabels[tg] if ds.clabels is not None else None
    for name, fn in fns.items():
        top = topk_all(fn, nq, k); rel = gid[top] == qid[:, None]; cat_ok = gcat[top] == qcat[:, None]
        col_ok = (gcol[top] == qcol[:, None]) if qcol is not None else rel
        r = {"method": name, "relevant": rel.mean(), "wrong_colour_right_category": (cat_ok & ~col_ok).mean(),
             "wrong_category_right_colour": (~cat_ok & col_ok).mean(), "both_wrong": (~cat_ok & ~col_ok).mean()}
        rows.append(r)
        if name == ref and qcol is not None:
            m = cat_ok & ~col_ok; qi, ri = np.where(m); conf = pd.DataFrame({"query_colour": qcol[qi], "retrieved_colour": gcol[top[qi, ri]]})
    pd.DataFrame(rows).round(4).to_csv(f"{out_dir}/failure_taxonomy.csv", index=False)
    if conf is not None and len(conf):
        qn = pd.Series(qcol).value_counts(); c = conf.groupby(["query_colour", "retrieved_colour"]).size().reset_index(name="count")
        c["share_of_slots_for_query_colour"] = c["count"] / (c.query_colour.map(qn) * k); c.sort_values("count", ascending=False).head(40).round(4).to_csv(f"{out_dir}/colour_confusions_{ref[:12]}.csv", index=False)
    cat = pd.DataFrame({"cat": qcat, "base": per_q[base]["P@10"], "ref": per_q[ref]["P@10"]}).groupby("cat").agg(n=("base", "size"), P10_base=("base", "mean"), P10_ref=("ref", "mean"))
    cat["diff"] = cat.P10_ref - cat.P10_base; cat[cat.n >= 10].sort_values("diff").round(4).to_csv(f"{out_dir}/per_category_{base[:8]}_vs_{ref[:10]}.csv")

def qualitative(ds, tq, tg, fns, per_q, ref, base, out_dir, n_random=8, n_each=6, k=4, seed=0):
    """Image grids: random queries, biggest wins and losses of `ref` vs `base`. Green border = relevant (same type+colour label)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True); rng = np.random.default_rng(seed); nq = len(tq)
    d = per_q[ref]["P@10"] - per_q[base]["P@10"]
    sets = {"random": rng.choice(nq, min(n_random, nq), replace=False), "wins": np.argsort(-d)[:n_each], "losses": np.argsort(d)[:n_each]}
    names = list(fns); meta = []
    for tag, qs in sets.items():
        fig, ax = plt.subplots(len(qs), 1 + k * len(names), figsize=(1.0 * (1 + k * len(names)), 1.25 * len(qs)), squeeze=False)
        for r, qi in enumerate(qs):
            a0 = ax[r, 0]; a0.imshow(ds.load(tq[qi])); a0.set_xticks([]); a0.set_yticks([])
            a0.set_ylabel(f"{ds.cats[tq[qi]]}\n{ds.clabels[tq[qi]] if ds.clabels is not None else ''}", fontsize=5, rotation=0, ha="right", va="center")
            if r == 0: a0.set_title("query", fontsize=6)
            meta.append({"set": tag, "query_index": int(qi), "dataset_index": int(tq[qi]), "category": ds.cats[tq[qi]], "colour": ds.clabels[tq[qi]] if ds.clabels is not None else "", "P10_diff_ref_minus_base": float(d[qi])})
            for m, name in enumerate(names):
                top = topk_all(lambda sl, f=fns[name], q=qi: f(slice(q, q + 1)), 1, k)[0]
                for j, g in enumerate(top):
                    a = ax[r, 1 + m * k + j]; a.imshow(ds.load(tg[g])); a.set_xticks([]); a.set_yticks([])
                    ok = ds.ids[tg[g]] == ds.ids[tq[qi]]
                    for sp in a.spines.values(): sp.set_edgecolor("green" if ok else "red"); sp.set_linewidth(2)
                    if r == 0 and j == 0: a.set_title(name[:22], fontsize=6, loc="left")
        plt.tight_layout(pad=0.2); plt.savefig(f"{out_dir}/qualitative_{tag}.png", dpi=170); plt.close()
    pd.DataFrame(meta).to_csv(f"{out_dir}/qualitative_queries.csv", index=False)
