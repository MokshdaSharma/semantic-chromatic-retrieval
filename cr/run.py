"""End-to-end experiment. Usage:
  python -m cr.run --dataset synthetic --encoder hog --out results/smoke
  python -m cr.run --dataset deepfashion --root /data/DeepFashion/InShop --encoder clip --out results/deepfashion_clip
"""
import argparse, json, os, sys, time, itertools, numpy as np, pandas as pd, yaml
from tqdm import tqdm
from . import data as D, features as F, colour as C, fusion as U, metrics as M, robustness as R, extras as X

def feats(ds, idx, enc, kinds, cache, tag, transform=None, rng=None):
    """Cached extraction. Returns dict: sem, hsv, lab, evalLab."""
    p = os.path.join(cache, f"{tag}.npz")
    if os.path.exists(p) and transform is None:
        z = np.load(p); return {k: z[k] for k in z.files}
    imgs = []
    for i in tqdm(idx, desc=tag, leave=False, disable=not sys.stderr.isatty()):
        im = ds.load(i); imgs.append(transform(im) if transform else im)
    out = {"sem": enc.encode(imgs)}
    for k in kinds: out[k] = np.stack([C.hist_descriptor(i, k) for i in imgs])
    out["evalLab"] = np.stack([C.fg_lab(i) for i in imgs])
    if transform is None: np.savez_compressed(p, **out)
    return out

def keys(r):
    return [k for k in ["AP", "R@1", "R@10", "P@10", "Sem@10", "Col@10", "Joint@10", "ColLab@10", "JointLab@10", "Col@1", "Joint@1"] if k in r]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic"); ap.add_argument("--root")
    ap.add_argument("--encoder", default="hog"); ap.add_argument("--hf_model")
    ap.add_argument("--out", required=True); ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--max_test_queries", type=int, default=None); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no_robustness", action="store_true"); ap.add_argument("--no_extras", action="store_true"); ap.add_argument("--qualitative", action="store_true")
    ap.add_argument("--name", default=None); ap.add_argument("--source_master", default=None); ap.add_argument("--target_master", default=None)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); seed = a.seed if a.seed is not None else cfg["seed"]
    os.makedirs(a.out, exist_ok=True); cache = os.path.join(a.out, "cache"); os.makedirs(cache, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---- data & splits -------------------------------------------------
    if a.dataset in ("synthetic", "fashion_small", "fashion_xdomain", "generic_labeled"):
        if a.dataset == "synthetic": ds = D.synthetic(seed)
        elif a.dataset == "fashion_small": ds = D.fashion_product(a.root, seed, **cfg.get("fashion_small", {}))
        elif a.dataset == "fashion_xdomain":
            kw = dict(cfg["fashion_xdomain"]); kw["source_master"] = a.source_master.split(",") if a.source_master else kw["source_master"]
            kw["target_master"] = a.target_master.split(",") if a.target_master else kw["target_master"]
            ds = D.fashion_product(a.root, seed, name="xdomain_" + "+".join(kw["source_master"]) + "_to_" + "+".join(kw["target_master"]), **kw)
        else: ds = D.fashion_product(a.root, seed, **{**cfg.get("generic_labeled", {}), "name": a.name or os.path.basename(a.root.rstrip("/"))})
        tq, tg = ds.idx("test_query"), ds.idx("test_gallery"); vq, vg = ds.idx("val_query"), ds.idx("val_gallery")
    else:
        ds = D.deepfashion_inshop(a.root); tq, tg = ds.idx("query"), ds.idx("gallery")
        tr = ds.idx("train"); u = np.unique(ds.ids[tr]); rng.shuffle(u); vids = set(u[: int(len(u) * cfg["val_identity_frac"])].tolist())
        vm = np.array([i in vids for i in ds.ids[tr]]); vsel = tr[vm]
        # per validation identity: first image -> query, rest -> gallery (identities disjoint from test)
        _, first = np.unique(ds.ids[vsel], return_index=True); isq = np.zeros(len(vsel), bool); isq[first] = True
        vq, vg = vsel[isq], vsel[~isq]
    if a.max_test_queries and len(tq) > a.max_test_queries: tq = rng.choice(tq, a.max_test_queries, replace=False)
    enc = F.HOGStandIn() if a.encoder == "hog" else F.FakeCLIP() if a.encoder == "fakeclip" else F.HFEncoder(a.encoder, a.hf_model)
    kinds = ["hsv", "lab"]; tagp = f"{ds.name}_{enc.name}_s{seed}"
    fv_q = feats(ds, vq, enc, kinds, cache, tagp + "_vq"); fv_g = feats(ds, vg, enc, kinds, cache, tagp + "_vg")
    ft_q = feats(ds, tq, enc, kinds, cache, tagp + "_tq"); ft_g = feats(ds, tg, enc, kinds, cache, tagp + "_tg")

    cdesc = cfg["colour_descriptor"]
    has_text = (not a.no_extras) and getattr(enc, "encode_text", None) is not None
    if has_text:   # CLIP colour-text prompts: fixed generic vocabulary, no dataset labels used
        T = enc.encode_text([X.PROMPT.format(c) for c in X.COLOUR_VOCAB])
        for f_ in (fv_q, fv_g, ft_q, ft_g): f_["txtcol"] = X.clip_colour_probs(f_["sem"], T, enc.logit_scale)
    def sims(fq, fg, kind): return fq["sem"] @ fg["sem"].T, fq[kind] @ fg[kind].T
    # ---- normalisation fitted on validation pairs only, ONE per colour descriptor ----
    # (v2 z-scored the Lab ablation with HSV statistics; now each descriptor has its own validation statistics)
    sv_s = fv_q["sem"] @ fv_g["sem"].T
    norms = {k_: U.Norm(sv_s, fv_q[k_] @ fv_g[k_].T) for k_ in ("hsv", "lab", "txtcol") if k_ in fv_q}; norm = norms[cdesc]

    # ---- evaluator-side colour threshold tau (from val, same-identity pairs) ----
    vqid, vgid = ds.ids[vq], ds.ids[vg]
    same = vqid[:, None] == vgid[None]
    de = np.array([C.delta_e00(fv_q["evalLab"][i][None], fv_g["evalLab"][np.where(same[i])[0]]).ravel() for i in range(len(vq)) if same[i].any()], dtype=object)
    de = np.concatenate(list(de)); tau = float(np.percentile(de, cfg["tau_percentile"]))

    learned = {}
    def score_fn(fq, fg, method, kind):
        S_s, S_c = sims(fq, fg, kind); nm = norms[kind]; zs, zc = nm.sem(S_s), nm.chr(S_c); t = method["type"]
        if t == "sem": return lambda sl: zs[sl]
        if t == "col": return lambda sl: zc[sl]
        if t == "fixed": return lambda sl: U.fixed(zs[sl], zc[sl], method["lam"])
        if t == "adaptive": return lambda sl: U.adaptive(zs[sl], zc[sl], method["lam0"], method["gamma"], method["pool"])[0]
        if t in ("learned_fixed", "learned_qcond"):
            return lambda sl: learned[t].score(zs[sl], zc[sl], X.query_ctx(zs[sl], zc[sl], fq["evalLab"][sl]))
        raise ValueError(t)

    def make_eval(fq, fg, qi, gi, method, kind=cdesc, tau_=None):
        kind = method.get("kind", kind); fn = score_fn(fq, fg, method, kind)
        return M.evaluate(fn, len(qi), ds.ids[qi], ds.ids[gi], ds.cats[qi], ds.cats[gi], fq["evalLab"], fg["evalLab"], (tau if tau_ is None else tau_),
                          q_cl=None if ds.clabels is None else ds.clabels[qi], g_cl=None if ds.clabels is None else ds.clabels[gi])

    # ---- hyper-parameter selection on VALIDATION only -----------------
    # Two PRE-DECLARED selection objectives (validation only). Primary = identity/class mAP (protocol v1);
    # secondary = joint category+human-colour precision@10 (protocol v2, added after a first exploratory run).
    sels = ["AP", "JointLab@10" if ds.clabels is not None else "Joint@10"]
    def v(method, kind=cdesc):
        r = make_eval(fv_q, fv_g, vq, vg, method, kind); return {k: float(r[k].mean()) for k in sels}
    grid_f = {l: v({"type": "fixed", "lam": l}) for l in cfg["lambda_grid"]}
    grid_a = {k: v({"type": "adaptive", "lam0": k[0], "gamma": k[1], "pool": k[2]})
              for k in itertools.product(cfg["lam0_grid"], cfg["gamma_grid"], cfg["pool_grid"])}
    best = lambda g, k_: max(g, key=lambda x: g[x][k_])
    lam_star, a_star = best(grid_f, sels[0]), best(grid_a, sels[0])
    lamJ, aJ = best(grid_f, sels[1]), best(grid_a, sels[1])
    pd.DataFrame([{"lam": l, **g} for l, g in grid_f.items()]).to_csv(f"{a.out}/val_fixed_grid.csv", index=False)
    pd.DataFrame([{"lam0": k[0], "gamma": k[1], "pool": k[2], **g} for k, g in grid_a.items()]).to_csv(f"{a.out}/val_adaptive_grid.csv", index=False)

    # ---- extra baselines (trained / tuned on VALIDATION only) ----
    extra_methods = {}
    if not a.no_extras:
        zs_v, zc_v = norm.sem(sv_s), norm.chr(fv_q[cdesc] @ fv_g[cdesc].T); rel_v = ds.ids[vq][:, None] == ds.ids[vg][None]
        ctx_v = X.query_ctx(zs_v, zc_v, fv_q["evalLab"]); coefs = {}
        for t_, qc in (("learned_fixed", False), ("learned_qcond", True)):
            learned[t_] = X.LearnedFusion(qc, seed=seed).fit(zs_v, zc_v, ctx_v, rel_v); coefs[t_] = learned[t_].describe(); extra_methods[t_] = {"type": t_}
        json.dump(coefs, open(f"{a.out}/learned_fusion_coefficients.json", "w"), indent=1)
        if has_text:
            grid_t = {l: v({"type": "fixed", "lam": l, "kind": "txtcol"}) for l in cfg["lambda_grid"]}; lam_t = best(grid_t, sels[0])
            extra_methods["cliptext_colour_only"] = {"type": "col", "kind": "txtcol"}
            extra_methods[f"fixed_best_cliptext(lam={lam_t})"] = {"type": "fixed", "lam": lam_t, "kind": "txtcol"}
            pd.DataFrame([{"lam": l, **g_} for l, g_ in grid_t.items()]).to_csv(f"{a.out}/val_cliptext_grid.csv", index=False)

    # ---- TEST ----------------------------------------------------------
    methods = {
        f"colour_only_{cdesc}": {"type": "col"}, "semantic_only": {"type": "sem"},
        "fixed_lambda0.5": {"type": "fixed", "lam": 0.5},
        f"fixed_best(lam={lam_star})": {"type": "fixed", "lam": lam_star},
        f"adaptive(lam0={a_star[0]},g={a_star[1]},N={a_star[2]})": {"type": "adaptive", "lam0": a_star[0], "gamma": a_star[1], "pool": a_star[2]},
        f"fixed_bestJ(lam={lamJ})": {"type": "fixed", "lam": lamJ},
        f"adaptiveJ(lam0={aJ[0]},g={aJ[1]},N={aJ[2]})": {"type": "adaptive", "lam0": aJ[0], "gamma": aJ[1], "pool": aJ[2]},
        **extra_methods,
    }
    per_q, rows = {}, []
    for n, m in methods.items():
        r = make_eval(ft_q, ft_g, tq, tg, m); per_q[n] = r
        row = {"method": n}
        for k in keys(r):
            mu, lo, hi = M.boot_ci(r[k], seed=seed); row[k if k != "AP" else "mAP"] = f"{mu:.4f} [{lo:.4f},{hi:.4f}]"
        rows.append(row)
    # alternative colour descriptor ablation
    other = "lab" if cdesc == "hsv" else "hsv"
    for n, m in {f"fixed_best_with_{other}": {"type": "fixed", "lam": lam_star}}.items():
        r = make_eval(ft_q, ft_g, tq, tg, m, other); per_q[n] = r
        row = {"method": n}
        for k in keys(r):
            mu, lo, hi = M.boot_ci(r[k], seed=seed); row[k if k != "AP" else "mAP"] = f"{mu:.4f} [{lo:.4f},{hi:.4f}]"
        rows.append(row)
    main_df = pd.DataFrame(rows); main_df.to_csv(f"{a.out}/metrics_main.csv", index=False)
    pq = []
    for n, r in per_q.items():
        d_ = {"method": n, "qi": tq, "cat": ds.cats[tq], "colour_label": ds.clabels[tq] if ds.clabels is not None else ""}
        d_.update({k: r[k] for k in keys(r)}); pq.append(pd.DataFrame(d_))
    pd.concat(pq).to_csv(f"{a.out}/per_query.csv", index=False)

    # Metric validity: how well does each score predict agreement of the HUMAN colour label (or identity if none)?
    from sklearn.metrics import roc_auc_score
    rs = np.random.default_rng(seed); npairs = 300000
    qi_, gi_ = rs.integers(0, len(tq), npairs), rs.integers(0, len(tg), npairs)
    lab_q, lab_g = (ds.clabels, ds.clabels) if ds.clabels is not None else (ds.ids, ds.ids)
    y = lab_q[tq][qi_] == lab_g[tg][gi_]; samecat = ds.cats[tq][qi_] == ds.cats[tg][gi_]
    def pair_scores(sl):
        q, g = qi_[sl], gi_[sl]
        return {"hsv_bhattacharyya": (ft_q["hsv"][q] * ft_g["hsv"][g]).sum(1), "lab_bhattacharyya": (ft_q["lab"][q] * ft_g["lab"][g]).sum(1),
                "neg_dE00_foreground_median": -C.delta_e00(ft_q["evalLab"][q], ft_g["evalLab"][g]).ravel(),
                "semantic_cosine": (ft_q["sem"][q] * ft_g["sem"][g]).sum(1)}
    parts = [pair_scores(slice(i, min(i + 20000, npairs))) for i in range(0, npairs, 20000)]   # chunked: low memory
    scores = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}
    ag = []
    for scope, mk in (("all_pairs", np.ones_like(y, bool)), ("same_category_pairs", samecat)):
        for nme, sc_ in scores.items():
            if 0 < y[mk].sum() < mk.sum(): ag.append({"scope": scope, "score": nme, "auc": roc_auc_score(y[mk], sc_[mk]), "positive_rate": float(y[mk].mean()), "n_pairs": int(mk.sum())})
    pd.DataFrame(ag).to_csv(f"{a.out}/metric_agreement.csv", index=False)

    # tau sensitivity (pixel-colour threshold): the 90th-percentile default can be very permissive
    sens = []
    for pct in (25, 50, 75, 90):
        t_ = float(np.percentile(de, pct))
        for n, m in methods.items():
            r = make_eval(ft_q, ft_g, tq, tg, m, tau_=t_)
            sens.append({"tau_pct": pct, "tau_dE00": t_, "method": n, "Col@10": r["Col@10"].mean(), "Joint@10": r["Joint@10"].mean()})
    pd.DataFrame(sens).to_csv(f"{a.out}/tau_sensitivity.csv", index=False)

    # paired bootstrap: adaptive minus best fixed
    an = [k for k in per_q if k.startswith("adaptive(")][0]; fn_ = [k for k in per_q if k.startswith("fixed_best(")][0]
    paired = {k: M.boot_ci(per_q[an][k] - per_q[fn_][k], seed=seed) for k in [k for k in ["AP", "R@1", "Sem@10", "Col@10", "Joint@10", "ColLab@10", "JointLab@10"] if k in per_q[an]]}
    anJ = [k for k in per_q if k.startswith("adaptiveJ(")][0]; fnJ = [k for k in per_q if k.startswith("fixed_bestJ(")][0]
    pairedJ = {k: M.boot_ci(per_q[anJ][k] - per_q[fnJ][k], seed=seed) for k in paired}
    paired_extra = {}
    for nm_, key_ in (("learned_fixed", "learned_fixed"), ("learned_qcond", "learned_qcond"), ("fixed_best_cliptext(", "fixed_cliptext")):
        hit = [k for k in per_q if k.startswith(nm_)]
        if hit: paired_extra[f"{key_}_minus_best_fixed (mean, lo95, hi95)"] = {k: M.boot_ci(per_q[hit[0]][k] - per_q[fn_][k], seed=seed) for k in paired}
    f5 = "fixed_lambda0.5"
    paired5 = {k: M.boot_ci(per_q[an][k] - per_q[f5][k], seed=seed) for k in paired}
    json.dump({"adaptive_minus_best_fixed (mean, lo95, hi95)": paired, "adaptive_minus_fixed_lambda0.5 (mean, lo95, hi95)": paired5, "adaptiveJ_minus_best_fixedJ (mean, lo95, hi95)": pairedJ, "selection_metrics": sels, **paired_extra, "lamJ": lamJ, "adaptiveJ": list(aJ), "tau_dE00": tau, "lam_star": lam_star,
               "adaptive_star": list(a_star), "n_test_queries": int(len(tq)), "n_test_gallery": int(len(tg)),
               "n_val_queries": int(len(vq)), "n_val_gallery": int(len(vg)), "colour_descriptor": cdesc,
               "encoder": enc.name, "dataset": ds.name, "seed": seed}, open(f"{a.out}/summary.json", "w"), indent=2)

    if a.qualitative:
        base_, ref_ = "semantic_only", fn_; col_ = f"colour_only_{cdesc}"
        show = [base_, col_, ref_] + [k for k in per_q if k.startswith("learned_qcond") or k.startswith("fixed_best_cliptext(")]
        fns_ = {n_: score_fn(ft_q, ft_g, methods[n_], methods[n_].get("kind", cdesc)) for n_ in show}
        X.failure_analysis(ds, tq, tg, fns_, per_q, ref_, base_, f"{a.out}/analysis")
        X.qualitative(ds, tq, tg, {n_: fns_[n_] for n_ in show[:3]}, per_q, ref_, base_, f"{a.out}/qualitative", seed=seed)

    # ---- lambda sweep for trade-off figure (test, reported not tuned) -----
    sweep = []
    for l in np.linspace(0, 1, 11):
        r = make_eval(ft_q, ft_g, tq, tg, {"type": "fixed", "lam": float(l)})
        sweep.append({"lam": l, "mAP": r["AP"].mean(), "Sem@10": r["Sem@10"].mean(), "Col@10": r["Col@10"].mean(), "Joint@10": r["Joint@10"].mean(), **{k: r[k].mean() for k in ("ColLab@10", "JointLab@10") if k in r}})
    pd.DataFrame(sweep).to_csv(f"{a.out}/tradeoff_sweep.csv", index=False)

    # ---- robustness (queries transformed, gallery fixed) ---------------
    rob = []
    for kind, levels in ([] if a.no_robustness else R.LEVELS.items()):
        for lv in levels:
            rr = np.random.default_rng(seed)
            tf = lambda im, kind=kind, lv=lv: R.apply(im, kind, lv, rr)
            fq = feats(ds, tq, enc, kinds, cache, "rob", transform=tf)
            for n, m in methods.items():
                if n.startswith("fixed_lambda") or m["type"] not in ("sem", "col", "fixed", "adaptive") or "kind" in m: continue
                r = make_eval(fq, ft_g, tq, tg, m)
                rob.append({"transform": kind, "level": lv, "method": n, "mAP": r["AP"].mean(), "R@1": r["R@1"].mean(), "Sem@10": r["Sem@10"].mean(), "Col@10": r["Col@10"].mean(), "Joint@10": r["Joint@10"].mean(), **{k: r[k].mean() for k in ("ColLab@10", "JointLab@10") if k in r}})
    if rob: pd.DataFrame(rob).to_csv(f"{a.out}/robustness.csv", index=False)

    # ---- latency / storage ----------------------------------------------
    S_s, S_c = sims(ft_q, ft_g, cdesc); zs, zc = norm.sem(S_s), norm.chr(S_c); nq = min(200, len(tq))
    def timeit(f, rep=5):
        t = time.perf_counter()
        for _ in range(rep): f()
        return (time.perf_counter() - t) / rep / nq * 1000
    lat = {"sem_scoring_ms_per_query": timeit(lambda: ft_q["sem"][:nq] @ ft_g["sem"].T),
           "colour_scoring_ms_per_query": timeit(lambda: ft_q[cdesc][:nq] @ ft_g[cdesc].T),
           "fixed_fusion_ms_per_query": timeit(lambda: U.fixed(zs[:nq], zc[:nq], .5)),
           "adaptive_fusion_ms_per_query": timeit(lambda: U.adaptive(zs[:nq], zc[:nq], *a_star)[0]),
           "storage_bytes_per_gallery_image": {"semantic_fp32": int(ft_g["sem"].shape[1] * 4), "hsv_fp32": int(ft_g["hsv"].shape[1] * 4), "lab_fp32": int(ft_g["lab"].shape[1] * 4)},
           "note": "CPU, brute-force numpy; scoring only (excludes encoder inference)."}
    json.dump(lat, open(f"{a.out}/latency.json", "w"), indent=2)
    print(main_df.to_string(index=False)); print(json.dumps(paired, indent=1)); print("tau", tau, "lam*", lam_star, "adaptive*", a_star)

if __name__ == "__main__": main()
