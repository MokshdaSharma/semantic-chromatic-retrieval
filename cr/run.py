"""End-to-end experiment. Usage:
  python -m cr.run --dataset synthetic --encoder hog --out results/smoke
  python -m cr.run --dataset deepfashion --root /data/DeepFashion/InShop --encoder clip --out results/deepfashion_clip
"""
import argparse, json, os, sys, time, itertools, numpy as np, pandas as pd, yaml
from tqdm import tqdm
from . import data as D, features as F, colour as C, fusion as U, metrics as M, robustness as R

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic"); ap.add_argument("--root")
    ap.add_argument("--encoder", default="hog"); ap.add_argument("--hf_model")
    ap.add_argument("--out", required=True); ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--max_test_queries", type=int, default=None)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); seed = cfg["seed"]
    os.makedirs(a.out, exist_ok=True); cache = os.path.join(a.out, "cache"); os.makedirs(cache, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---- data & splits -------------------------------------------------
    if a.dataset == "synthetic":
        ds = D.synthetic(seed); tq, tg = ds.idx("test_query"), ds.idx("test_gallery"); vq, vg = ds.idx("val_query"), ds.idx("val_gallery")
    else:
        ds = D.deepfashion_inshop(a.root); tq, tg = ds.idx("query"), ds.idx("gallery")
        tr = ds.idx("train"); u = np.unique(ds.ids[tr]); rng.shuffle(u); vids = set(u[: int(len(u) * cfg["val_identity_frac"])].tolist())
        vm = np.array([i in vids for i in ds.ids[tr]]); vsel = tr[vm]
        # per validation identity: first image -> query, rest -> gallery (identities disjoint from test)
        _, first = np.unique(ds.ids[vsel], return_index=True); isq = np.zeros(len(vsel), bool); isq[first] = True
        vq, vg = vsel[isq], vsel[~isq]
    if a.max_test_queries and len(tq) > a.max_test_queries: tq = rng.choice(tq, a.max_test_queries, replace=False)
    enc = F.HOGStandIn() if a.encoder == "hog" else F.HFEncoder(a.encoder, a.hf_model)
    kinds = ["hsv", "lab"]; tagp = f"{ds.name}_{enc.name}"
    fv_q = feats(ds, vq, enc, kinds, cache, tagp + "_vq"); fv_g = feats(ds, vg, enc, kinds, cache, tagp + "_vg")
    ft_q = feats(ds, tq, enc, kinds, cache, tagp + "_tq"); ft_g = feats(ds, tg, enc, kinds, cache, tagp + "_tg")

    cdesc = cfg["colour_descriptor"]
    def sims(fq, fg, kind): return fq["sem"] @ fg["sem"].T, fq[kind] @ fg[kind].T
    # ---- normalisation fitted on validation pairs only ----------------
    sv_s, sv_c = sims(fv_q, fv_g, cdesc); norm = U.Norm(sv_s, sv_c)

    # ---- evaluator-side colour threshold tau (from val, same-identity pairs) ----
    vqid, vgid = ds.ids[vq], ds.ids[vg]
    same = vqid[:, None] == vgid[None]
    de = np.array([C.delta_e00(fv_q["evalLab"][i][None], fv_g["evalLab"][np.where(same[i])[0]]).ravel() for i in range(len(vq)) if same[i].any()], dtype=object)
    de = np.concatenate(list(de)); tau = float(np.percentile(de, cfg["tau_percentile"]))

    def make_eval(fq, fg, qi, gi, method, kind=cdesc):
        S_s, S_c = sims(fq, fg, kind); zs, zc = norm.sem(S_s), norm.chr(S_c)
        fn = {"sem": lambda sl: zs[sl], "col": lambda sl: zc[sl],
              "fixed": lambda sl: U.fixed(zs[sl], zc[sl], method["lam"]),
              "adaptive": lambda sl: U.adaptive(zs[sl], zc[sl], method["lam0"], method["gamma"], method["pool"])[0]}[method["type"]]
        return M.evaluate(fn, len(qi), ds.ids[qi], ds.ids[gi], ds.cats[qi], ds.cats[gi], fq["evalLab"], fg["evalLab"], tau)

    # ---- hyper-parameter selection on VALIDATION only -----------------
    sel = cfg["selection_metric"]; score = lambda r: float(r[sel].mean())
    def v(method, kind=cdesc): return score(make_eval(fv_q, fv_g, vq, vg, method, kind))
    grid_f = {l: v({"type": "fixed", "lam": l}) for l in cfg["lambda_grid"]}
    lam_star = max(grid_f, key=grid_f.get)
    grid_a = {k: v({"type": "adaptive", "lam0": k[0], "gamma": k[1], "pool": k[2]})
              for k in itertools.product(cfg["lam0_grid"], cfg["gamma_grid"], cfg["pool_grid"])}
    a_star = max(grid_a, key=grid_a.get)
    pd.DataFrame([{"lam": l, sel: s} for l, s in grid_f.items()]).to_csv(f"{a.out}/val_fixed_grid.csv", index=False)
    pd.DataFrame([{"lam0": k[0], "gamma": k[1], "pool": k[2], sel: s} for k, s in grid_a.items()]).to_csv(f"{a.out}/val_adaptive_grid.csv", index=False)

    # ---- TEST ----------------------------------------------------------
    methods = {
        f"colour_only_{cdesc}": {"type": "col"}, "semantic_only": {"type": "sem"},
        "fixed_lambda0.5": {"type": "fixed", "lam": 0.5},
        f"fixed_best(lam={lam_star})": {"type": "fixed", "lam": lam_star},
        f"adaptive(lam0={a_star[0]},g={a_star[1]},N={a_star[2]})": {"type": "adaptive", "lam0": a_star[0], "gamma": a_star[1], "pool": a_star[2]},
    }
    per_q, rows = {}, []
    for n, m in methods.items():
        r = make_eval(ft_q, ft_g, tq, tg, m); per_q[n] = r
        row = {"method": n}
        for k in ["AP", "R@1", "R@10", "P@10", "Sem@10", "Col@10", "Joint@10", "Col@1", "Joint@1"]:
            mu, lo, hi = M.boot_ci(r[k], seed=seed); row[k if k != "AP" else "mAP"] = f"{mu:.4f} [{lo:.4f},{hi:.4f}]"
        rows.append(row)
    # alternative colour descriptor ablation
    other = "lab" if cdesc == "hsv" else "hsv"
    for n, m in {f"fixed_best_with_{other}": {"type": "fixed", "lam": lam_star}}.items():
        r = make_eval(ft_q, ft_g, tq, tg, m, other); per_q[n] = r
        row = {"method": n}
        for k in ["AP", "R@1", "R@10", "P@10", "Sem@10", "Col@10", "Joint@10", "Col@1", "Joint@1"]:
            mu, lo, hi = M.boot_ci(r[k], seed=seed); row[k if k != "AP" else "mAP"] = f"{mu:.4f} [{lo:.4f},{hi:.4f}]"
        rows.append(row)
    main_df = pd.DataFrame(rows); main_df.to_csv(f"{a.out}/metrics_main.csv", index=False)

    # paired bootstrap: adaptive minus best fixed
    an = [k for k in per_q if k.startswith("adaptive")][0]; fn_ = [k for k in per_q if k.startswith("fixed_best(")][0]
    paired = {k: M.boot_ci(per_q[an][k] - per_q[fn_][k], seed=seed) for k in ["AP", "R@1", "Sem@10", "Col@10", "Joint@10"]}
    json.dump({"adaptive_minus_best_fixed (mean, lo95, hi95)": paired, "tau_dE00": tau, "lam_star": lam_star,
               "adaptive_star": list(a_star), "n_test_queries": int(len(tq)), "n_test_gallery": int(len(tg)),
               "n_val_queries": int(len(vq)), "n_val_gallery": int(len(vg)), "colour_descriptor": cdesc,
               "encoder": enc.name, "dataset": ds.name, "seed": seed}, open(f"{a.out}/summary.json", "w"), indent=2)

    # ---- lambda sweep for trade-off figure (test, reported not tuned) -----
    sweep = []
    for l in np.linspace(0, 1, 11):
        r = make_eval(ft_q, ft_g, tq, tg, {"type": "fixed", "lam": float(l)})
        sweep.append({"lam": l, "mAP": r["AP"].mean(), "Sem@10": r["Sem@10"].mean(), "Col@10": r["Col@10"].mean(), "Joint@10": r["Joint@10"].mean()})
    pd.DataFrame(sweep).to_csv(f"{a.out}/tradeoff_sweep.csv", index=False)

    # ---- robustness (queries transformed, gallery fixed) ---------------
    rob = []
    for kind, levels in R.LEVELS.items():
        for lv in levels:
            rr = np.random.default_rng(seed)
            tf = lambda im, kind=kind, lv=lv: R.apply(im, kind, lv, rr)
            fq = feats(ds, tq, enc, kinds, cache, "rob", transform=tf)
            for n, m in methods.items():
                if n.startswith("fixed_lambda"): continue
                r = make_eval(fq, ft_g, tq, tg, m)
                rob.append({"transform": kind, "level": lv, "method": n, "mAP": r["AP"].mean(), "R@1": r["R@1"].mean(), "Sem@10": r["Sem@10"].mean(), "Col@10": r["Col@10"].mean(), "Joint@10": r["Joint@10"].mean()})
    pd.DataFrame(rob).to_csv(f"{a.out}/robustness.csv", index=False)

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
