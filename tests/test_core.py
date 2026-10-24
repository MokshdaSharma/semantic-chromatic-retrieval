import numpy as np, sys; sys.path.insert(0, ".")
from cr import fusion as U, metrics as M, colour as C, robustness as R

def test_gamma0_equals_fixed():
    rng = np.random.default_rng(0); zs, zc = rng.normal(size=(5, 100)), rng.normal(size=(5, 100))
    s, a = U.adaptive(zs, zc, lam0=0.3, gamma=0.0, pool=20)
    assert np.allclose(s, U.fixed(zs, zc, 0.3)) and np.allclose(a, 0.3)

def test_alpha_simplex():
    rng = np.random.default_rng(1); zs, zc = rng.normal(size=(8, 50)), 3 * rng.normal(size=(8, 50))
    _, a = U.adaptive(zs, zc, 0.5, 1.0, 10); assert ((a >= 0) & (a <= 1)).all()

def test_ap():
    assert M.average_precision(np.array([1, 0, 1, 0])) == (1 + 2 / 3) / 2
    assert M.average_precision(np.array([0, 0])) == 0.0

def test_bhattacharyya_self_is_one():
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    for k in ("hsv", "lab"):
        h = C.hist_descriptor(img, k); assert abs(h @ h - 1) < 1e-4

def test_dE_identity_zero_and_lab_scale():
    red = np.zeros((32, 32, 3), np.uint8); red[..., 0] = 255
    lab = C.fg_lab(red); assert abs(lab[0] - 53.2) < 2  # true CIELAB L of sRGB red ~53.2
    assert C.delta_e00(lab[None], lab[None])[0] < 1e-6

def test_transforms_deterministic_and_shape():
    img = np.random.default_rng(0).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    for k, lv in R.LEVELS.items():
        a = R.apply(img, k, lv[1], np.random.default_rng(3)); b = R.apply(img, k, lv[1], np.random.default_rng(3))
        assert a.shape == img.shape and (a == b).all()

def test_pool_stats_matches_adaptive_alpha():
    from cr import fusion as U
    rng = np.random.default_rng(2); zs, zc = rng.normal(size=(6, 80)), 2 * rng.normal(size=(6, 80))
    ss, sc = U.pool_stats(zs, zc, 20); _, a = U.adaptive(zs, zc, 0.5, 1.0, 20)
    assert np.allclose(a, ss / (ss + sc))   # lam0=.5, gamma=1 -> alpha = ss/(ss+sc)

def test_clip_colour_probs_is_unit_norm_sqrt_dist():
    from cr import extras as X
    rng = np.random.default_rng(0); img = rng.normal(size=(5, 16)); img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt = rng.normal(size=(7, 16)); txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    p = X.clip_colour_probs(img, txt); assert p.shape == (5, 7) and np.allclose((p ** 2).sum(1), 1, atol=1e-5)

def test_learned_fusion_learns_that_colour_matters():
    from cr import extras as X
    rng = np.random.default_rng(0); nq, G = 40, 300
    zc = rng.normal(size=(nq, G)); zs = 0.3 * rng.normal(size=(nq, G)); rel = (zc + 0.3 * zs + rng.normal(scale=.5, size=(nq, G))) > 1.8
    ctx = {"ss": np.abs(rng.normal(size=nq)) + .5, "sc": np.abs(rng.normal(size=nq)) + .5, "ch": np.abs(rng.normal(size=nq))}
    for qc in (False, True):
        m = X.LearnedFusion(qc, seed=0).fit(zs, zc, ctx, rel); d = m.describe()
        assert d["z_chr"] > d["z_sem"] > 0 and m.score(zs, zc, ctx).shape == zs.shape

def test_topk_all_matches_argsort():
    from cr import extras as X
    S = np.random.default_rng(0).normal(size=(10, 50)); t = X.topk_all(lambda sl: S[sl], 10, 5, chunk=4)
    assert (t == np.argsort(-S, 1)[:, :5]).all()
