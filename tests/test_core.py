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
