"""Similarity normalisation, fixed-weight fusion, and the proposed query-conditioned rule."""
import numpy as np

class Norm:
    """Global per-modality z-score. Statistics are fitted ONCE on validation query-gallery
    pairs (never on test), so scores stay comparable across queries."""
    def __init__(self, s_sem_val, s_chr_val):
        self.m_s, self.sd_s = float(s_sem_val.mean()), float(s_sem_val.std() + 1e-8)
        self.m_c, self.sd_c = float(s_chr_val.mean()), float(s_chr_val.std() + 1e-8)
    def sem(self, s): return (s - self.m_s) / self.sd_s
    def chr(self, s): return (s - self.m_c) / self.sd_c

def fixed(zs, zc, lam):
    return lam * zs + (1 - lam) * zc

def adaptive(zs, zc, lam0=0.5, gamma=1.0, pool=50):
    """Training-free, query-conditioned dispersion rule (the single proposed mechanism).
    For each query, take the pool of its `pool` most semantically similar gallery items.
    The modality whose (globally z-normalised) scores vary MORE inside that pool is the one
    that can actually discriminate among semantically plausible candidates, so it is
    up-weighted:
        w_s = lam0 * sigma_s^gamma,  w_c = (1-lam0) * sigma_c^gamma
        alpha = w_s/(w_s+w_c),  beta = 1-alpha
    gamma=0 recovers fixed-weight fusion with lambda=lam0 (so fixed is a special case).
    Returns fused scores and alpha(q)."""
    n = zs.shape[1]; pool = min(pool, n)
    idx = np.argpartition(-zs, pool - 1, axis=1)[:, :pool]
    ss = np.take_along_axis(zs, idx, 1).std(1) + 1e-6
    sc = np.take_along_axis(zc, idx, 1).std(1) + 1e-6
    ws, wc = lam0 * ss ** gamma, (1 - lam0) * sc ** gamma
    a = ws / (ws + wc)
    return a[:, None] * zs + (1 - a)[:, None] * zc, a
