"""Datasets. Each exposes: ids, cats, role ('train'|'query'|'gallery'), load(i)->RGB uint8."""
import os, numpy as np, cv2

class Dataset:
    def __init__(self, ids, cats, role, loader, name):
        self.ids, self.cats, self.role, self._load, self.name = np.asarray(ids), np.asarray(cats), np.asarray(role), loader, name
    def load(self, i): return self._load(i)
    def idx(self, role): return np.where(self.role == role)[0]

# ---------- DeepFashion In-Shop (real experiment) ----------
def deepfashion_inshop(root):
    """root contains Eval/list_eval_partition.txt and img/ (as in the official release).
    Official roles: train / query / gallery. Identity = item_id; category = WOMEN/Dresses etc.
    NOTE: the official release has NO validation split; run.py carves a validation set out
    of the *train* identities (disjoint from test) for hyper-parameter selection."""
    rows = [l.split() for l in open(os.path.join(root, "Eval", "list_eval_partition.txt")).read().splitlines()[2:]]
    paths = [r[0] for r in rows]; ids = [int(r[1].split("_")[1]) for r in rows]; role = [r[2] for r in rows]
    cats = ["/".join(p.split("/")[1:3]) for p in paths]
    def loader(i):
        im = cv2.imread(os.path.join(root, paths[i])); return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return Dataset(ids, cats, role, loader, "deepfashion_inshop")

# ---------- Synthetic (pipeline smoke test ONLY; not evidence for any claim) ----------
def _shape(canvas, kind, c, s, col):
    x, y = c; 
    if kind == 0: cv2.circle(canvas, (x, y), s, col, -1)
    elif kind == 1: cv2.rectangle(canvas, (x - s, y - s), (x + s, y + s), col, -1)
    elif kind == 2: cv2.fillPoly(canvas, [np.array([[x, y - s], [x - s, y + s], [x + s, y + s]])], col)
    elif kind == 3: cv2.rectangle(canvas, (x - s, y - s // 3), (x + s, y + s // 3), col, -1); cv2.rectangle(canvas, (x - s // 3, y - s), (x + s // 3, y + s), col, -1)
    elif kind == 4: cv2.circle(canvas, (x, y), s, col, max(2, s // 3))
    elif kind == 5: cv2.fillPoly(canvas, [np.array([[x, y - s], [x + s, y], [x, y + s], [x - s, y]])], col)

def synthetic(seed=0, n_shapes=6, n_colours=14, views_q=2, views_g=5, size=96):
    """item = (shape, colour). Semantic category = shape; identity = shape+colour.
    Colours include low-saturation ones so colour reliability varies across queries."""
    rng = np.random.default_rng(seed)
    cols = []
    for h in np.linspace(0, 179, n_colours - 3, endpoint=False):
        cols.append(cv2.cvtColor(np.uint8([[[h, 200, 200]]]), cv2.COLOR_HSV2RGB)[0, 0])
    cols += [np.uint8([235, 235, 235]), np.uint8([128, 128, 128]), np.uint8([30, 30, 30])]
    items = [(s, c) for s in range(n_shapes) for c in range(n_colours)]
    split = rng.permutation(len(items)); val_set = set(split[: len(items) // 4].tolist())
    recs = []
    for it_i, (s, c) in enumerate(items):
        part = "val" if it_i in val_set else "test"
        for v in range(views_q + views_g):
            recs.append((it_i, s, c, part, "query" if v < views_q else "gallery", int(rng.integers(1 << 30))))
    ids = [r[0] for r in recs]; cats = [r[1] for r in recs]
    role = [f"{r[3]}_{r[4]}" for r in recs]
    def loader(i):
        _, s, c, _, _, sd = recs[i]; r = np.random.default_rng(sd)
        bg = int(r.integers(215, 255)); img = np.full((size, size, 3), bg, np.uint8)
        col = np.clip(cols[c].astype(float) * r.uniform(.8, 1.15) + r.normal(0, 8, 3), 0, 255).astype(np.uint8)
        _shape(img, s, (size // 2 + int(r.integers(-8, 9)), size // 2 + int(r.integers(-8, 9))), int(r.integers(22, 33)), tuple(int(t) for t in col))
        return np.clip(img.astype(float) + r.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return Dataset(ids, cats, role, loader, "synthetic_smoketest")
