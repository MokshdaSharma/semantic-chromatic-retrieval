"""Colour descriptors (retrieval-side) and the evaluator-side colour definition."""
import numpy as np, cv2
from skimage.color import deltaE_ciede2000

def _ellipse_mask(h, w):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (int(w * .42), int(h * .46)), 0, 0, 360, 1, -1)
    return m

def hist_descriptor(img_rgb, kind="hsv", size=96):
    """L1-normalised 3-D histogram over a central elliptical region, returned as
    sqrt(hist) so that a dot product equals the Bhattacharyya coefficient in [0,1]."""
    img = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    m = _ellipse_mask(size, size)
    if kind == "hsv":
        x = cv2.cvtColor(img, cv2.COLOR_RGB2HSV); bins, rng = [16, 4, 4], [0, 180, 0, 256, 0, 256]
    elif kind == "lab":
        x = cv2.cvtColor(img, cv2.COLOR_RGB2LAB); bins, rng = [4, 8, 8], [0, 256, 0, 256, 0, 256]
    else:
        raise ValueError(kind)
    h = cv2.calcHist([x], [0, 1, 2], m, bins, rng).ravel().astype(np.float32)
    h /= max(h.sum(), 1e-8)
    return np.sqrt(h)

def fg_lab(img_rgb, size=96, bg_thresh=10.0):
    """EVALUATOR-side colour: median CIELAB of foreground pixels (border-median background
    removed). Deliberately different from the retrieval histograms to limit circularity."""
    img = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB).reshape(-1, 3)  # true Lab scale
    L = lab.reshape(size, size, 3)
    border = np.concatenate([L[0], L[-1], L[:, 0], L[:, -1]])
    bg = np.median(border, 0)
    fg = np.linalg.norm(lab - bg, axis=1) > bg_thresh
    if fg.mean() < 0.02:
        fg = _ellipse_mask(size, size).ravel().astype(bool)
    return np.median(lab[fg], 0).astype(np.float32)

def delta_e00(lab_a, lab_b):
    """CIEDE2000 between arrays of Lab triplets."""
    return deltaE_ciede2000(np.atleast_2d(lab_a), np.atleast_2d(lab_b))
