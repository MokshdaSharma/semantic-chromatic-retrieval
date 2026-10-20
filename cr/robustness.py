"""Query-side transformations with predefined severity levels (gallery stays untouched)."""
import numpy as np, cv2
LEVELS = {
 "brightness": [0.6, 0.8, 1.2, 1.5],
 "contrast":   [0.5, 0.75, 1.25, 1.5],
 "saturation": [0.4, 0.7, 1.3, 1.6],
 "hue_shift":  [10, 20, 40, 60],     # degrees (a *meaningful* colour change, see PLAN.md)
 "blur":       [1.0, 2.0, 3.0, 5.0], # gaussian sigma
 "occlusion":  [0.1, 0.2, 0.3, 0.4], # fraction of area, random grey rectangle
}
def apply(img, kind, level, rng):
    x = img.astype(np.float32)
    u8 = lambda a: np.clip(a, 0, 255).astype(np.uint8)
    if kind == "brightness": x = x * level
    elif kind == "contrast": m = x.mean(); x = (x - m) * level + m
    elif kind == "saturation":
        h = cv2.cvtColor(u8(x), cv2.COLOR_RGB2HSV).astype(np.float32); h[..., 1] *= level
        x = cv2.cvtColor(u8(h), cv2.COLOR_HSV2RGB).astype(np.float32)
    elif kind == "hue_shift":
        h = cv2.cvtColor(u8(x), cv2.COLOR_RGB2HSV)
        h[..., 0] = ((h[..., 0].astype(int) + int(level / 2)) % 180).astype(np.uint8)
        x = cv2.cvtColor(h, cv2.COLOR_HSV2RGB).astype(np.float32)
    elif kind == "blur": x = cv2.GaussianBlur(x, (0, 0), level)
    elif kind == "occlusion":
        H, W = x.shape[:2]; a = level * H * W; r = rng.uniform(.6, 1.6)
        h = int(min(H, np.sqrt(a * r))); w = int(min(W, a / max(h, 1)))
        y0, x0 = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1); x[y0:y0 + h, x0:x0 + w] = 127
    else: raise ValueError(kind)
    return u8(x)
