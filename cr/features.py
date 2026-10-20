"""Semantic encoders. All return L2-normalised float32 (N,D). Frozen; no training."""
import numpy as np, cv2

class HOGStandIn:
    """CPU stand-in used ONLY for the synthetic smoke test (greyscale HOG, colour-blind)."""
    name = "hog_standin"
    def __init__(self): self.h = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    def encode(self, imgs):
        f = np.stack([self.h.compute(cv2.resize(cv2.cvtColor(i, cv2.COLOR_RGB2GRAY), (64, 64))).ravel() for i in imgs]).astype(np.float32)
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)

class HFEncoder:
    """CLIP (openai/clip-vit-base-patch32, ...) or DINOv2 (facebook/dinov2-base) via transformers."""
    def __init__(self, kind="clip", model=None, device=None, bs=128):
        import torch
        from transformers import AutoImageProcessor, CLIPModel, AutoModel
        self.torch, self.bs = torch, bs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kind = kind; model = model or ("openai/clip-vit-base-patch32" if kind == "clip" else "facebook/dinov2-base")
        self.name = model.replace("/", "_")
        self.proc = AutoImageProcessor.from_pretrained(model)
        self.m = (CLIPModel if kind == "clip" else AutoModel).from_pretrained(model).to(self.device).eval()
        if self.device == "cuda": self.m.half()
    def encode(self, imgs):
        torch, out = self.torch, []
        with torch.no_grad():
            for s in range(0, len(imgs), self.bs):
                x = self.proc(images=list(imgs[s:s + self.bs]), return_tensors="pt")["pixel_values"].to(self.device)
                if self.device == "cuda": x = x.half()
                if self.kind == "clip": f = self.m.get_image_features(pixel_values=x)
                else: f = self.m(pixel_values=x).last_hidden_state[:, 0]  # CLS token
                out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)
