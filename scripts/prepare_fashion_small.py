"""Download a light-weight subsample of Fashion Product Images (small) via HuggingFace `datasets`
(run on Colab/Kaggle/your machine; ~100-200 MB for 15k images) and write the layout cr.data expects:
    <out>/styles.csv   (id, articleType, baseColour)
    <out>/images/<id>.jpg
Usage: python scripts/prepare_fashion_small.py --out /content/fashion_small --n 15000
Alternative: Kaggle 'paramaggarwal/fashion-product-images-small' already has styles.csv + images/ -> pass its folder as --root.
NOTE: HF column names are those I expect for 'ashraq/fashion-product-images-small'; the script prints the real
columns and fails loudly if they differ."""
import argparse, os, pandas as pd
from datasets import load_dataset
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=15000)
ap.add_argument("--repo", default="ashraq/fashion-product-images-small"); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args(); os.makedirs(f"{a.out}/images", exist_ok=True)
ds = load_dataset(a.repo, split="train", streaming=True).shuffle(seed=a.seed, buffer_size=5000)
rows = []
for k, ex in enumerate(ds):
    if k == 0: print("columns:", list(ex.keys()))
    if k >= a.n: break
    at, bc = ex.get("articleType"), ex.get("baseColour")
    if at is None or bc is None: continue
    i = ex.get("id", k); ex["image"].convert("RGB").save(f"{a.out}/images/{i}.jpg", quality=92)
    rows.append({"id": i, "articleType": at, "baseColour": bc, "masterCategory": ex.get("masterCategory"), "productDisplayName": ex.get("productDisplayName")})
pd.DataFrame(rows).to_csv(f"{a.out}/styles.csv", index=False); print("saved", len(rows), "images to", a.out)
