"""Prepare ANY HuggingFace image dataset that has (a) an image, (b) a category, (c) a colour signal, into the layout cr.data expects
(<out>/styles.csv with id, articleType, baseColour [, masterCategory] + <out>/images/<id>.jpg). Then:
    python -m cr.run --dataset generic_labeled --name <short_name> --root <out> --encoder clip --seed 0 --out results/v3/<short_name>_clip_s0

Colour signal, either:
  --colour_col COL          a column that already holds a colour label, or
  --colour_text_col COL     a caption/title column; the colour is the single basic-colour word found in it (images with zero or
                            several DIFFERENT colour words are skipped as ambiguous). This is a human-written, pixel-independent signal.
Category: --category_col COL (required). Optional: --master_col COL.
CHECK THE DATASET CARD FIRST: image + category + colour info present, download size you can afford, and the licence.
Nothing here verifies any specific dataset name; the column names are yours to supply.
Example: python scripts/prepare_hf_labeled.py --repo <hf/repo> --split train --image_col image --category_col <cat> --colour_text_col <caption> --n 20000 --out /content/second"""
import argparse, os, re, pandas as pd
from datasets import load_dataset
BASIC = ["black", "white", "grey", "gray", "red", "orange", "yellow", "green", "blue", "navy", "purple", "pink", "brown", "beige", "maroon", "teal", "gold", "silver", "cream", "khaki", "olive", "burgundy", "turquoise"]
CANON = {"gray": "grey", "navy": "navy blue"}
ap = argparse.ArgumentParser()
for k in ("repo", "out", "category_col"): ap.add_argument("--" + k, required=True)
ap.add_argument("--split", default="train"); ap.add_argument("--image_col", default="image"); ap.add_argument("--colour_col"); ap.add_argument("--colour_text_col"); ap.add_argument("--master_col")
ap.add_argument("--n", type=int, default=20000); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--max_side", type=int, default=224)
a = ap.parse_args(); assert a.colour_col or a.colour_text_col, "need --colour_col or --colour_text_col"
os.makedirs(f"{a.out}/images", exist_ok=True)
ds = load_dataset(a.repo, split=a.split, streaming=True).shuffle(seed=a.seed, buffer_size=5000)
rx = re.compile(r"\b(" + "|".join(BASIC) + r")\b"); rows = []; seen = skipped = 0
for k, ex in enumerate(ds):
    if k == 0: print("columns:", list(ex.keys()))
    if len(rows) >= a.n: break
    seen += 1
    if a.colour_col: col = ex.get(a.colour_col)
    else:
        found = {CANON.get(w, w) for w in rx.findall(str(ex.get(a.colour_text_col, "")).lower())}; col = found.pop() if len(found) == 1 else None
    cat = ex.get(a.category_col)
    if col is None or cat is None or str(col).strip() == "": skipped += 1; continue
    im = ex[a.image_col].convert("RGB"); im.thumbnail((a.max_side, a.max_side)); im.save(f"{a.out}/images/{len(rows)}.jpg", quality=92)
    r = {"id": len(rows), "articleType": str(cat), "baseColour": str(col)}
    if a.master_col: r["masterCategory"] = ex.get(a.master_col)
    rows.append(r)
pd.DataFrame(rows).to_csv(f"{a.out}/styles.csv", index=False); print(f"saved {len(rows)} images (seen {seen}, skipped {skipped} without a usable colour/category) -> {a.out}")
