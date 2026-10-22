# Plan & strategy — query-conditioned semantic–chromatic retrieval

Status legend: ✅ done & tested · 🟡 code ready, needs GPU/data access · ⛔ not done

## 0. Honest status
| Item | Status |
|---|---|
| Pipeline (features → cached → normalise → fuse → evaluate → robustness → latency → figures) | ✅ unit-tested (`pytest tests`) |
| Synthetic smoke test (`results/smoke_synthetic`) | ✅ runs; **validates plumbing only, not a scientific result** |
| DeepFashion In-Shop + CLIP/DINOv2 run | 🟡 needs Colab/Kaggle (the build sandbox had no GPU and no access to HuggingFace / dataset hosts) |
| External validation (Stanford Online Products) | 🟡 loader TODO after the primary run is stable |
| Literature review, novelty gate, published colour-aware re-rank baseline | ⛔ must be done by a human with publisher access; no claims of novelty are made here |

## 0b. Dataset pivot (DeepFashion too large to download)
Primary dataset for the POC is now **Fashion Product Images (small)** (HF `ashraq/fashion-product-images-small` or Kaggle `paramaggarwal/fashion-product-images-small`), subsampled to ~15k images.
* Semantic category = `articleType`. Human colour label = `baseColour`. **Relevance (replaces instance identity) = same articleType AND same baseColour.**
* Colour consistency is now reported two ways: pixel-based CIEDE2000 (`Col@K`) **and** human-label-based (`ColLab@K`, `JointLab@K`) — the latter removes the pixel-circularity concern.
* Caveats to state in the paper: class-level, not exact-product, relevance; images are low resolution (~60x80); `baseColour` labels are coarse/noisy (e.g. Navy Blue vs Blue); random image-level val/test split (no instance identities exist to split by).
* HF column names are unverified from this sandbox: `scripts/prepare_fashion_small.py` prints the real columns and fails loudly if they differ. DeepFashion loader remains for later.
* The loader was tested on a mock folder with the same layout (`styles.csv` + `images/`), not on the real data.

## 1. Single proposed mechanism (chosen from the plan's alternatives)
**Training-free dispersion-reliability rule.** Scores of each modality are globally z-normalised
(statistics fitted on validation pairs only). For query q, take its top-N semantic neighbours;
the modality that varies more within that pool is the one able to separate semantically plausible
candidates, so it is up-weighted:

`alpha(q) = l0·σs^γ / (l0·σs^γ + (1−l0)·σc^γ)`, `beta = 1 − alpha`, `S = alpha·z_sem + beta·z_chr`.

* γ = 0 ⇒ exactly the fixed-weight baseline (tested), so fixed fusion is a *special case*.
* Search-budget parity: λ (fixed) and λ0 (adaptive) use the same 21-point grid; adaptive additionally tunes γ∈{0,.5,1,2}, N∈{20,50}. Selection metric = validation identity mAP. Test is never used for tuning.
* **Prior-art risk (unverified):** dispersion/score-distribution-based query-adaptive fusion is an old family (e.g., query-adaptive late fusion, score-distribution fusion). The novelty gate in §5 of the research plan may well *fail*; if so, reframe as an empirical study of fixed vs. adaptive semantic–chromatic fusion on colour-sensitive retrieval.

## 2. Relevance definitions (fixed before any test run)
* **Identity relevance**: same `item_id` (DeepFashion In-Shop official query/gallery). Drives mAP, R@K, P@K.
* **Semantic relevance@K**: share of top-K with the query's product category (independent of colour).
* **Chromatic relevance (evaluator-side)**: CIEDE2000 between the *median CIELAB of background-removed foreground pixels* of query and result ≤ τ. τ = 90th percentile of ΔE00 between same-item validation pairs (documented, data-derived from validation only, no test labels).
* **Joint@K**: top-K results that are both category-correct and colour-consistent — a single number that cannot hide a semantic drop.
* *Circularity caveat*: retrieval uses colour histograms, evaluation uses a different colour statistic, but both come from pixels. Colour-text labels from DeepFashion attributes (category-and-attribute benchmark) are the recommended independent check — TODO.
* *Known limit*: DeepFashion identity labels do not guarantee that different `item_id`s differ in colour; report this.

## 3. Protocol
1. DeepFashion In-Shop official train/query/gallery. No official val split ⇒ 15 % of **train identities** are held out (seeded) for val: first image = query, rest = gallery.
2. Frozen CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) image embeddings; optional DINOv2-base CLS. Features cached to `.npz`.
3. Colour: HSV (16×4×4) sqrt-histogram on a central ellipse (Bhattacharyya = dot product). Lab (4×8×8) as the descriptor ablation.
4. Baselines: colour-only, semantic-only, fixed λ=0.5, validation-best fixed λ, adaptive. (Published colour-aware re-ranker: ⛔ after literature check.)
5. Metrics with 95 % bootstrap CIs over queries; paired bootstrap for adaptive − best-fixed.
6. Robustness on queries only: brightness, contrast, saturation, hue-shift, blur, occlusion × 4 levels, seeded. Hue-shift is a *meaningful* colour change, not a nuisance — interpret separately (the right behaviour is for Col@K to follow the new colour, not stay constant).
7. Latency/storage: brute-force numpy scoring, ms/query, bytes/gallery image.

## 4. Go/No-Go (decided on the real run, not the smoke test)
Proceed with the adaptive method as the contribution only if (a) the novelty check finds a defensible difference, (b) paired CI of adaptive − best-fixed on Joint@10/mAP excludes 0 on test and on the external set, (c) gain survives the Lab descriptor and DINOv2. Otherwise report as an empirical study.

## 5. Next steps for you
1. `notebooks/run_colab.ipynb` on Colab GPU: `--encoder clip` then `--encoder dinov2`, DeepFashion In-Shop (needs the dataset password from the authors' page).
2. Share the `results/<run>/` folder (or push) → I analyse, build tables/figures, and draft the manuscript sections.
3. Do the literature table (§5 of your plan); tell me the closest competing method so I can implement it as a baseline.
