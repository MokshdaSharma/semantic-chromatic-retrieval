# v2 results summary (5 seeds x {CLIP ViT-B/32, DINOv2-B}; Fashion Product Images small; ~1.8k test queries, ~16k gallery per split)
All numbers are generated: `results/v2/*` (raw), `results/v2_tables/*` (aggregates), `paper/tables/*.tex`. Protocol: `PLAN.md`.

| Hypothesis (declared in PLAN.md) | Verdict | Evidence |
|---|---|---|
| H1 fusion beats either modality | **Supported** | P@10: CLIP 0.441→0.498, DINOv2 0.337→0.493 (paired CI excludes 0 in 5/5 splits); category relevance −0.019/−0.027 |
| H2 adaptive ≈ tuned fixed (margin ±0.01) | **Supported** | mean |Δ| ≤ 0.0026; CI inside margin 5/5 splits (mAP selection). One DINOv2 split breaks it under P@10 selection (mAP −0.0375). γ=0 chosen in 6/10 (mAP) and 4/10 (P@10) runs |
| H3 pixel colour is a weak/uneven proxy for human colour | **Supported (moderate)** | ROC-AUC 0.62 HSV, 0.66 Lab, 0.74 ΔE00-median (all pairs); semantic cosine 0.54–0.57. Labels are noisy, so this is agreement, not ground truth about pixels |
| H4 fusion helps chromatic > achromatic | **Supported (caveat: headroom)** | colour-match gain CLIP +0.096 vs +0.064; DINOv2 +0.248 vs +0.151; Yellow +0.28 … Silver/Steel/Gold ≈ 0 |
| H5 fused ranking tracks semantic under nuisance | **Mixed** | CLIP: yes (blur/occlusion fragile). DINOv2: fused retains less than semantic-only under photometric changes (0.70 vs 0.92 brightness) but ≥ semantic in absolute P@10 in 11/12 pairs |

Other findings: colour match plateaus (~0.62) over a wide λ range; objective for choosing λ (mAP vs P@10) moves the operating point more than adaptivity; pixel colour-consistency saturates at τ = 90th pct (ΔE00≈30.7); adaptive scoring 0.12 ms/query vs 0.03 fixed (CPU, ~16k gallery).

Caveats: class-level relevance; coarse noisy labels; single dataset; seeds' test sets overlap (spread understates new-data uncertainty); 7 metrics without multiplicity correction; robustness has no per-condition CIs; several analyses were added after the first exploratory run (see protocol log); novelty/literature NOT checked.
