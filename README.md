# colour-retrieval
Frozen-encoder semantic + perceptual-colour retrieval with fixed and query-conditioned fusion. See `PLAN.md`.

```bash
pip install -r requirements.txt
pytest tests                                              # unit tests
python -m cr.run --dataset synthetic --encoder hog --out results/smoke_synthetic   # plumbing check only
python -m cr.run --dataset deepfashion --root /path/to/InShop --encoder clip --out results/df_clip
python -m cr.run --dataset deepfashion --root /path/to/InShop --encoder dinov2 --out results/df_dinov2
python -m cr.report results/df_clip                       # figures from CSVs

# Light-weight POC (recommended): Fashion Product Images (small)
python scripts/prepare_fashion_small.py --out /content/fashion_small --n 15000
python -m cr.run --dataset fashion_small --root /content/fashion_small --encoder clip --out results/fs_clip
# or open notebooks/run_colab_fashion_small.ipynb
```
Outputs per run: `metrics_main.csv`, `summary.json` (τ, selected hyper-parameters, paired CIs), `val_*_grid.csv`,
`tradeoff_sweep.csv`, `robustness.csv`, `latency.json`, `figures/`, `cache/` (features, git-ignored).
`results/v1_fashion_small` holds the first REAL runs (single seed, protocol v1); v2 notebooks supersede them.
`results/smoke_synthetic` uses a colour-blind HOG stand-in encoder on synthetic shapes. **It is not evidence for any claim.**

## Empirical study v2
```bash
# Colab: notebooks/run_colab_v2_multiseed.ipynb   (or manually)
python -m cr.run --dataset fashion_small --root /content/fashion_small --encoder clip --seed 0 --out results/v2/fs_clip_s0
python -m cr.aggregate --runs results/v2/fs_clip_s* results/v2/fs_dinov2_s* --out results/v2_tables
cp results/v2_tables/*.tex paper/tables/
```
See `PLAN.md` (hypotheses, protocol log) and `paper/manuscript.tex` (draft; numbers only via generated tables).
