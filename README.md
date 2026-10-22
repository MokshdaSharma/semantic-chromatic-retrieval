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
`results/smoke_synthetic` uses a colour-blind HOG stand-in encoder on synthetic shapes. **It is not evidence for any claim.**
