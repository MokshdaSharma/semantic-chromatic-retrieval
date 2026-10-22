# Literature comparison table (to be completed by a human with publisher access)
**Nothing below has been verified.** The five seed references come from the research plan; titles, authors, venues,
dates and results must be checked on the publisher/official page before citing. Do not cite from this file.

Required columns: problem | dataset & relevance definition | deep extractor | colour/texture descriptor | normalisation |
ranking/fusion equation | weighting (fixed / learned / query-dependent / user-controlled) | training required | metrics |
reported compute cost | limitations | code & weights available.

| # | Reference (as given in plan) | Verified? | Weighting type | Closest to our adaptive rule? | Notes |
|---|---|---|---|---|---|
| 1 | Adaptive feature fusion for image retrieval, 2021 — https://www.mdpi.com/1099-4300/23/12/1670 | NO | | | |
| 2 | Adaptive two-tier CBIR, 2025 — https://doi.org/10.1049/ipr2.70192 | NO | | | |
| 3 | Entropy-guided multilevel feature fusion, 2026 — https://www.nature.com/articles/s41598-026-38699-x | NO | | | |
| 4 | Colour-aware re-ranking for fashion retrieval, 2026 — https://doi.org/10.1145/3801839.3801868 | NO | | | **Candidate published baseline to reproduce** |
| 5 | Revisited Oxford & Paris benchmark — https://cmp.felk.cvut.cz/revisitop/ | NO | n/a | n/a | benchmark, not a fusion method |
| 6–15 | (find via the 10 search queries in the plan; include classical query-adaptive late fusion / score-distribution fusion work) | | | | |

Decision to record: does an existing method already implement dispersion-based query-adaptive weighting? If yes, state that
the adaptive rule is a *re-implemented reference method* in this study, not a contribution.
