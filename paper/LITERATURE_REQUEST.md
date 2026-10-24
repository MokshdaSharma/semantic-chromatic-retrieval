# Literature to share with Claude (by heading)
For each heading send: (1) verified BibTeX (from the publisher page), (2) 2–3 lines on what the method does, (3) for fusion/re-ranking papers: is weighting **fixed / learned / query-dependent / user-controlled**, what is the fusion equation, datasets, reported numbers, whether code exists. Aim for the numbers in brackets; quality over quantity. Mark anything you could not verify.

1. **Colour descriptors & perceptual colour similarity** [3–5] — HSV/Lab/colour histograms, colour moments, CIEDE2000 and its use in retrieval.
2. **Deep image retrieval with pretrained / foundation encoders** [4–6] — CLIP, DINOv2, instance vs category retrieval, fashion/product retrieval.
3. **Hybrid fusion of deep and handcrafted features** [4–6] — early/late fusion, score normalisation (the plan's 2021 entropy-fusion and 2025 two-tier CBIR papers belong here or in 4).
4. **Adaptive / query-dependent weighting and score-distribution fusion** [6–10] — **most important for the novelty check**: query-adaptive late fusion, score-dispersion/entropy-based weighting, learning-to-rank fusion. Please state for each whether it is equivalent to our dispersion rule.
5. **Colour-aware retrieval and re-ranking** [4–6] — especially fashion/product (the plan's 2026 colour-aware re-ranking paper), colour-conditioned or colour-controllable search.
6. **Colour understanding in vision–language models** [3–5] — CLIP colour sensitivity/attribute binding, zero-shot colour classification (justifies the colour-text-prompt baseline).
7. **Small learned fusion / learning-to-rank baselines** [2–4] — logistic/pairwise LTR for combining similarity cues.
8. **Evaluation methodology** [4–6] — validity of attribute/colour labels and label noise, agreement between colour metrics and human judgement, paired bootstrap / equivalence testing, multi-seed reporting in retrieval.
9. **Robustness of retrieval to photometric changes** [2–4] — illumination/colour-constancy effects on descriptors and embeddings.
10. **Datasets & licences** [exact citations + licence terms] — Fashion Product Images (small) (original source/Kaggle page, licence), the second dataset you choose, and for context DeepFashion, Stanford Online Products, ABO if relevant.

Also tell me: the **closest competing method** you found (so I can implement it as a baseline), and the **venue** you target (page limit and template).
