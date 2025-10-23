# HW2: GNN Equivariance & Invariance — Report

**Learning goals (from assignment):**  
- Verify node-level permutation **equivariance** of message-passing GNNs.  
- Verify graph-level permutation **invariance** using permutation-invariant readouts.  
- Show when/why these properties **break** (counterexample). :contentReference[oaicite:2]{index=2}

## A) Node-level permutation equivariance (Cora)
**Experiment:** 2-layer GCN (`cached=False`, `eval()`), 3 random permutations P of nodes.  
**Test:** Check `allclose(f(PX, PAP^T), P f(X, A))` with `atol=1e-6`.  
**Result:** All three permutations passed.

**Sketch:** Each GCN layer applies \(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2} X W\) then a pointwise nonlinearity.  
For a permutation matrix \(P\): \(\hat{A}' = P\hat{A}P^\top\), \(\hat{D}' = P\hat{D}P^\top\), so  
\(\hat{D}'^{-1/2} \hat{A}' \hat{D}'^{-1/2} (PX) = P \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2} X\).  
Thus each layer is equivariant, and stacking preserves equivariance; \(f(P\cdot)=P f(\cdot)\).  
We use `cached=False` so normalization recomputes after permutation (as requested). :contentReference[oaicite:3]{index=3}

## B) Graph-level permutation invariance (MUTAG)
**Experiment:** Compute node embeddings, then pool with **sum/mean/max**.  
**Test:** For 3 permutations per graph, verify pooled vectors unchanged.  
**Result:** All permutations passed for all three pools.

**Sketch:** Let \(h \in \mathbb{R}^{N\times d}\) be node embeddings and \(P\) a permutation.  
- Sum: \(\sum_i (Ph)_i = \sum_i h_i\).  
- Mean: same up to division by \(N\).  
- Max: permutation of indices does not change the elementwise maxima.  
Therefore \(g(f(PX,PAP^\top)) = g(f(X,A))\) for sum/mean/max. :contentReference[oaicite:4]{index=4}

## Counterexample (required)
**Readout:** “take embedding of node 0”.  
**Why it fails:** After permutation, “node 0” refers to a **different node**, so the readout changes; it is **not permutation-invariant**. Our test confirms a mismatch for random permutations. :contentReference[oaicite:5]{index=5}

## Notes & sanity
- All tests are run in `eval()` mode to avoid randomness; tolerance `1e-6` matches the spec.  
- We used simple features; if any MUTAG graph lacks `x`, we substitute node degree (float) just to compute embeddings.  
- Deliverables match the requested structure (tests.py, model.py, brief report). :contentReference[oaicite:6]{index=6}
