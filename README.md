GNN Equivariance & Invariance
===========================================

A) Node-level permutation equivariance (Cora)
---------------------------------------------

**Experiment:**\
2-layer GCN (`cached=False`, `eval()`), 3 random permutations **P** of nodes.

**Test:**

`allclose(f(PX, PAP^T), P f(X, A), atol=1e-6)`

**Result:**\
All three permutations passed

**Sketch:**\
Each GCN layer applies

$$
\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2} X W
$$


followed by a pointwise nonlinearity.\
For a permutation matrix P:

$$
\hat{A}' = P \hat{A} P^\top, \quad \hat{D}' = P \hat{D} P^\top
$$

$$
\Rightarrow \hat{D}'^{-1/2} \hat{A}' \hat{D}'^{-1/2} (P X)
  = P \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} X
$$


Thus each layer is equivariant, and stacking preserves equivariance:

$$
f(P\cdot) = P\,f(\cdot)
$$


We use `cached=False` so normalization recomputes after permutation (as required).

* * * * *

B) Graph-level permutation invariance (MUTAG)
---------------------------------------------

**Experiment:**\
Compute node embeddings, then pool with **sum / mean / max**.

**Test:**\
For 3 permutations per graph, verify pooled vectors are unchanged.

**Result:**\
All permutations passed for all three pools

**Sketch:**\
Let $h \in \mathbb{R}^{N \times d}$ be node embeddings and $P$ a permutation.

* **Sum:** $\sum_i (Ph)_i = \sum_i h_i$
* **Mean:** same up to division by $N$
* **Max:** permutation of indices does not change the elementwise maxima

Therefore,
$$g(f(PX, PAP^T)) = g(f(X,A))$$
for sum, mean, and max pooling.


* * * * *

Counterexample (required)
-------------------------

**Readout:**\
Take the embedding of **node 0** (no pooling).

**Why it fails:**\
After permutation, "node 0" refers to a *different* vertex, so the readout changes --- it is **not permutation-invariant**.\
Our test confirmed a mismatch for random permutations, as expected.

* * * * *

Notes & sanity checks
---------------------

-   All tests run in `eval()` mode to avoid randomness; tolerance `atol=1e-6` matches the specification.

-   Used simple features; if a MUTAG graph lacks `x`, node degree (float) is substituted.

-   Deliverables match the requested structure (`tests.py`, `model.py`, `report.md`).
