HW2: GNN Equivariance & Invariance --- Report
===========================================

**Learning goals (from assignment):**

-   Verify node-level permutation **equivariance** of message-passing GNNs.

-   Verify graph-level permutation **invariance** using permutation-invariant readouts.

-   Show when/why these properties **break** (counterexample).

* * * * *

A) Node-level permutation equivariance (Cora)
---------------------------------------------

**Experiment:**\
2-layer GCN (`cached=False`, `eval()`), 3 random permutations **P** of nodes.

**Test:**

`allclose(f(PX, PAP^T), P f(X, A), atol=1e-6)`

**Result:**\
All three permutations passed ✅

**Sketch:**\
Each GCN layer applies

D^-1/2A^D^-1/2XW\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2} X WD^-1/2A^D^-1/2XW

followed by a pointwise nonlinearity.\
For a permutation matrix PPP:

A^′=PA^P⊤,D^′=PD^P⊤\hat{A}' = P\hat{A}P^\top, \quad \hat{D}' = P\hat{D}P^\topA^′=PA^P⊤,D^′=PD^P⊤ ⇒D^′-1/2A^′D^′-1/2(PX)=PD^-1/2A^D^-1/2X\Rightarrow \hat{D}'^{-1/2} \hat{A}' \hat{D}'^{-1/2} (PX) = P \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2} X⇒D^′-1/2A^′D^′-1/2(PX)=PD^-1/2A^D^-1/2X

Thus each layer is equivariant, and stacking preserves equivariance:

f(P⋅)=Pf(⋅)f(P\cdot) = P f(\cdot)f(P⋅)=Pf(⋅)

We use `cached=False` so normalization recomputes after permutation (as required).

* * * * *

B) Graph-level permutation invariance (MUTAG)
---------------------------------------------

**Experiment:**\
Compute node embeddings, then pool with **sum / mean / max**.

**Test:**\
For 3 permutations per graph, verify pooled vectors are unchanged.

**Result:**\
All permutations passed for all three pools ✅

**Sketch:**\
Let h∈RN×dh \in \mathbb{R}^{N \times d}h∈RN×d be node embeddings and PPP a permutation.

-   **Sum:** ∑i(Ph)i=∑ihi\sum_i (Ph)_i = \sum_i h_i∑i​(Ph)i​=∑i​hi​

-   **Mean:** same up to division by NNN

-   **Max:** permutation of indices does not change the elementwise maxima

Therefore,

g(f(PX,PAP⊤))=g(f(X,A))g(f(PX, PAP^\top)) = g(f(X, A))g(f(PX,PAP⊤))=g(f(X,A))

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
