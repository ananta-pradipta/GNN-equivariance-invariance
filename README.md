# GNN Equivariance and Invariance

Empirical verification that Graph Convolutional Networks (GCNs) are **node-level permutation equivariant** and, when paired with a symmetric readout (sum, mean, or max pooling), produce **graph-level permutation invariant** representations. The experiments use a 2-layer GCN implemented with PyTorch Geometric, tested on the Cora and MUTAG benchmarks. A deliberate counterexample demonstrates how a naive, non-symmetric readout breaks invariance.


## Background

A function $f$ over graphs is **permutation equivariant** if reordering the nodes of the input reorders the output in exactly the same way:

$$f(PX,\; PAP^\top) = P\,f(X, A)$$

A graph-level readout $g$ is **permutation invariant** if the final representation does not depend on node ordering at all:

$$g\bigl(f(PX,\; PAP^\top)\bigr) = g\bigl(f(X, A)\bigr)$$

These properties are fundamental to GNNs: they guarantee that predictions are independent of how nodes happen to be indexed, which is essential since graphs have no canonical node ordering.


## Experiments

### Part A: Node-Level Permutation Equivariance (Cora)

A 2-layer GCN (hidden dim 64, ReLU, `cached=False`) is initialized with random weights and set to `eval()` mode. For each of 3 random permutation matrices $P$, the test checks:

```
torch.allclose(f(PX, PAP^T), P @ f(X, A), atol=1e-6)
```

**Why it holds.** Each GCN layer computes $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}XW$ followed by a pointwise nonlinearity. Under permutation, $\hat{A}' = P\hat{A}P^\top$ and $\hat{D}' = P\hat{D}P^\top$, so the normalized adjacency transforms as $P(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2})P^\top$. This makes each layer equivariant, and stacking equivariant layers preserves the property.

The `cached=False` flag is critical: it forces recomputation of the degree-based normalization after permutation. With caching enabled, stale normalization from the original ordering would break the test.

### Part B: Graph-Level Permutation Invariance (MUTAG)

Node embeddings from the same 2-layer GCN are aggregated via three symmetric pooling functions (sum, mean, max). For 10 graphs from MUTAG, each tested with 3 random permutations:

```
torch.allclose(pool(f(PX, PAP^T)), pool(f(X, A)), atol=1e-6)
```

**Why it holds.** Sum, mean, and max are order-agnostic aggregations. Since the node embeddings are equivariant (Part A), permuting the input simply reorders the rows of the embedding matrix, which does not change the result of any symmetric aggregation.

### Counterexample: Non-Invariant Readout

Taking the embedding of node 0 as the graph representation is **not** permutation invariant. After permutation, "node 0" refers to a different vertex in the original graph, so the readout changes. The test confirms that the node-0 readout produces different vectors under random permutations.


## Repository Structure

```
.
├── README.md
├── model.py      # 2-layer GCN with configurable caching and optional MLP head
└── tests.py      # Equivariance, invariance, and counterexample tests
```


## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric (`torch_geometric`)

Install PyTorch Geometric following the [official guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), then:

```bash
pip install torch-geometric
```

The Cora and MUTAG datasets are downloaded automatically on first run.


## Usage

```bash
python tests.py
```

Expected output:

```
Node-level permutation equivariance passed (Cora, 3 perms, atol=1e-6).
Graph-level permutation invariance passed (sum/mean/max, MUTAG, 3 perms each).
Counterexample (node-0 readout) correctly failed invariance.
```


## Key Implementation Details

- **`cached=False`**: Ensures GCNConv recomputes the symmetric normalization $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ on every forward pass. Without this, the cached normalization from the original node ordering would be reused on the permuted graph, breaking equivariance.
- **`eval()` mode**: Disables dropout so that the forward pass is deterministic, making the allclose comparison meaningful.
- **`atol=1e-6`**: Tolerance for floating-point comparison, accounting for minor numerical differences in the normalized adjacency computation.
- **Feature fallback**: If a MUTAG graph lacks node features (`data.x`), node degree is used as a simple substitute.
