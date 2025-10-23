# tests.py
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import degree
from model import GCN2Layers
import random

# ---------------------------
# Utilities
# ---------------------------
def set_seeds(seed=42):
    random.seed(seed); torch.manual_seed(seed)

def permute_single_graph(data, p):
    """
    Permute a PyG Data object:
    - x' = x[p]
    - edge_index' = inv[edge_index], where inv[p[i]] = i
    - (optional) reorder masks if present
    """
    N = data.num_nodes
    device = data.edge_index.device
    inv = torch.empty_like(p)
    inv[p] = torch.arange(N, device=device)

    x_perm = data.x[p]
    ei = data.edge_index
    edge_index_perm = inv[ei]

    data_perm = data.clone()
    data_perm.x = x_perm
    data_perm.edge_index = edge_index_perm

    for name in ["train_mask", "val_mask", "test_mask"]:
        if hasattr(data, name) and getattr(data, name) is not None:
            setattr(data_perm, name, getattr(data, name)[p])

    return data_perm

def ensure_x_exists(data):
    """
    Some TU graphs may lack x; build a simple feature if needed.
    Prefer existing data.x; otherwise use node degree (float).
    """
    if getattr(data, "x", None) is not None:
        x = data.x
        if not torch.is_floating_point(x):
            x = x.float()
        return x

    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).view(-1, 1)
    return deg.float()

# ---------------------------
# Part I — Node-level equivariance (Cora)
# ---------------------------
@torch.no_grad()
def test_equivariance_node():
    """
    Goal: Verify f(PX, P A P^T) = P f(X, A) for a 2-layer GCN on Cora.
    - eval() to disable dropout
    - cached=False to recompute normalization post-permutation
    - atol=1e-6 (as required)
    """
    set_seeds(0)

    dataset = Planetoid(root="./data/Planetoid", name="Cora",
                        transform=NormalizeFeatures())
    data = dataset[0]
    data.x = data.x.float()
    device = torch.device("cpu")
    data = data.to(device)

    model = GCN2Layers(in_dim=dataset.num_node_features,
                       hid_dim=64,
                       out_dim=dataset.num_classes,
                       dropout=0.5,
                       cached=False).to(device)
    model.eval()  # deterministic

    out = model(data.x, data.edge_index)  # [N, C]

    N = data.num_nodes
    for _ in range(3):
        p = torch.randperm(N, device=device)
        data_p = permute_single_graph(data, p)
        out_p = model(data_p.x, data_p.edge_index)  # f(Px, P A P^T)

        # Equivariance: outputs should permute the same way
        assert torch.allclose(out_p, out[p], atol=1e-6), "Node-level equivariance failed."

    print("Node-level permutation equivariance passed (Cora, 3 perms, atol=1e-6).")

# ---------------------------
# Part II — Graph-level invariance (MUTAG)
# ---------------------------
@torch.no_grad()
def test_invariance_graph():
    """
    Goal: With a permutation-invariant readout g (sum/mean/max),
    verify g(f(PX, P A P^T)) = g(f(X, A)) on MUTAG.
    Also show a counterexample readout that fails (take node 0).
    """
    set_seeds(0)

    dataset = TUDataset(root="./data/TUDataset", name="MUTAG")
    device = torch.device("cpu")

    # Infer input dim robustly (use degree if x missing on any graph)
    sample = dataset[0]
    in_dim = sample.x.size(-1) if getattr(sample, "x", None) is not None else 1

    model = GCN2Layers(in_dim=in_dim, hid_dim=64, out_dim=None,
                       dropout=0.0, cached=False).to(device)
    model.eval()

    pools = {
        "sum": global_add_pool,
        "mean": global_mean_pool,
        "max": global_max_pool,
    }

    # We’ll check 3 permutations per (first) 10 graphs to keep it quick
    num_graphs = min(10, len(dataset))
    for idx in range(num_graphs):
        g = dataset[idx].to(device)
        g.x = ensure_x_exists(g)

        # Original embeddings & pooled vectors
        emb = model(g.x, g.edge_index, return_emb=True)   # [Ni, H]
        batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)
        pooled_ref = {name: op(emb, batch) for name, op in pools.items()}

        # 3 random permutations for this graph
        for _ in range(3):
            p = torch.randperm(g.num_nodes, device=device)
            g_p = g.clone()
            g_p.x = g.x[p]
            inv = torch.empty_like(p); inv[p] = torch.arange(g.num_nodes, device=device)
            g_p.edge_index = inv[g.edge_index]
            batch_p = batch[p]  # coherent with node reorder

            emb_p = model(g_p.x, g_p.edge_index, return_emb=True)
            pooled_perm = {
                name: op(emb_p, batch_p) for name, op in pools.items()
            }

            # Invariance check: all identical up to 1e-6
            for name in pools.keys():
                assert torch.allclose(pooled_perm[name], pooled_ref[name], atol=1e-6), \
                    f"Graph-level invariance failed for {name} pool on graph {idx}."

        # --- Counterexample readout: pick node 0 embedding (NOT permutation invariant)
        # original node-0 embedding vs. permuted node-0 embedding should generally differ
        # (node-0 refers to a different node after permutation)
        p = torch.randperm(g.num_nodes, device=device)
        g_p = g.clone()
        g_p.x = g.x[p]
        inv = torch.empty_like(p); inv[p] = torch.arange(g.num_nodes, device=device)
        g_p.edge_index = inv[g.edge_index]

        emb_orig = model(g.x, g.edge_index, return_emb=True)
        emb_perm = model(g_p.x, g_p.edge_index, return_emb=True)

        # "readout": take node 0 vector (bad readout; not invariant)
        r0_orig = emb_orig[0]
        r0_perm = emb_perm[0]
        if g.num_nodes > 1:
            assert not torch.allclose(r0_orig, r0_perm, atol=1e-6), \
                "Counterexample failed: 'node 0' readout appeared invariant unexpectedly."

    print("Graph-level permutation invariance passed (sum/mean/max, MUTAG, 3 perms each).")
    print("Counterexample (node-0 readout) correctly failed invariance.")

if __name__ == "__main__":
    test_equivariance_node()
    test_invariance_graph()
