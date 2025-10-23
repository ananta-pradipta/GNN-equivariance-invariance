# model.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCN2Layers(nn.Module):
    """
    Two-layer GCN. In eval() it is deterministic (no dropout).
    Set cached=False to recompute normalization after permutations (required).
    """
    def __init__(self, in_dim, hid_dim=64, out_dim=None, dropout=0.5, cached=False):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim, cached=cached, normalize=True)
        self.conv2 = GCNConv(hid_dim, hid_dim, cached=cached, normalize=True)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.mlp_head = None
        if out_dim is not None:
            self.mlp_head = nn.Linear(hid_dim, out_dim)

    @torch.no_grad()
    def forward(self, x, edge_index, return_emb=False):
        # x: [N, in_dim], edge_index: [2, E]
        h = self.conv1(x, edge_index)
        h = self.act(h)
        h = self.dropout(h)  # dropped in eval()
        h = self.conv2(h, edge_index)
        h = self.act(h)
        if return_emb:
            return h
        if self.mlp_head is not None:
            return self.mlp_head(h)
        return h
