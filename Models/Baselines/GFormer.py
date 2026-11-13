"""
GFormer: Graph Transformer for Recommendation.
Reference: Li et al. (2023) SIGIR.
"""

import torch
from torch import nn
import torch.nn.functional as F


class GFormer(nn.Module):
    def __init__(self, num_users, num_items, latdim=128, gnn_layer=2, num_heads=4):
        super(GFormer, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latdim = latdim
        self.gnn_layer = gnn_layer
        self.num_heads = num_heads

        init = nn.init.xavier_uniform_
        self.user_embedding = nn.Parameter(init(torch.empty(num_users, latdim)))
        self.item_embedding = nn.Parameter(init(torch.empty(num_items, latdim)))

        # Graph transformer layers for selective augmentation
        self.attention = nn.MultiheadAttention(latdim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(latdim, latdim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latdim * 2, latdim)
        )
        self.norm1 = nn.LayerNorm(latdim)
        self.norm2 = nn.LayerNorm(latdim)

    def forward(self, adj):
        ini_embeds = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        embeds = [ini_embeds]
        for _ in range(self.gnn_layer):
            embeds.append(torch.spmm(adj, embeds[-1]))
        main_embeds = sum(embeds)
        return main_embeds[:self.num_users], main_embeds[self.num_users:]

    def forward_transformer(self, embeddings, batch_size=256):
        """Graph transformer for selective interaction augmentation."""
        # Process in batches for memory efficiency
        x = embeddings[:batch_size].unsqueeze(0)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x.squeeze(0)
