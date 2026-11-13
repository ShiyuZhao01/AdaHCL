"""
AutoCF: Automated Self-supervised Learning for Recommendation.
Reference: Xia et al. (2023) WWW.
"""

import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy


class AutoCF(nn.Module):
    def __init__(self, num_users, num_items, latdim=128, gnn_layer=2):
        super(AutoCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latdim = latdim
        self.gnn_layer = gnn_layer

        init = nn.init.xavier_uniform_
        self.user_embedding = nn.Parameter(init(torch.empty(num_users, latdim)))
        self.item_embedding = nn.Parameter(init(torch.empty(num_items, latdim)))

        # Masked graph autoencoder for automated augmentation
        self.mask_predictor = nn.Sequential(
            nn.Linear(latdim * 2, latdim),
            nn.ReLU(inplace=True),
            nn.Linear(latdim, 1),
            nn.Sigmoid()
        )

    def forward(self, adj):
        ini_embeds = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        embeds = [ini_embeds]
        for _ in range(self.gnn_layer):
            embeds.append(torch.spmm(adj, embeds[-1]))
        main_embeds = sum(embeds)
        return main_embeds[:self.num_users], main_embeds[self.num_users:]

    def generate_masked_view(self, adj, embeddings):
        """Generate masked graph view via autoencoder."""
        idxs = adj._indices()
        vals = adj._values()

        src_embeds = embeddings[idxs[0]]
        dst_embeds = embeddings[idxs[1]]
        mask_input = torch.cat([src_embeds, dst_embeds], dim=1)
        mask_scores = self.mask_predictor(mask_input).squeeze()
        mask = (mask_scores > 0.5).float()

        new_vals = vals * mask
        new_adj = torch.sparse.FloatTensor(idxs, new_vals, adj.shape)
        return new_adj
