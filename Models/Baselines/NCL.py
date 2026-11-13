"""
NCL: Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning.
Reference: Lin et al. (2022) WWW.
"""

import torch
from torch import nn
import torch.nn.functional as F


class NCL(nn.Module):
    def __init__(self, num_users, num_items, latdim=128, gnn_layer=2, ssl_reg=1e-2, temp=0.5):
        super(NCL, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latdim = latdim
        self.gnn_layer = gnn_layer
        self.ssl_reg = ssl_reg
        self.temp = temp

        init = nn.init.xavier_uniform_
        self.user_embedding = nn.Parameter(init(torch.empty(num_users, latdim)))
        self.item_embedding = nn.Parameter(init(torch.empty(num_items, latdim)))

    def forward(self, adj):
        ini_embeds = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        embeds = [ini_embeds]
        for _ in range(self.gnn_layer):
            embeds.append(torch.spmm(adj, embeds[-1]))
        main_embeds = sum(embeds)
        return main_embeds[:self.num_users], main_embeds[self.num_users:]

    def ssl_loss(self, embeddings, neighbors, temp=0.5):
        """Neighborhood-enriched contrastive loss.
        Treats structural neighbors as positive pairs.
        """
        embeddings = F.normalize(embeddings, dim=1)
        neighbor_embeds = F.normalize(embeddings[neighbors], dim=1)

        pos_sim = torch.sum(embeddings * neighbor_embeds.mean(dim=1), dim=-1) / temp
        all_sim = torch.mm(embeddings, embeddings.T) / temp
        all_sim = torch.exp(all_sim).sum(dim=1)
        loss = -torch.log(torch.exp(pos_sim) / (all_sim + 1e-8))
        return loss.mean()
