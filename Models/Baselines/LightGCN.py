"""
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.
Reference: He et al. (2020) SIGIR.
"""

import torch
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, latdim=128, gnn_layer=2):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latdim = latdim
        self.gnn_layer = gnn_layer

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
