"""
HCCF: Hypergraph Contrastive Collaborative Filtering.
Reference: Xia et al. (2022) SIGIR.
"""

import torch
from torch import nn
import torch.nn.functional as F


class HCCF(nn.Module):
    def __init__(self, num_users, num_items, latdim=128, gnn_layer=2, hyperNum=128, ssl_reg=1e-2, temp=0.5):
        super(HCCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latdim = latdim
        self.gnn_layer = gnn_layer
        self.hyperNum = hyperNum
        self.ssl_reg = ssl_reg
        self.temp = temp

        init = nn.init.xavier_uniform_
        self.user_embedding = nn.Parameter(init(torch.empty(num_users, latdim)))
        self.item_embedding = nn.Parameter(init(torch.empty(num_items, latdim)))
        self.user_hyper = nn.Parameter(init(torch.empty(latdim, hyperNum)))
        self.item_hyper = nn.Parameter(init(torch.empty(latdim, hyperNum)))

    def forward_gcn(self, adj):
        ini_embeds = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        embeds = [ini_embeds]
        for _ in range(self.gnn_layer):
            embeds.append(torch.spmm(adj, embeds[-1]))
        main_embeds = sum(embeds)
        return main_embeds[:self.num_users], main_embeds[self.num_users:]

    def forward_hgcn(self):
        """Hypergraph convolution for global collaborative relations."""
        embeds = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        lats = [embeds]

        uu_hyper = self.user_embedding @ self.user_hyper
        ii_hyper = self.item_embedding @ self.item_hyper

        for _ in range(self.gnn_layer):
            hyper_u = uu_hyper.T @ lats[-1][:self.num_users]
            hyper_i = ii_hyper.T @ lats[-1][self.num_users:]
            hyper_lat = torch.cat([uu_hyper @ hyper_u, ii_hyper @ hyper_i], dim=0)
            lats.append(hyper_lat)

        return sum(lats)[:self.num_users], sum(lats)[self.num_users:]

    def ssl_loss(self, emb1, emb2, users, items):
        """Cross-view contrastive loss (local GCN vs global HGCN)."""
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        u1, i1 = emb1[:self.num_users], emb1[self.num_users:]
        u2, i2 = emb2[:self.num_users], emb2[self.num_users:]

        u_emb1 = F.embedding(users, u1)
        u_emb2 = F.embedding(users, u2)
        i_emb1 = F.embedding(items, i1)
        i_emb2 = F.embedding(items, i2)

        all_emb1 = torch.cat([u_emb1, i_emb1], dim=0)
        all_emb2 = torch.cat([u_emb2, i_emb2], dim=0)

        sim_matrix = torch.exp(torch.mm(all_emb1, all_emb2.T) / self.temp)
        pos_sim = sim_matrix.diag()
        loss = -torch.log(pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-8))
        return loss.mean()
