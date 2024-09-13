import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# from lightGCN import LightGCN

class RecEraser(torch.nn.Module):

    def __init__(self, args, embedding_dim=64, num_layers=2, **kwargs,):

        super().__init__()
        
        self.emb_dim = embedding_dim
        self.num_layers = num_layers

        self.num_users, self.num_items = args.num_users, args.num_items
        self.attention_size = int(self.emb_dim/2)
        self.num_local = 10

        # User attention
        self.WA = nn.Parameter(torch.randn(self.emb_dim, self.attention_size) * torch.sqrt(torch.tensor(2.0 / (self.attention_size + self.emb_dim))))
        self.BA = nn.Parameter(torch.zeros(self.attention_size))
        self.HA = nn.Parameter(torch.ones(self.attention_size, 1) * 0.01)

        # Item attention
        self.WB = nn.Parameter(torch.randn(self.emb_dim, self.attention_size) * torch.sqrt(torch.tensor(2.0 / (self.attention_size + self.emb_dim))))
        self.BB = nn.Parameter(torch.zeros(self.attention_size))
        self.HB = nn.Parameter(torch.ones(self.attention_size, 1) * 0.01)

        # Trans weights
        self.trans_W = nn.Parameter(torch.randn(self.num_local, self.emb_dim, self.emb_dim) * 0.01)
        self.trans_B = nn.Parameter(torch.randn(self.num_local, self.emb_dim) * 0.01)


    def attention_based_agg(self, embs, flag):
        if flag == 0:
            embs_w = torch.exp(
                torch.einsum('abc,ck->abk', F.relu(
                    torch.einsum('abc,ck->abk', embs, self.WA) + self.BA), self.HA))
        else:
            embs_w = torch.exp(
                torch.einsum('abc,ck->abk', F.relu(
                    torch.einsum('abc,ck->abk', embs, self.WB) + self.BB), self.HB))

        embs_w = embs_w / torch.sum(embs_w, dim=1, keepdim=True)
        agg_emb = torch.sum(embs_w * embs, dim=1)

        return agg_emb, embs_w
    
    def _create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)

        mf_loss = torch.mean(F.softplus(-(pos_scores - neg_scores)))

        return mf_loss
    
    def train_single_model(self, local_num):
        pass

    # def train_agg_model(self, emb, indices):
    def forward(self, emb, indices):

        users, pos_items, neg_items = indices

        user_emb, item_emb = emb[:self.num_users, :, :], emb[self.num_users:, :, :]

        u_es = torch.einsum('abc,bcd->abd', user_emb, self.trans_W)+self.trans_B
        i_es = torch.einsum('abc,bcd->abd', item_emb, self.trans_W)+self.trans_B

        u_e, u_w = self.attention_based_agg(u_es, 0)
        i_e, i_w = self.attention_based_agg(i_es, 1)

        u_g_embeddings = u_e[users]
        pos_i_g_embeddings = i_e[pos_items]
        neg_i_g_embeddings = i_e[neg_items]

        mf_loss = self._create_bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        loss = mf_loss

        return loss, u_e, i_e





     