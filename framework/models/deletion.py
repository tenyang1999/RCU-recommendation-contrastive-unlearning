import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# from . import GCN, GAT, GIN, RGCN, RGAT
from .lightGCN import LightGCN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeletionLayer(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim))
        # self.deletion_weight = nn.Parameter(torch.eye(dim, dim))
        # init.xavier_uniform_(self.deletion_weight)
    
    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask
        
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)

            return new_rep

        return x
    
class FairLayer(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
        # self.deletion_weight = nn.Parameter(torch.eye(dim, dim))
        # init.xavier_uniform_(self.deletion_weight)
    
    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask
        
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)

            return new_rep

        return x
    
class DeletionLayerKG(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
    
    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask
        
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)

            return new_rep

        return x
    
class LightGCNDelete(LightGCN):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args.out_dim, mask_1hop)
        self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

        self.emb = torch.nn.Embedding(
            num_embeddings=self.num_users+self.num_items , embedding_dim=args.out_dim)

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        # with torch.no_grad():
        if x == None:
            x = self.emb.weight.to(device)

        embs = [x]

        x1 = self.conv1(x, edge_index)
        
        x1 = self.deletion1(x1, mask_1hop)

        embs.append(x1)
        # x = F.relu(x1)
        
        x2 = self.conv2(x1, edge_index)
        x2 = self.deletion2(x2, mask_2hop)

        embs.append(x2)

        embs = torch.stack(embs, dim=1)
        out = torch.mean(embs, dim=1)

        if return_all_emb:
            return x, x1, x2
        
        return out
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)

class LightGCNFair(LightGCN):
    def __init__(self, args, mask_group=None, **kwargs):
        super().__init__(args)
        self.fair = FairLayer(args.out_dim, mask_group)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

        self.emb = torch.nn.Embedding(
            num_embeddings=self.num_users+self.num_items , embedding_dim=args.out_dim)

    def forward(self, x, edge_index, mask_group=None, return_all_emb=False):
        # with torch.no_grad():
        if x == None:
            x = self.emb.weight.to(device)

        embs = [x]

        x1 = self.conv1(x, edge_index)
        embs.append(x1)

        x2 = self.conv2(x1, edge_index)
        embs.append(x2)
        embs = torch.stack(embs, dim=1)
        
        out = torch.mean(embs, dim=1)
        out = self.fair(out, mask_group=mask_group)

        if return_all_emb:
            return x, x1, x2
        
        return out
    
    def get_original_embeddings(self, x, edge_index, return_all_emb=False):
        return super().forward(x, edge_index, return_all_emb)

# class GCNDelete(GCN):
#     def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
#         super().__init__(args)
#         self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
#         self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

#         self.conv1.requires_grad = False
#         self.conv2.requires_grad = False

#     def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
#         # with torch.no_grad():
#         x1 = self.conv1(x, edge_index)
        
#         x1 = self.deletion1(x1, mask_1hop)

#         x = F.relu(x1)
        
#         x2 = self.conv2(x, edge_index)
#         x2 = self.deletion2(x2, mask_2hop)

#         if return_all_emb:
#             return x1, x2
        
#         return x2
    
#     def get_original_embeddings(self, x, edge_index, return_all_emb=False):
#         return super().forward(x, edge_index, return_all_emb)

# class GATDelete(GAT):
#     def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
#         super().__init__(args)
#         self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
#         self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

#         self.conv1.requires_grad = False
#         self.conv2.requires_grad = False

#     def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
#         with torch.no_grad():
#             x1 = self.conv1(x, edge_index)
#         x1 = self.deletion1(x1, mask_1hop)

#         x = F.relu(x1)
        
#         x2 = self.conv2(x, edge_index)
#         x2 = self.deletion2(x2, mask_2hop)

#         if return_all_emb:
#             return x1, x2
        
#         return x2
    
#     def get_original_embeddings(self, x, edge_index, return_all_emb=False):
#         return super().forward(x, edge_index, return_all_emb)

# class GINDelete(GIN):
#     def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
#         super().__init__(args)
#         self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
#         self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

#         self.conv1.requires_grad = False
#         self.conv2.requires_grad = False

#     def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None, return_all_emb=False):
#         with torch.no_grad():
#             x1 = self.conv1(x, edge_index)
        
#         x1 = self.deletion1(x1, mask_1hop)

#         x = F.relu(x1)
        
#         x2 = self.conv2(x, edge_index)
#         x2 = self.deletion2(x2, mask_2hop)

#         if return_all_emb:
#             return x1, x2

#         return x2
    
#     def get_original_embeddings(self, x, edge_index, return_all_emb=False):
#         return super().forward(x, edge_index, return_all_emb)

# class RGCNDelete(RGCN):
#     def __init__(self, args, num_nodes, num_edge_type, mask_1hop=None, mask_2hop=None, **kwargs):
#         super().__init__(args, num_nodes, num_edge_type)
#         self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
#         self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

#         self.node_emb.requires_grad = False
#         self.conv1.requires_grad = False
#         self.conv2.requires_grad = False

#     def forward(self, x, edge_index, edge_type, mask_1hop=None, mask_2hop=None, return_all_emb=False):
#         with torch.no_grad():
#             x = self.node_emb(x)
#             x1 = self.conv1(x, edge_index, edge_type)
        
#         x1 = self.deletion1(x1, mask_1hop)

#         x = F.relu(x1)
        
#         x2 = self.conv2(x, edge_index, edge_type)
#         x2 = self.deletion2(x2, mask_2hop)

#         if return_all_emb:
#             return x1, x2
        
#         return x2
    
#     def get_original_embeddings(self, x, edge_index, edge_type, return_all_emb=False):
#         return super().forward(x, edge_index, edge_type, return_all_emb)

# class RGATDelete(RGAT):
#     def __init__(self, args, num_nodes, num_edge_type, mask_1hop=None, mask_2hop=None, **kwargs):
#         super().__init__(args, num_nodes, num_edge_type)
#         self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
#         self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

#         self.node_emb.requires_grad = False
#         self.conv1.requires_grad = False
#         self.conv2.requires_grad = False

#     def forward(self, x, edge_index, edge_type, mask_1hop=None, mask_2hop=None, return_all_emb=False):
#         with torch.no_grad():
#             x = self.node_emb(x)
#             x1 = self.conv1(x, edge_index, edge_type)
        
#         x1 = self.deletion1(x1, mask_1hop)

#         x = F.relu(x1)
        
#         x2 = self.conv2(x, edge_index, edge_type)
#         x2 = self.deletion2(x2, mask_2hop)

#         if return_all_emb:
#             return x1, x2
        
#         return x2
    
#     def get_original_embeddings(self, x, edge_index, edge_type, return_all_emb=False):
#         return super().forward(x, edge_index, edge_type, return_all_emb)
