import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss
# from LightGCN import LightGCN
from torch_geometric.nn.conv import LGConv

class SimGCL(torch.nn.Module):
    def __init__(self, args, embedding_dim=64, num_layers=2, **kwargs,):
        super(SimGCL, self).__init__()

        self.eps = float(args.eps)
        self.num_layers = num_layers

        self.num_users, self.num_items = args.num_users, args.num_items

        # self.user_embeddings = torch.nn.Embedding(num_embeddings=self.num_users , embedding_dim = embedding_dim)
        # self.item_embeddings = torch.nn.Embedding(num_embeddings=self.num_items , embedding_dim = embedding_dim)
        self.emb = torch.nn.Embedding(
            num_embeddings=self.num_users+self.num_items , embedding_dim=embedding_dim)
        # self.ego_embeddings = torch.cat([self.user_embeddings, self.item_embeddings], dim=0)

        self.conv1 = LGConv(**kwargs) #ModuleList([LGConv(**kwargs) for _ in range(num_layers)])
        self.conv2 = LGConv(**kwargs)

        self.reset_parameters()

                 #encoding
        # self.main_user_embeddings, self.main_item_embeddings = self.LightGCN_encoder(x=None, edge_index= edge_index, return_all_emb= False)
        # self.perturbed_user_embeddings1, self.perturbed_item_embeddings1 = self.perturbed_LightGCN_encoder(x=None, edge_index= edge_index, return_all_emb= False)
        # self.perturbed_user_embeddings2, self.perturbed_item_embeddings2 = self.perturbed_LightGCN_encoder(x=None, edge_index= edge_index, return_all_emb= False)
        
        # self.neg_idx = None
        # #adjaceny matrix
        # self.norm_adj = None

        # self.batch_neg_item_emb = self.main_item_embeddings[self.neg_idx]
        # self.batch_user_emb = self.main_user_embeddings[self.u_idx]
        # self.batch_pos_item_emb = self.main_item_embeddings[self.v_idx]

    def reset_parameters(self):

        # torch.nn.init.xavier_uniform_(self.user_embeddings.weight)
        # torch.nn.init.xavier_uniform_(self.item_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.emb.weight)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, return_all_emb= False, perturbed = False):
        # emb, adj, n_layers):.
        if x == None:
            emb_0 = self.emb.weight
        else:
            emb_0 = x

        embs = [emb_0]
        emb_k = emb_0

        emb_1 = self.conv1(emb_0, edge_index)
        if perturbed :
            random_noise = torch.rand_like(emb_1)
            emb_1 += torch.sign(emb_1) * F.normalize(random_noise, p=2, dim=1) * self.eps
        embs.append(emb_1)
        
        emb_2 = self.conv2(emb_1, edge_index)
        if perturbed:
            random_noise = torch.rand_like(emb_2)
            emb_2 += torch.sign(emb_2) * F.normalize(random_noise, p=2, dim=1) * self.eps
        embs.append(emb_2)
        
        all_embs = torch.stack(embs, dim=1)
        out = torch.mean(all_embs, dim=1) # E^K 

        if return_all_emb:
            return out, embs
        
        # users_emb, items_emb = out[:self.num_users], out[self.num_users:]
        return out
    
    def decode(self, z, edge_label_index) -> Tensor:
        
        """Computes rankings for pairs of nodes.

        Args:
            edge_index (torch.Tensor or SparseTensor): Edge tensor specifying
                the connectivity of the graph.
            edge_label_index (torch.Tensor, optional): Edge tensor specifying
                the node pairs for which to compute rankings or probabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used instead. (default: :obj:`None`)
            edge_weight (torch.Tensor, optional): The weight of each edge in
                :obj:`edge_index`. (default: :obj:`None`)
        """
        # out = self.get_embedding(x, edge_index)

        out_src = z[edge_label_index[0]]
        out_dst = z[edge_label_index[1]]

        return (out_src * out_dst).sum(dim=-1)    
    
