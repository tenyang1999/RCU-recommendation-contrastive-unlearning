import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv

# defines LightGCN model
class LightGCN(torch.nn.Module):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, args, embedding_dim=64, num_layers=2, **kwargs,):
        """Initializes LightGCN Model
        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.num_users, self.num_items = args.num_users, args.num_items

        self.emb = torch.nn.Embedding(
            num_embeddings=self.num_users+self.num_items , embedding_dim=self.embedding_dim)
        # try :
        #     self.dim_users, self.dim_items = args.dim_users, args.dim_items
        #     self.users_lin = torch.nn.Linear(self.dim_users, self.embedding_dim)
        #     self.items_lin = torch.nn.Linear(self.dim_items, self.embedding_dim) 
        # except:
        #     pass

        # self.users_emb = torch.nn.Embedding(
        #     num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0
        # self.items_emb = torch.nn.Embedding(
        #     num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0
        


        # self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])
        self.conv1 = LGConv(**kwargs) #ModuleList([LGConv(**kwargs) for _ in range(num_layers)])
        self.conv2 = LGConv(**kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.emb.weight)
        # for conv in self.convs:
        #     conv.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, return_all_emb= False) -> Tensor:

        """Returns the embedding of nodes in the graph."""
        if x == None:
            emb_0 = self.emb.weight
        else:
            emb_0 = x
        # if items_x == None:
        #     items_x = self.items_emb.weight
        # else:
        #     items_x = self.items_lin(items_x)

        # emb_0 = torch.cat([users_x, items_x]) # E^0
        embs = [emb_0]
        emb_k = emb_0

        # for i in range(self.num_layers):
        #     emb_k = self.convs[i](emb_k, edge_index)
        #     embs.append(emb_k)
        emb_1 = self.conv1(emb_0, edge_index)
        embs.append(emb_1)
        
        emb_2 = self.conv2(emb_1, edge_index)
        embs.append(emb_2)
        
        all_embs = torch.stack(embs, dim=1)
        out = torch.mean(all_embs, dim=1) # E^K 

        if return_all_emb:
            return out, embs

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
    
