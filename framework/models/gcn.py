import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, args, embedding_dim=128, **kwargs):
        super().__init__()

        self.conv1 = GCNConv(embedding_dim, 64)
        self.conv2 = GCNConv(64, 32)
        # self.dropout = nn.Dropout(args.dropout)
        # self.out = nn.Linear(args.hidden_dim, 3)
        self.num_users, self.num_items = args.num_users, args.num_items

        self.emb = torch.nn.Embedding(
            num_embeddings=self.num_users+self.num_items , embedding_dim=embedding_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.emb.weight)
        # for conv in self.convs:
        #     conv.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


    def forward(self, x, edge_index, return_all_emb=False):

        if x == None:
            x = self.emb.weight

        xs = [x]
        
        # x1 = self.conv1(x, edge_index)
        # x1 = F.relu(x1)
        # xs.append(x1)
        # # x = self.dropout(x)
        # x2 = self.conv2(x1, edge_index)
        # xs.append(x2)
        x1 = self.conv1(x, edge_index)
        xs.append(x1)
        x = F.relu(x1)
        # x = self.dropout(x)
        x2 = self.conv2(x, edge_index)
        xs.append(x2)
                        
        if return_all_emb:
            return x2, xs
        
        # out = self.out(x)
        
        return x2

    def decode(self, z, edge_label_index) -> torch.Tensor:
        
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
