import os
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import copy
import numpy as np
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, degree, scatter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch_geometric.data import Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BPRLoss(_Loss):
    def __init__(self, lambda_reg: float = 0, **kwargs):
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, rank, indices, emb) -> torch.Tensor:
        r"""Compute the mean Bayesian Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`positives` vector and i-th entry
            in the :obj:`negatives` entry should correspond to the same
            entity (*.e.g*, user), as the BPR is a personalized ranking loss.

        Args:
            positives (Tensor): The vector of positive-pair rankings.
            negatives (Tensor): The vector of negative-pair rankings.
            parameters (Tensor, optional): The tensor of parameters which
                should be used for :math:`L_2` regularization
                (default: :obj:`None`).
        """
        user_indices, pos_item_indices, neg_item_indices = indices
        user = emb[user_indices, :] 
        pos = emb[pos_item_indices, :] #pos
        neg = emb[neg_item_indices, :] #neg

        positives, negatives = rank.chunk(2)
        log_prob = F.logsigmoid(positives - negatives).mean()

        regularization = 0
        if self.lambda_reg != 0:
            regularization = self.lambda_reg * (user.norm(p=2, dim=1).pow(2).sum() + 
                                                pos.norm(p=2, dim=1).pow(2).sum() + 
                                                neg.norm(p=2, dim=1).pow(2).sum())
            regularization = regularization / positives.size(0)

        return -log_prob + regularization

def sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

def semi_loss(tau, z_f: torch.Tensor, z_ni: torch.Tensor, z_ori: torch.Tensor, df_nodes, ni_nodes):
    f = lambda x: torch.exp(x / tau)

    between_sim = f(sim(z_ni[ni_nodes], z_ori[ni_nodes]))
    refl_sim = f(sim(z_f[df_nodes], z_ori[df_nodes]))
    positive_pairs = between_sim.mean()
    negative_pairs = refl_sim.mean()

    loss = -torch.log(positive_pairs / negative_pairs + positive_pairs)
    
    return loss


def nora(args, model, data):
    
    model = model.cpu()
    model.eval()   # Using eval() can stabilize the results

    if hasattr(data, 'dtrain_mask'):
        mask = data.dtrain_mask
    else:
        mask = data.dr_mask
    
    strange = ['AmazonBook', 'subDouban', 'subAmazonBook']
    if args.dataset in strange :
        num_nodes = data.num_users + data.num_items 
    else:
        num_nodes = data.num_nodes 
    dr_edge = data.train_pos_edge_index[:, mask].cpu()

    # graph_ori = Data(edge_index=dr_edge,  num_nodes=num_nodes)
    # graph = graph_ori.clone()

    edge_index, _ = remove_self_loops(dr_edge)
    row, col = edge_index
    deg = degree(col, num_nodes=num_nodes, dtype=torch.float)
    mean_deg = deg.float().mean()

    x = None
    if args.gnn == 'UltraGCN':
        out = torch.concat([model.user_embeds.weight, model.item_embeds.weight])
    else:
        out, hidden_list = model(x, dr_edge, return_all_emb= True)
    
    row, col = dr_edge
    direct_mask = row < col
    dr_edge = dr_edge[:, direct_mask]

    out = (out[dr_edge[0]] * out[dr_edge[1]]).sum(dim=-1)

    if args.gnn == 'gcn':
      drop_out = 20
    else:
      drop_out = 1
    # if args.dataset == 'Douban':
    #   drop_out /=1000  
    out = torch.sigmoid(out/drop_out)

    # dr_edge = to_undirected(dr_edge)
    link_info1 = torch.cat([dr_edge[0], dr_edge[1]])
    link_info2 = torch.cat([dr_edge[1], dr_edge[0]])
    edge_index = torch.stack([link_info1, link_info2], dim=0)
    edge_attr = torch.cat([out, out])
    

    graph = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    row, col = graph.edge_index

    graph.node_score = scatter(graph.edge_attr, row, reduce='sum')
    out = graph.node_score

    for hs in hidden_list:
        hs.retain_grad()
    out.backward(gradient=out, retain_graph=True)
    hidden_grad_list = []
    for i in range(len(hidden_list)):
        hidden_grad_list.append(hidden_list[i].grad.detach())


    gradient = torch.zeros(num_nodes)
    rate = 1.0
    # assert len(hidden_list) == args.num_layers + 1
    for i in range(len(hidden_list) - 2, -1, -1):
        new_grad = hidden_grad_list[i] * hidden_list[i]
        new_grad = torch.norm(new_grad, p=args.grad_norm, dim=1)
        new_grad = new_grad * deg / (deg + args.self_buff)
        # new_grad = new_grad * torch.abs(deg - args.self_buff)/deg 
        gradient = gradient + new_grad * rate
        rate = rate * (1 - deg / (num_nodes - 1) / (mean_deg + args.self_buff))

    assert (gradient < 0).sum() == 0
    deg_delta1 = 1 / torch.sqrt(deg - 1) - 1 / torch.sqrt(deg)
    deg_delta2 = 1 / (deg-1) - 1 / deg
    deg_delta1[deg_delta1 == np.nan] = 1.0
    deg_delta2[deg_delta2 == np.nan] = 1.0
    deg_delta1[deg_delta1.abs() == np.inf] = 1.0
    deg_delta2[deg_delta2.abs() == np.inf] = 1.0
    deg_delta = args.k1 * deg_delta1 + (1 - args.k1) * deg_delta2
    deg_inv = args.k2[0] / torch.sqrt(deg) + args.k2[1] / deg + (1 - args.k2[0] - args.k2[1])
    
    deg_inv_sum = scatter(deg_inv[row], col, reduce='sum')
    deg_gather = deg_inv_sum

    if  deg_gather.size(0) < deg_inv.size(0):
        pad_size = deg_inv.size(0)- deg_gather.size(0)
        deg_gather = F.pad(deg_gather, (0, pad_size), 'constant', 0)

    deg_delta = deg_gather * deg_delta
    deg_gather = scatter(deg_delta[row], col, reduce='sum')

    if  deg_gather.size(0) < deg_inv.size(0):
        pad_size = deg_inv.size(0)- deg_inv_sum.size(0)
        deg_gather = F.pad(deg_gather, (0, pad_size), 'constant', 0)
    
    deg_gather = deg_gather / deg_gather.mean() * gradient.mean()  # Normalize
    influence = gradient + args.k3 * deg_gather
    influence = influence.abs().detach().cpu().numpy()

    model = model.to(device)

    return influence


def plot_result(args,popularity_bias, rf):
    popularity_of_item = popularity_bias['popularity_of_item']

    best_rf,  diverse_rf, niche_rf = rf
    df = pd.DataFrame(np.array(popularity_of_item), columns=['Popularity of item'])
    df['best_rf'] = best_rf
    df['diverse_rf'] = diverse_rf
    df['niche_rf'] = niche_rf
    # df['rf'] = best_rf+diverse_rf+niche_rf
    fig, ax = plt.subplots()
    sns.set_style("darkgrid")
    # sns.regplot(data=df, x='Popularity of item', y='rf', line_kws=dict(color="r"), truncate=False)
    sns.regplot(data=df, x='Popularity of item', y='best_rf', color="green")
    sns.regplot(data=df, x='Popularity of item', y='diverse_rf')
    sns.regplot(data=df, x='Popularity of item', y='niche_rf', color='orange')
    sns.scatterplot(data=df, x='Popularity of item', y='best_rf', color="green", label="best")
    sns.scatterplot(data=df, x='Popularity of item', y='diverse_rf', label="diverse")
    sns.scatterplot(data=df, x='Popularity of item', y='niche_rf', color='orange', label="niche")

    ax.legend()
    # ax.set_ylim(top=150)
    ax.set_xlabel('Popularity of item')
    ax.set_ylabel('Recommendation Frequency')
    plt.savefig(os.path.join(args.checkpoint_dir, "Popularity_bias.png"))

# for UltraGCN
def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero = False):
    print('Computing \\Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)	# I * I
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    all_ii_constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))

    print('Computation \\Omega OK!')
    
    
    return res_mat.long(), res_sim_mat.float()




def nora_ultra(args, model, data):
    
    model = model.cpu()
    model.eval()   # Using eval() can stabilize the results

    if hasattr(data, 'dtrain_mask'):
        mask = data.dtrain_mask
    else:
        mask = data.dr_mask
    
    num_nodes = data.num_users + data.num_items if args.dataset == 'AmazonBook' else data.num_nodes 
    
    dr_edge = data.train_pos_edge_index[:, mask].cpu()

    edge_index, _ = remove_self_loops(dr_edge)
    row, col = to_undirected(edge_index)
    deg = degree(col, num_nodes=num_nodes, dtype=torch.float)
    mean_deg = deg.float().mean()

    x = None
    out = torch.concat([model.user_embeds.weight, model.item_embeds.weight])
    
    row, col = dr_edge
    direct_mask = row < col
    dr_edge = dr_edge[:, direct_mask]

    out = (out[dr_edge[0]] * out[dr_edge[1]]).sum(dim=-1)
    
    out = torch.sigmoid(out)

    # dr_edge = to_undireted(dr_edge)
    link_info1 = torch.cat([dr_edge[0], dr_edge[1]])
    link_info2 = torch.cat([dr_edge[1], dr_edge[0]])
    edge_index = torch.stack([link_info1, link_info2], dim=0)
    edge_attr = torch.cat([out, out])
    

    graph = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    row, col = graph.edge_index

    graph.node_score = scatter(graph.edge_attr, row, reduce='sum')
    out = graph.node_score

    model.user_embeds.weight.retain_grad()
    model.item_embeds.weight.retain_grad()

    out.backward(gradient=out, retain_graph=True)

    grad = torch.concat([model.user_embeds.weight.grad.detach(), model.item_embeds.weight.grad.detach()])

    hidden_grad_list = [grad]
    hidden_list = [ torch.concat([model.user_embeds.weight, model.item_embeds.weight])]
    # for i in range(len(hidden_list)):
    #     hidden_grad_list.append(hidden_list[i].grad.detach())
    # print(hidden_grad_list[0].device, hidden_list[0].device)

    gradient = torch.zeros(num_nodes)
    rate = 1.0
    # assert len(hidden_list) == args.num_layers + 1
    # for i in range(len(hidden_list) - 2, -1, -1):
    new_grad = hidden_grad_list[0] * hidden_list[0]
    new_grad = torch.norm(new_grad, p=args.grad_norm, dim=1)
    new_grad = new_grad * deg / (deg + args.self_buff)
    # new_grad = new_grad * torch.abs(deg - args.self_buff)/deg 
    gradient = gradient + new_grad * rate
    rate = rate * (1 - deg / (num_nodes - 1) / (mean_deg + args.self_buff))

    assert (gradient < 0).sum() == 0
    deg_delta1 = 1 / torch.sqrt(deg - 1) - 1 / torch.sqrt(deg)
    deg_delta2 = 1 / (deg-1) - 1 / deg
    deg_delta1[deg_delta1 == np.nan] = 1.0
    deg_delta2[deg_delta2 == np.nan] = 1.0
    deg_delta1[deg_delta1.abs() == np.inf] = 1.0
    deg_delta2[deg_delta2.abs() == np.inf] = 1.0
    deg_delta = args.k1 * deg_delta1 + (1 - args.k1) * deg_delta2
    deg_inv = args.k2[0] / torch.sqrt(deg) + args.k2[1] / deg + (1 - args.k2[0] - args.k2[1])
    
    deg_inv_sum = scatter(deg_inv[row], col, reduce='sum')
    deg_gather = deg_inv_sum

    if  deg_gather.size(0) < deg_inv.size(0):
        pad_size = deg_inv.size(0)- deg_gather.size(0)
        deg_gather = F.pad(deg_gather, (0, pad_size), 'constant', 0)

    deg_delta = deg_gather * deg_delta
    deg_gather = scatter(deg_delta[row], col, reduce='sum')

    if  deg_gather.size(0) < deg_inv.size(0):
        pad_size = deg_inv.size(0)- deg_inv_sum.size(0)
        deg_gather = F.pad(deg_gather, (0, pad_size), 'constant', 0)
    
    deg_gather = deg_gather / deg_gather.mean() * gradient.mean()  # Normalize
    influence = gradient + args.k3 * deg_gather
    influence = influence.abs().detach().cpu().numpy()

    model = model.to(device)

    return influence





