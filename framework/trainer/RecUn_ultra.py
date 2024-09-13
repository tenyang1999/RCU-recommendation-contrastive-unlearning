import os
import time
# import wandb
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling, k_hop_subgraph
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from sklearn.metrics.pairwise import cosine_similarity
from .base import Trainer
# from 
# from ..evaluation import *
from ..utils import *
from .. import diffusion as gd
import scipy.sparse as sp

import gc

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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

    
class RecUnUltraTrainer(Trainer):

    def train(self, model, data, optimizer, args, popularity_bias= None, z_ori=None, attack_model_all=None, attack_model_sub=None):
        
        return self.train_minibatch(model, data, optimizer, args, popularity_bias, z_ori, attack_model_all, attack_model_sub)
    
    def k_hop_batch_neighbor(self, data, batch_data, num_nodes):

        # Edges in S_Df
        _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
            batch_data.flatten().unique(), 
            2, 
            data,
            num_nodes=num_nodes)


        # Nodes in S_Df
        _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
            batch_data.flatten().unique(), 
            1, 
            data,
            num_nodes=num_nodes)
        
        sdf_node_1hop = torch.zeros(num_nodes, dtype=torch.bool)
        sdf_node_2hop = torch.zeros(num_nodes, dtype=torch.bool)

        sdf_node_1hop[one_hop_edge.flatten().unique()] = True
        sdf_node_2hop[two_hop_edge.flatten().unique()] = True

        assert sdf_node_1hop.sum() == len(one_hop_edge.flatten().unique())
        assert sdf_node_2hop.sum() == len(two_hop_edge.flatten().unique())

        return sdf_node_1hop, sdf_node_2hop, one_hop_mask
    
    def train_minibatch(self, model, data, optimizer, args, popularity_bias= None, z_ori=None, attack_model_all=None, attack_model_sub=None):

        best_recall = best_precision = best_ndcg = best_metric = 0
              
        # model = model.to(device)
        data = data.to(device)
        best_metric = 10000
        best_metric = best_auc = best_recall =0
        start_time = time.time()

        if self.with_fea:
            dim_users, dim_items = data['user'].x.size(1), data['item'].x.size(1)
            users_lin = torch.rand(dim_users, args.out_dim).to(device)
            items_lin = torch.rand(dim_items, args.out_dim).to(device)
            user_emb = torch.mm(data['user'].x, users_lin)
            item_emb = torch.mm(data['item'].x, items_lin)
            X = torch.cat([user_emb, item_emb])
            x = torch.tensor(X, requires_grad= True)
        else:
            x = None
        
        dr_edge = data.train_pos_edge_index[:, data.dr_mask]
        df_edge = data.train_pos_edge_index[:, data.df_mask]
        early_stop_cnt = 0
        for epoch in trange(args.epochs, desc='Unlerning'):

            model.train()

            loader = torch.utils.data.DataLoader(
                range(df_edge.size(1)),
                shuffle=True,
                batch_size=args.batch_size,
            )

            idx = torch.randperm(dr_edge.size(1))[:df_edge.size(1)]

            sub_dr_edge = dr_edge[:,idx]

            epoch_loss_e, epoch_loss_l, epoch_loss_f= 0, 0, 0
            epoch_loss, epoch_time = 0, 0
            for step, ind  in enumerate(tqdm(loader, leave=False)):
                start = time.time()

                optimizer.zero_grad()

                pos_edge_index = sub_dr_edge[:, ind].to(device)
                forget_edge_index = df_edge[:, ind].to(device)
                user_indices, pos_item_indices = pos_edge_index
                
                neg_candidates = np.arange( data.num_items)
                neg_items = np.random.choice(neg_candidates, (ind.size(0), args.negative_num), replace = True)
                neg_items = torch.from_numpy(neg_items).to(device)
                model.zero_grad()
                # print(pos_item_indices-data.num_users)
                loss_e = model(user_indices, pos_item_indices-data.num_users, neg_items)
                # print(model.user_embeds.weight)

                sdf_node_1hop_mask, sdf_node_2hop_mask, one_hop_mask = self.k_hop_batch_neighbor(dr_edge, forget_edge_index, data.num_users+data.num_items)
                sdf_node_1hop_mask, sdf_node_2hop_mask, one_hop_mask = sdf_node_1hop_mask.to(device), sdf_node_2hop_mask.to(device), one_hop_mask.to(device)
                
                z_df = z_ni = torch.concat([model.user_embeds.weight, model.item_embeds.weight])
                # print(two_hop_mask.sum())
                row, col = forget_edge_index
                nodes = torch.concat([row, col]).unique()
                # print(two_hop_mask.sum())
                row, col = dr_edge[:, one_hop_mask]
                ni_nodes = torch.concat([row, col]).unique()

                loss_l = semi_loss(0.6, z_df, z_ni, z_ori, nodes, ni_nodes)

                alpha =  0.6
                loss = alpha*loss_e + (1-alpha)* torch.abs(loss_l)
                # loss = loss_e + 10* torch.abs(loss_l)


                loss.backward()
                optimizer.step()
                end_time = time.time()
                epoch_loss_e += loss_e.item()
                epoch_loss_l += loss_l.item()
                epoch_loss += loss.item()
                epoch_time += end_time - start

                # if epoch == 0 and step==0 :
                #     dr_edge = self.rewiring( z_df, dr_edge, data.num_nodes)
                # print(z)


            if (epoch+1) % args.valid_freq == 0:

                score = nora_ultra(args, model, data)
                user_score, item_score = score[:data.num_users], score[data.num_users:]
                
                user_score = torch.log(torch.Tensor(user_score))
                item_score = torch.log(torch.Tensor(item_score))
                # score = [user_score, item_score]
                
                precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, valid_log= self.eval(args, model, data, group= popularity_bias, stage='val', score=item_score)
                # valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')
                if step ==0:
                    step =1
                train_log = {
                    'epoch': epoch,
                    'train_loss': epoch_loss / step,
                    'train_loss_l': epoch_loss_l / step,
                    'train_loss_e': epoch_loss_e / step,
                    'train_loss_f': epoch_loss_f / step, 
                    
                    'train_time': epoch_time / step,
                }
                
                for log in [train_log, valid_log]:
                    # wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))
                    self.trainer_log['log'].append(log)

                if best_auc == 0 :
                    best_metric = 1
                else:
                    best_metric =  ( df_auc-best_auc)/best_auc+(recall- best_recall)/ best_recall

                if 0 < best_metric:
                # if abs(args.retrain_auc-df_auc) < best_metric :
                
                    best_recall = recall
                    best_epoch = epoch
                    best_precision = precision
                    best_ndcg = ndcg
                    best_auc = df_auc
                    # best_metric = abs(args.retrain_auc-df_auc)
                    # best_metric = recall+df_auc
                    best_epoch = epoch
                    best_time =  time.time()-start_time
                    print(f'Save best checkpoint at epoch {epoch:04d}. recall = {recall:.4f}. df_auc = {df_auc:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    # torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
                    early_stop_cnt = 0
                else:
                    early_stop_cnt +=1

            if early_stop_cnt >= 10:
                print(f'early stop training at epoch {epoch}')
                break                    

        self.trainer_log['best_training_time'] = best_time
        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))



