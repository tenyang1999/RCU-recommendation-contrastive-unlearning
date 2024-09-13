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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def BoundedKLD(logits, truth):
    # print('aaaaaaaaa', truth.shape, truth)
    return 1 - torch.exp(-F.kl_div(F.log_softmax(logits, -1), truth.softmax(-1), None, None, 'batchmean'))

def CosineDistance(logits, truth):
    if len(logits.shape) == 1:
        return 1 - F.cosine_similarity(logits.view(1, -1), truth.view(1, -1))
    else:
        return 1 - F.cosine_similarity(logits, truth)

def get_loss_fct(name):
    if name == 'kld':
        loss_fct = BoundedKLD
    elif name == 'mse':
        loss_fct = nn.MSELoss()
    elif name == 'cosine':
        loss_fct = CosineDistance
    else:
        raise NotImplementedError

    return loss_fct


    
class RecUnTrainer(Trainer):

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

        return sdf_node_1hop, sdf_node_2hop, two_hop_mask
    
    def rewiring(self, z, dr_edge, num_nodes):

        z = z.cpu().detach().numpy()
        
        cs = torch.Tensor(cosine_similarity(z, z))

        # neg = torch.triu(cs < 0, diagonal=1)
        neg = cs < 0
        cut_edge = []
        for tar in dr_edge.t():
            cut_edge.append(neg[tar[0], tar[1]].item())
        cut_edge = torch.Tensor(cut_edge).bool()
        num_ce = cut_edge.sum().int().item()
        print(f'graph rewiring cut_edge numbers: {num_ce}')

        pos = torch.triu(cs, diagonal=1)
        pos_val, ind = torch.sort( pos.reshape(1,-1), dim=1,descending=True)

        add_edge = torch.zeros_like(ind[0], dtype=bool)
        add_edge[ind[0, :num_ce]] = True

        add_edge = add_edge.reshape(num_nodes, num_nodes)
        add_edge = add_edge.nonzero().squeeze().t().to(device)

        rw_edge = dr_edge[:, ~cut_edge]
        dr_edge = torch.concat([rw_edge, add_edge], dim=1)

        return dr_edge
    
    def batch_rewiring(self, z1, z2, dr_edge, num_users):

        # start, end, mid_start, mid_end = interval 

        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        cs = torch.mm(z1, z2.t()) 

        # cs = torch.Tensor(cosine_similarity(z1, z2))
        neg = cs < 0
        cut_edge = []
        for tar in dr_edge.t():
            if tar[0]> num_users:
                cut_edge.append(neg[tar[1], tar[0]-num_users].item())
            else:
                cut_edge.append(neg[tar[0], tar[1]-num_users].item())
            
        cut_edge = torch.Tensor(cut_edge).bool()
        num_ce = cut_edge.sum().int().item()
        print(f'graph rewiring cut_edge numbers: {num_ce}')

        pos = torch.triu(cs, diagonal=1)
        pos_val, ind = torch.sort( pos.reshape(1,-1), dim=1,descending=True)

        add_edge = torch.zeros_like(ind[0], dtype=bool)
        add_edge[ind[0, :num_ce]] = True

        add_edge = add_edge.reshape(z1.size(0), z2.size(0))
        add_edge = add_edge.nonzero().squeeze().t()
        # add_edge[0] += start
        add_edge[1] += z2.size(0)

        rw_edge = dr_edge[:, ~cut_edge]
        dr_edge = torch.concat([rw_edge, add_edge], dim=1)

        return dr_edge
    
    def train_minibatch(self, model, data, optimizer, args, popularity_bias= None, z_ori=None, attack_model_all=None, attack_model_sub=None):

        best_recall = best_precision = best_ndcg = best_metric = 0
        if 'kld' in args.unlearning_model:
            loss_fct = BoundedKLD
        else:
            loss_forget = nn.MSELoss()
            loss_fct = BPRLoss()
            
            
        # model = model.to(device)
        data = data.to(device)
        best_metric = 10000

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
        
        # z_ori = self.get_embedding(model, x, data, on_cpu=True).to(device)
        # z_ori = logits_ori.to(device)
        dr_edge = data.train_pos_edge_index[:, data.dr_mask]
        df_edge = data.train_pos_edge_index[:, data.df_mask]

        for epoch in trange(args.epochs, desc='Unlerning'):

            model.train()

            loader = torch.utils.data.DataLoader(
                range(df_edge.size(1)),
                shuffle=True,
                batch_size=args.batch_size,
            )
            dr_edge = self.rewiring( z_df, dr_edge, data.num_nodes)         

            epoch_loss_e, epoch_loss_l, epoch_loss_f= 0, 0, 0
            epoch_loss, epoch_time = 0, 0
            for step, ind  in enumerate(tqdm(loader, leave=False)):
                start_time = time.time()
                optimizer.zero_grad()
                pos_edge_index = df_edge[:, ind].to(device)
                # row, col = pos_edge_index
                # col -= data.num_users
                # batch = sp.csr_matrix(
                #         (torch.ones(row.size(0)), (row.numpy(), col.numpy())), shape=(data.num_users, data.num_items))

                user_indices, pos_item_indices = pos_edge_index
                neg_item_indices = torch.randint(data.num_users, data.num_users + data.num_items,(ind.numel(), ), device=device)
                indices = [user_indices, pos_item_indices, neg_item_indices]
                neg_edge_index = torch.stack([user_indices, neg_item_indices], dim=0).long()
            
                edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

                sdf_node_1hop_mask, sdf_node_2hop_mask, two_hop_mask = self.k_hop_batch_neighbor(dr_edge, pos_edge_index, data.num_users+data.num_items)
                sdf_node_1hop_mask, sdf_node_2hop_mask, two_hop_mask = sdf_node_1hop_mask.to(device), sdf_node_2hop_mask.to(device), two_hop_mask.to(device)
                
                z_df = model(x, pos_edge_index)
                z_ni = model(x, dr_edge[:, two_hop_mask])
                # print(two_hop_mask.sum())
                row, col = pos_edge_index
                nodes = torch.concat([row, col]).unique()
                # print(two_hop_mask.sum())
                row, col = dr_edge[:, two_hop_mask]
                ni_nodes = torch.concat([row, col]).unique()

                if args.dataset == "AmazonBook":
                    loss_l = 0
                    for start in range(0, data.num_users+data.num_items, args.batch_size):
                        end = start + args.batch_size

                        batch_z_df = z_df[start:end].cpu()
                        batch_z_ni = z_ni[start:end].cpu()
                        batch_z_ori = z_ori[start:end].cpu()
                        # print(batch_z_df.device, batch_z_ni.device)
                        batch_nodes = (nodes[ (nodes >= start) & (nodes < end)]-start).cpu()
                        batch_ni_nodes = (ni_nodes[ (ni_nodes >= start) & (ni_nodes < end)]-start).cpu()
                        # print(batch_ni_nodes.device)

                        loss_l += semi_loss(0.6, batch_z_df, batch_z_ni, batch_z_ori, batch_nodes, batch_ni_nodes)
                    del batch_z_df, batch_z_ni, batch_z_ori, batch_nodes, batch_ni_nodes
                    loss_l = loss_l.mean()    
                else:
                    loss_l = semi_loss(0.6, z_df, z_ni, z_ori, nodes, ni_nodes)


                neg_size = ind.numel()

                z = model(x, dr_edge)
                rank = model.decode(z, edge_label_index)
         
                loss_e = self.loss_fn(rank, indices, z)

                alpha = 0.8
                loss = alpha*loss_e + (1-alpha)* torch.abs(loss_l)

                loss.backward()
                optimizer.step()

                end_time = time.time()
                epoch_loss_e += loss_e.item()
                epoch_loss_l += loss_l.item()
                epoch_loss += loss.item()
                epoch_time += end_time - start_time

                # if epoch == 0 and step==0 :
                #     dr_edge = self.rewiring( z_df, dr_edge, data.num_nodes)


            if (epoch+1) % args.valid_freq == 0:

                score = nora(args, model, data)
                user_score, item_score = score[:data.num_users], score[data.num_users:]
                user_score = torch.log(torch.Tensor(user_score))
                item_score = torch.log(torch.Tensor(item_score))
                # score = [user_score, item_score]

                precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, valid_log= self.eval(args, model, data, group= popularity_bias, stage='val', score=item_score)
                # valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

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

                # if dt_auc + df_auc > best_metric:
                # if abs(recall-0.1547) + abs(0.7574-df_auc) < best_metric  :
                if abs(args.retrain_auc-df_auc) < best_metric :
                
                    best_recall = recall
                    best_epoch = epoch
                    best_precision = precision
                    best_ndcg = ndcg
                    best_metric = abs(args.retrain_auc-df_auc)
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. recall = {recall:.4f}. df_auc = {df_auc:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    # torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))