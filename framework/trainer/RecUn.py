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


class RecUnTrainer(Trainer):

    def train(self, model, data, optimizer, args, popularity_bias= None, z_ori=None, attack_model_all=None, attack_model_sub=None):
        
        return self.train_minibatch(model, data, optimizer, args, popularity_bias, z_ori, attack_model_all, attack_model_sub)

    def compute_loss(self, model, data, random_loss_fct, compute_random_on, random_layer, local_loss_fct, compute_local_on, local_layer, 
                     z1=None, z2=None, z1_ori=None, z2_ori=None, logits_ori=None, 
                     sdf1_all_pair_without_df_mask=None, sdf2_all_pair_without_df_mask=None):
        
        # Randomness
        loss_r = 0
        if random_layer == '1':
            all_z = [z1]
        elif random_layer == '2':
            all_z = [z2]
        elif random_layer == 'both':
            all_z = [z1, z2]
        else:
            raise NotImplementedError
        
        neg_size = data.df_mask.sum()
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=neg_size)

        if compute_random_on == 'edgeprob':       # Compute Randomness on edge probability
            
            for z in all_z:
                df_logits = model.decode(z, data.train_pos_edge_index[:, data.df_mask], neg_edge_index)
                loss_r += random_loss_fct(df_logits[:neg_size], df_logits[neg_size:])

        elif compute_random_on == 'nodeemb':
            for z in all_z:
                z_random_source, z_random_target = z[neg_edge_index[0]], z[neg_edge_index[1]]
                z_source, z_target = z[data.train_pos_edge_index[:, data.df_mask][0]], z[data.train_pos_edge_index[:, data.df_mask][1]]
                loss_r += (random_loss_fct(z_source, z_random_source) + random_loss_fct(z_target, z_random_target))

        elif compute_random_on == 'none':
            loss_r = None

        else:
            raise NotImplementedError


        # Local causality
        loss_l = 0
        if local_layer == '1':
            all_z = [z1]
            all_z_ori = [z1_ori]
            all_sdf_lower_triangular_mask = [sdf1_all_pair_without_df_mask]
            all_sdf_node_mask = [data.sdf_node_1hop_mask]
        elif local_layer == '2':
            all_z = [z2]
            all_z_ori = [z2_ori]
            all_sdf_lower_triangular_mask = [sdf2_all_pair_without_df_mask]
            all_sdf_node_mask = [data.sdf_node_2hop_mask]
        elif local_layer == 'both':
            all_z = [z1, z2]
            all_z_ori = [z1_ori, z2_ori]
            all_sdf_lower_triangular_mask = [sdf1_all_pair_without_df_mask, sdf2_all_pair_without_df_mask]
            all_sdf_node_mask = [data.sdf_node_1hop_mask, data.sdf_node_2hop_mask]
        else:
            raise NotImplementedError


        if compute_local_on == 'edgeprob':

            for z_ori, z, sdf_lower_triangular_mask in zip(all_z_ori, all_z, all_sdf_lower_triangular_mask):
                logits = (z @ z.t())[sdf_lower_triangular_mask].sigmoid()
                logits_ori = (z_ori @ z_ori.t())[sdf_lower_triangular_mask].sigmoid()

                loss_l += local_loss_fct(logits, logits_ori)
        
        elif compute_local_on == 'nodeemb':

            for z_ori, z, sdf_node_mask in zip(all_z_ori, all_z, all_sdf_node_mask):
                print(z_ori.shape, z.shape, sdf_node_mask.shape, sdf_node_mask.sum())
                loss_l += local_loss_fct(z_ori[sdf_node_mask], z[sdf_node_mask])

        elif compute_local_on == 'none':
            loss_l = None

        else:
            raise NotImplementedError


        if compute_random_on == 'none':
            loss = loss_l
        elif compute_local_on == 'none':
            loss = loss_r
        else:
            alpha = 0.5
            loss = alpha * loss_r + (1 - alpha) * loss_l

        return loss, loss_r, loss_l
    
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
    

    
    def train_minibatch(self, model, data, optimizer, args, popularity_bias= None, z_ori=None, attack_model_all=None, attack_model_sub=None):
        start_time = time.time()
        best_recall = best_precision = best_ndcg = best_metric = 0
        early_stop_cnt = 0    
        # model = model.to(device)
        data = data.to(device)
        best_metric = best_auc = best_recall =0

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

        idx = torch.randperm(dr_edge.size(1))[:df_edge.size(1)]
        sub_dr_edge = dr_edge[:,idx]

        for epoch in trange(args.epochs, desc='Unlerning'):

            model.train()

            loader = torch.utils.data.DataLoader(
                range(df_edge.size(1)),
                shuffle=True,
                batch_size=args.batch_size,
                num_workers=4)
            

            epoch_loss_e, epoch_loss_l, epoch_loss_f= 0, 0, 0
            epoch_loss, epoch_time = 0, 0
            for step, ind  in enumerate(tqdm(loader, leave=False)):
                start = time.time()
                optimizer.zero_grad()

                pos_edge_index = sub_dr_edge[:, ind].to(device)
                forget_edge_index = df_edge[:, ind].to(device)

                user_indices, pos_item_indices = pos_edge_index
                neg_item_indices = torch.randint(data.num_users, data.num_users + data.num_items,(ind.numel(), ), device=device)
                indices = [user_indices, pos_item_indices, neg_item_indices]
                neg_edge_index = torch.stack([user_indices, neg_item_indices], dim=0).long()
            
                edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

                sdf_node_1hop_mask, sdf_node_2hop_mask, two_hop_mask = self.k_hop_batch_neighbor(dr_edge, forget_edge_index, data.num_users+data.num_items)
                sdf_node_1hop_mask, sdf_node_2hop_mask, two_hop_mask = sdf_node_1hop_mask.to(device), sdf_node_2hop_mask.to(device), two_hop_mask.to(device)
                
                z_df = model(x, forget_edge_index)
                z_ni = model(x, dr_edge[:, two_hop_mask])

                row, col = forget_edge_index
                nodes = torch.concat([row, col]).unique()

                row, col = dr_edge[:, two_hop_mask]
                ni_nodes = torch.concat([row, col]).unique()

                if args.dataset == "AmazonBook":
                    loss_l = 0
                    cnt = 0
                    for start in range(0, data.num_users+data.num_items, args.batch_size*4):
                        end = start + args.batch_size*4

                        batch_z_df = z_df[start:end].cpu()
                        batch_z_ni = z_ni[start:end].cpu()
                        batch_z_ori = z_ori[start:end].cpu()
                        # print(batch_z_df.device, batch_z_ni.device)
                        batch_nodes = (nodes[ (nodes >= start) & (nodes < end)]-start).cpu()
                        batch_ni_nodes = (ni_nodes[ (ni_nodes >= start) & (ni_nodes < end)]-start).cpu()
                        # print(batch_ni_nodes.device)

                        loss_batch = semi_loss(0.6, batch_z_df, batch_z_ni, batch_z_ori, batch_nodes, batch_ni_nodes)
                        
                        loss_batch[torch.isnan(loss_batch)] = 0

                        loss_l += loss_batch
                        cnt += 1

                    del batch_z_df, batch_z_ni, batch_z_ori, batch_nodes, batch_ni_nodes
                    loss_l = loss_l/cnt
                else:
                    loss_l = semi_loss(0.6, z_df, z_ni, z_ori, nodes, ni_nodes)


                neg_size = ind.numel()

                z = model(x, dr_edge)
                rank = model.decode(z, edge_label_index)
         
                loss_e = self.loss_fn(rank, indices, z)

                if args.dataset == 'MovieLens1M' and args.dataset == 'Douban':
                    alpha = 0.8
                else:
                    alpha = 0.6
                
                if args.dataset == "Douban":
                    loss_l  = loss_l
                else:
                    loss_l = torch.abs(loss_l)
                    
                loss = alpha*loss_e + (1-alpha)* loss_l
                # if args.dataset == 'Douban':
                #      loss = alpha*loss_e + 0.4* loss_l

                loss.backward()
                optimizer.step()

                end_time = time.time()
                epoch_loss_e += loss_e.item()
                epoch_loss_l += loss_l.item()
                epoch_loss += loss.item()
                epoch_time += end_time - start

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
                    torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
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