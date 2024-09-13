import os
import time
import json
# import wandb
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
# from ogb.graphproppred import Evaluator
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling, degree, to_undirected
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

# from ..evaluation import *
from ..training_args import parse_args
# from ..utils import *
from ..utils import BPRLoss, plot_result, nora, nora_ultra
from ..metric import gap
from ..simGCL_utils import simGCL_cl_loss
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = 'cpu'

class Trainer:
    def __init__(self, args, with_fea = False):
        self.args = args
        self.trainer_log = {
            'unlearning_model': args.unlearning_model, 
            'dataset': args.dataset, 
            'log': []}
        self.logit_all_pair = None
        self.df_pos_edge = []
        self.loss_fn = BPRLoss()
        self.with_fea = with_fea

        with open(os.path.join(self.args.checkpoint_dir, 'training_args.json'), 'w') as f:
            json.dump(vars(args), f)

    def freeze_unused_weights(self, model, mask):
        grad_mask = torch.zeros_like(mask)
        grad_mask[mask] = 1

        model.deletion1.deletion_weight.register_hook(lambda grad: grad.mul_(grad_mask))
        model.deletion2.deletion_weight.register_hook(lambda grad: grad.mul_(grad_mask))

    def train(self, model, data, optimizer, args):
        if self.args.dataset in ['to_be_continue']:
            pass
            # return self.train_fullbatch(model, data, optimizer, args)

        if self.args.dataset in ['MovieLens100K', 'Douban', 'AmazonBook', 'MovieLens1M','subDouban', 'subAmazonBook']:
            return self.train_minibatch(model, data, optimizer, args)
        
    def get_embedding(self, model, x, data, on_cpu=False):
        original_device = next(model.parameters()).device

        if on_cpu:
            model = model.cpu()
            data = data.cpu()
        
        z = model(x, data.train_pos_edge_index)

        model = model.to(original_device)

        return z

    def train_minibatch(self, args, model, data, optimizer):
        start_time = time.time()
        best_recall = best_precision = best_ndcg = 0

        # if self.with_fea:
        #     dim_users, dim_items = data['user'].x.size(1), data['item'].x.size(1)
        #     users_lin = torch.rand(dim_users, args.out_dim).to(device)
        #     items_lin = torch.rand(dim_items, args.out_dim).to(device)
        #     user_emb = torch.mm(data['user'].x, users_lin)
        #     item_emb = torch.mm(data['item'].x, items_lin)
        #     X = torch.cat([user_emb, item_emb])
        #     x = torch.tensor(X, requires_grad= True)
        # else:
        #     x = None
        x = None
        train_edge_index = data.train_pos_edge_index
        early_stop_cnt = 0

        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

            loader = torch.utils.data.DataLoader(
                range(train_edge_index.size(1)),
                shuffle=True,
                batch_size=args.batch_size,
            )
            epoch_loss = epoch_time = total_examples = 0
            for step, ind in enumerate(tqdm(loader, desc='Step',position=0, leave=True)):
                start = time.time()
                # Positive and negative sample

                pos_edge_index = train_edge_index[:, ind].to(device)
                user_indices, pos_item_indices = pos_edge_index

                if args.gnn =='UltraGCN':

                    neg_candidates = np.arange( data.num_items)
                    neg_items = np.random.choice(neg_candidates, (ind.size(0), args.negative_num), replace = True)
                    neg_items = torch.from_numpy(neg_items).to(device)
                    model.zero_grad()
                    loss = model(user_indices, pos_item_indices-data.num_users, neg_items)
                else:
                    
                    neg_item_indices = torch.randint(data.num_users, data.num_users + data.num_items,(ind.numel(), ), device=device)
                    indices = [user_indices, pos_item_indices, neg_item_indices]
                    
                    # user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(batch.size(0), edge_index)

                    neg_edge_index = torch.stack([user_indices, neg_item_indices], dim=0).long()
                
                    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

                    optimizer.zero_grad()
                    emb= model(x, train_edge_index)
                    rank = model.decode(emb, edge_label_index)
                    
                    loss = self.loss_fn(rank, indices, emb)

                    if args.gnn =='simGCL':
                        perturbed_emb1 = model(x, train_edge_index, perturbed = True)
                        perturbed_emb2 = model(x, train_edge_index, perturbed = True)

                        loss_c = simGCL_cl_loss(indices, perturbed_emb1, perturbed_emb2)

                        loss += args.cl_rate*loss_c
                    
                loss.backward()
                optimizer.step()

                log = {
                    'epoch': epoch,
                    'step': step,
                    'train_loss': loss.item(),
                }
                # wandb.log(log)
                # msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                # tqdm.write(' | '.join(msg))

                epoch_loss += float(loss.item()) * (ind.numel()/2)
                total_examples += (ind.numel()/2)
                epoch_time += time.time() - start

            train_loss = epoch_loss/ total_examples

            if (epoch+1) % args.valid_freq == 0:
                precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, valid_log= self.eval(args, model, data, stage='val')
                # precision, recall, ndcg = self.test(args, model, data, k=20)
                # valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')
                train_log = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'epoch_time': epoch_time
                }
                
                for log in [train_log, valid_log]:
                    # wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(train_log)
                self.trainer_log['log'].append(valid_log)

                if best_recall < recall:
                    best_recall = recall
                    best_epoch = epoch
                    best_precision = precision
                    best_ndcg = ndcg
                    best_time =  time.time()-start_time

                    print(f'Save best checkpoint at epoch {epoch:04d}. recall = {recall:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    if args.gnn =='UltraGCN':
                        user_embeds = model.user_embeds.weight 
                        item_embeds = model.item_embeds.weight
                        emb = torch.cat([user_embeds, item_embeds])

                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    torch.save(emb, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
                    early_stop_cnt = 0
                else:
                    early_stop_cnt +=1

            if early_stop_cnt >= 10:
                print(f'early stop training at epoch {epoch}')
                break

        # self.trainer_log['training_time'] = time.time() - start_time
        self.trainer_log['best_training_time'] = best_time

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d},\
        best_precison = {best_precision}, best_recall = {best_recall}, best_ndcg = {best_ndcg} ')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_precison'] = best_precision
        self.trainer_log['best_recall'] = best_recall
        self.trainer_log['best_ndcg'] = best_ndcg
        
        self.trainer_log['training_time'] = np.mean([i['epoch_time'] for i in self.trainer_log['log'] if 'epoch_time' in i])


    @torch.no_grad()
    def eval(self, args, model, data, group=None, stage='val', score=None, pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']
        
        if 'item' in args.df:
            df_item_mask = torch.Tensor([ True if i in data.df_node else False for i in pos_edge_index[1]-data.num_users]).bool()
            pos_edge_index[:, ~df_item_mask]

        if self.args.eval_on_cpu:
            model = model.to('cpu')
        
        if hasattr(data, 'dtrain_mask'):
            mask = data.dtrain_mask
        else:
            mask = data.dr_mask
        
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
        if args.gnn =='UltraGCN':
            users_emb, items_emb = model.user_embeds.weight, model.item_embeds.weight
            emb = torch.cat([users_emb, items_emb])
        else:
            emb = model(x, data.train_pos_edge_index[:, mask])
            users_emb, items_emb = emb[:args.num_users], emb[args.num_users:]

        dr_edge = data.train_pos_edge_index[:, mask].to(device)
        dr_edge = dr_edge[:, dr_edge[0] < args.num_users]

        precision = recall = ndcg = total_examples = 0

        if args.unlearning_model == 'RecUn' or args.unlearning_model == 'RecUn_ultra':

            score[score.isnan() | score.isinf() ] =0
            # print(score)
            best, diverse, niche, = group['best'], group['diverse'], group['niche']

            
            if args.dataset == 'MovieLens1M' :
                i1, i2, i3 = [0.4, 1, 0.4]
            elif args.dataset == 'Douban' or args.dataset == 'subDouban':
                i1, i2, i3 = [0.02, 0, 0.02]
            # elif args.dataset == 'subDouban':
            #     i1, i2, i3 = [0.08, 0.04, 0.04]
            elif args.dataset == 'AmazonBook' or args.dataset == 'subAmazonBook':
                i1, i2, i3 = [0.08, 0.1, 0.08]
            elif args.dataset == 'MovieLens100K' :
                i1, i2, i3 = [0.4, 0.8, 0.4]
            
            if  args.gnn == "UltraGCN":
                i1, i2, i3 = [-0.05, -0.05, -0.05]
                # i1, i2, i3 = [0, 0, 0]
            elif args.gnn == "gcn":
                # i1, i2, i3 = args.i1, args.i2, args.i3
                i1, i2, i3 = [0.2, 0.4, 0.4]
                # i1, i2, i3 = [0, 0, 0]

            i1, i2, i3 = [0, 0, 0]
            # print(i1, i2, i3)
            

            group_user = torch.zeros(data.num_users)
            group_user[best] = i1
            group_user[diverse] = i2
            group_user[niche] = i3


        if stage == 'test':
            rf = torch.zeros(data.num_items), torch.zeros(data.num_items), torch.zeros(data.num_items)
            group_precision = torch.Tensor([0, 0, 0]) # group precision
            group_recall = torch.Tensor([0, 0, 0]) # group recall
            group_ndcg = torch.Tensor([0, 0, 0])# group ndcg
            group_examples = torch.Tensor([0, 0, 0])# group examples
            group3 = group['best'], group['diverse'], group['niche']

        if args.dataset == 'AmazonBook' :
            batch_size = args.batch_size//4
        else:
            batch_size = args.batch_size

        for start in range(0, args.num_users, batch_size):

            end = start + batch_size

            logits = users_emb[start:end] @ items_emb.t()

            if args.unlearning_model == 'RecUn' or args.unlearning_model == 'RecUn_ultra' :
                batch_group_user = group_user[start:end]
                score, batch_group_user = score.to(device), batch_group_user.to(device)
                rerank_score = (logits*score)*batch_group_user.reshape(-1,1)
                logits = logits + rerank_score

            # Exclude training edges:
            train_mask = ((dr_edge[0] >= start) & (dr_edge[0] < end))
            logits[dr_edge[0, train_mask] - start,
                    dr_edge[1, train_mask] - args.num_users] = float('-inf')
            
            # Computing precision and recall:
            ground_truth = torch.zeros_like(logits, dtype=torch.bool)
            val_mask = ((pos_edge_index[0] >= start) & (pos_edge_index[0] < end))

            # ground_truth[test_edge_index[0, mask] - start,
            #                 test_edge_index[1, mask] - num_users] = True

            ground_truth[pos_edge_index[0, val_mask] - start,
                            pos_edge_index[1, val_mask]- args.num_users] = True
            
            node_count = degree(pos_edge_index[0, val_mask] - start,
                            num_nodes=logits.size(0)).int().to(device)
            
            if args.df_node:
                if 'user' in args.df:
                    dr_user = torch.ones(logits.size(0), dtype=torch.bool)
                    dr_user[data.df_node[(start < data.df_node) & (data.df_node < end )]-start] = False
                    logits = logits[dr_user]
                    ground_truth = ground_truth[dr_user]
                    node_count = node_count[dr_user]
                else:
                    dr_item = torch.ones(logits.size(1), dtype=torch.bool)
                    dr_item[data.df_node]= False
                    logits = logits[:, dr_item]
                    ground_truth = ground_truth[:, dr_item]
                    # node_count = node_count[:, dr_item]
            
            topk_val, topk_ind = logits.topk(args.topk, dim=-1)

            isin_mat = ground_truth.gather(1, topk_ind)

            precision += float((isin_mat.sum(dim=-1) / args.topk).sum())
            recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
            
            # count NDCG 
            gt_topk = torch.zeros(logits.size(0), args.topk)
            for i, items in enumerate(node_count):
                    length = args.topk if args.topk <= items.item() else items.item()
                    gt_topk[i, :length] = 1
            gt_topk = gt_topk.to(device)
            # print(isin_mat.int(), gt_topk)
            multiplier = (1.0 / torch.arange(2, args.topk + 2).log2()).to(device)

            dcg = ((2**isin_mat.int()-1)*multiplier).sum(dim=-1)
            idcg = ((2**gt_topk-1)*multiplier).sum(dim=-1)


            idcg[idcg == 0.] = 1.
            # idcg = (torch.sort(topk_val, dim=1, descending=True).values*multiplier).sum(dim=-1)
            out = dcg/idcg
            out[out.isnan() | out.isinf()] = 0.0
            out = out.sum().item()
            # print(out/logits.size(0))
            ndcg += out 
            total_examples += int((node_count > 0).sum())

            # count recommendation frequence
            # if stage == 'test':
            #     topk_ind = topk_ind.cpu()
            #     for id, (g, r) in enumerate(zip(group3, rf)): #, group_precision, group_recall, group_ndcg, group_examples):

            #         g = g[(g >= start) & (g < end)]
            #         g -= start
            #         ind, cnt = topk_ind[g].unique(return_counts=True)
            #         for i in range(ind.size(0)):
            #             r[ind[i]] += cnt[i]

            #         group_precision[id] += float((isin_mat[g].sum(dim=-1) / args.topk).sum())
            #         group_recall[id] += float((isin_mat[g].sum(dim=-1) / node_count[g].clamp(1e-6)).sum())

            #         dcg = ((2**isin_mat[g].int()-1)*multiplier).sum(dim=-1)
            #         idcg = ((2**gt_topk[g]-1)*multiplier).sum(dim=-1)

            #         idcg[idcg == 0.] = 1.
            #         # idcg = (torch.sort(topk_val, dim=1, descending=True).values*multiplier).sum(dim=-1)
            #         out = dcg/idcg
            #         out[out.isnan() | out.isinf()] = 0.0
            #         out = out.sum().item()
            #         # print(out/logits.size(0))
            #         group_ndcg[id] += out 
            #         group_examples[id] += int((node_count[g] > 0).sum())

        precision = round(precision / total_examples, 4)
        recall = round(recall / total_examples, 4)
        ndcg = round(ndcg / total_examples, 4) 

        if stage == 'test':
            group_precision = torch.round(group_precision/group_examples, decimals= 4)
            group_recall = torch.round(group_recall/group_examples, decimals= 4)
            group_ndcg = torch.round(group_ndcg/group_examples, decimals= 4)
        else:
            group_precision = np.nan
            group_recall = np.nan
            group_ndcg = np.nan

        # DF AUC AUP
        if self.args.unlearning_model in ['original', 'RecEraser_Ori']:
            df_logit = []
        else:
            # df_logit = model.decode(z, data.train_pos_edge_index[:, data.df_mask]).sigmoid().tolist()
            
            # 應該僅取某幾個user 或 item作為目標
            out_src = emb[data.directed_df_edge_index[0]]
            out_dst = emb[data.directed_df_edge_index[1]]
            df_logit = (out_src * out_dst).sum(dim=-1)
            # df_logit = model.decode(z, data.directed_df_edge_index).sigmoid().tolist()

        if len(df_logit) > 0:
            df_auc = []
            df_aup = []
        
            # Sample pos samples
            if len(self.df_pos_edge) == 0:
                for i in range(500):
                    mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
                    idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
                    mask[idx] = True
                    self.df_pos_edge.append(mask)
            
            # Use cached pos samples
            for mask in self.df_pos_edge:
                # pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()
                out_src = emb[data.train_pos_edge_index[:, data.dr_mask][:, mask][0]]
                out_dst = emb[data.train_pos_edge_index[:, data.dr_mask][:, mask][1]]
                pos_logit = (out_src * out_dst).sum(dim=-1)
                logit = torch.cat([df_logit, pos_logit]).cpu()
                label = [0] * len(df_logit) +  [1] * len(df_logit)
                # loss = F.binary_cross_entropy_with_logits(label, logit).cpu().item()
                df_auc.append(roc_auc_score(label, logit))
                df_aup.append(average_precision_score(label, logit))
        
            df_auc = round(np.mean(df_auc), 4)
            df_aup = round(np.mean(df_aup), 4)

        else:
            df_auc = np.nan
            df_aup = np.nan

        # Logits for all node pairs
        if pred_all:
            logit_all_pair = (users_emb @ items_emb.t()).cpu()
        else:
            logit_all_pair = None

        log = {
            # f'{stage}_loss': loss,
            f'{stage}_precision@{args.topk}': precision,
            f'{stage}_recall@{args.topk}': recall,
            f'{stage}_nDCG@{args.topk}': ndcg,
            # f'{stage}_df_loss': round(loss, 4),
            f'{stage}_df_auc': df_auc,
            f'{stage}_df_aup': df_aup,
            f'{stage}_group_precision': group_precision,
            f'{stage}_group_recall': group_recall,
            f'{stage}_group_nDCG': group_ndcg,
        }



        if self.args.eval_on_cpu:
            model = model.to(device)
        if stage == 'test':
            return precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, log, rf, topk_ind
        else:
            return precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, log

    # @torch.no_grad()
    def test(self, args, model, data, popularity_bias, attack_model_all=None, attack_model_sub=None, ckpt='best'):
        
        if ckpt == 'best':    # Load best ckpt
            ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'model_best.pt'))
            model.load_state_dict(ckpt['model_state'])

        if  self.args.dataset == 'AmazonBook':
            pred_all = False
        else:
            pred_all = True
            
        if args.unlearning_model == 'RecUn' :
            score = nora(args, model, data)
            item_score = score[data.num_users:]
            score = torch.log(torch.Tensor(item_score))
        elif args.unlearning_model == 'RecUn_ultra':
            score = nora_ultra(args, model, data)
            item_score = score[data.num_users:]
            score = torch.log(torch.Tensor(item_score))
        else:
            score = np.nan
        
        # popularity_bias = torch.load(os.path.join(args.data_dir, args.dataset,'popularity_bias.pt'))
        # group = popularity_bias['best'], popularity_bias['diverse'], popularity_bias['niche']

        precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, test_log, rf, topk_ind= self.eval(args, model, data, popularity_bias, stage='test', score=score,  pred_all= pred_all)
        
        self.trainer_log['test_precison'] = precision
        self.trainer_log['test_recall'] = recall
        self.trainer_log['test_ndcg'] = ndcg
        self.trainer_log['df_auc'] = df_auc
        self.trainer_log['df_aup'] = df_aup
        self.logit_all_pair = logit_all_pair



        # plot_result(args, popularity_bias, rf)
        

        # delta_gap_ALL, delta_gap_best, delta_gap_diverse, delta_gap_niche = gap(data.num_users, popularity_bias, topk_ind) 
        
        # self.trainer_log['delta_gap_ALL'] = delta_gap_ALL
        # self.trainer_log['delta_gap_best'] = delta_gap_best
        # self.trainer_log['delta_gap_diverse'] = delta_gap_diverse
        # self.trainer_log['delta_gap_niche'] = delta_gap_niche
        # delta_gap_ALL, delta_gap_best, delta_gap_diverse, delta_gap_niche

        return precision, recall, ndcg, df_auc, df_aup, df_logit, test_log

    # @torch.no_grad()
    # def get_output(self, model, node_embedding, data):
    #     model.eval()
    #     node_embedding = node_embedding.to(device)
    #     edge = data.edge_index.to(device)
    #     output = model.decode(node_embedding, edge, edge_type)

    #     return output

    def save_log(self):
        # print(self.trainer_log)
        with open(os.path.join(self.args.checkpoint_dir, 'trainer_log.json'), 'w') as f:
            json.dump(self.trainer_log, f)
        
        torch.save(self.logit_all_pair, os.path.join(self.args.checkpoint_dir, 'pred_proba.pt'))


