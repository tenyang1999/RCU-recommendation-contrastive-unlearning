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
from sklearn.metrics import roc_auc_score, average_precision_score
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import json
from ..models import RecEraser, LightGCN
from ..metric import gap

class RecEraserTrainer():

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
        
        self.emb_dim = embedding_dim
        self.num_layers = num_layers

        self.num_users, self.num_items = args.num_users, args.num_items
        self.attention_size = int(self.emb_dim/2)
        self.args = args
        self.trainer_log = {
            'unlearning_model': args.unlearning_model, 
            'dataset': args.dataset, 
            'log': []}
        self.logit_all_pair = None
        self.df_pos_edge = []
        self.loss_fn = BPRLoss()

    
    def train_single_model(self, args, model, x, train_edge_index, data, shard_num):

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_recall = early_stop_cnt = 0

        train_edge_index = to_undirected(train_edge_index)
        
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
                neg_item_indices = torch.randint(self.num_users, self.num_users + self.num_items,(ind.numel(), ), device=device)
                indices = [user_indices, pos_item_indices, neg_item_indices]
                
                # user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(batch.size(0), edge_index)

                neg_edge_index = torch.stack([user_indices, neg_item_indices], dim=0).long()
            
                edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

                optimizer.zero_grad()
                
                # rank, emb = model(data['user'].x, data[item].x , train_edge_index, edge_label_index)
                emb= model(x, train_edge_index)
                rank = model.decode(emb, edge_label_index)
                
                loss = self.loss_fn(rank, indices, emb)
                    
                loss.backward()
                optimizer.step()

            if (epoch+1) % args.valid_freq == 0:
                precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, valid_log=  self.eval(args, model, None, data, stage='val')

                if best_recall < recall:
                    best_recall = recall
                    best_epoch = epoch
                    best_precision = precision
                    best_ndcg = ndcg

                    print(f'Save best checkpoint at epoch {epoch:04d}. recall = {recall:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, f'{shard_num}_model_best.pt'))
                    torch.save(emb, os.path.join(args.checkpoint_dir, f'{shard_num}_node_embeddings.pt'))
                    early_stop_cnt = 0
                else:
                    early_stop_cnt +=1

            if early_stop_cnt >= 10:
                print(f'shard: {shard_num} early stop training at epoch {epoch}')
                break

    def edge_exists(self, edge_index, edge_to_check):
        mask = (edge_index[0] == edge_to_check[0]) & (edge_index[1] == edge_to_check[1])
        return mask

    def train(self, model, data, args, popularity_bias, z_ori, shard, attack_model_all, attack_model_sub):
        
        return self.train_minibatch(model, data,  args, popularity_bias, z_ori, shard, attack_model_all, attack_model_sub)
        

    def train_minibatch(self, model, data,  args, popularity_bias, z_ori, shard, attack_model_all, attack_model_sub):
        start_time = time.time()
        best_recall = best_precision = best_ndcg = 0

        x = None

        df_edge = data.train_pos_edge_index[:, data.df_mask]
        dele = df_edge.T

        retrain_shard = []
        all_mask = []
        for num_shard in range(len(shard)):
            shard[num_shard] = shard[num_shard].to(device)
            shard_mask = torch.zeros(shard[num_shard].size(1), dtype=bool).to(device)
            for i in range(len(dele)):
                mask = self.edge_exists(shard[num_shard], dele[i])
                if mask.any().item():
                    shard_mask = shard_mask | mask
                    
            all_mask.append(shard_mask)
            if shard_mask.sum() != 0:
                retrain_shard.append(num_shard)

        for shard_num in retrain_shard:
            shard[shard_num] = shard[shard_num][:, ~all_mask[shard_num]]

            # model.load_state_dict(model_para[shard_num]['model_state'], strict=False)
            model = LightGCN(args, embedding_dim=64, num_layers=2).to(device)
            
            self.train_single_model(args, model, x, shard[shard_num], data, shard_num)

        self.trainer_log['shard_retrain_time'] = time.time() - start_time

        shard_emb = []

        for shard_num in range(len(shard)):
            if os.path.exists(os.path.join(args.checkpoint_dir, f'{shard_num}_model_best.pt')):
                model_para = torch.load(os.path.join(args.checkpoint_dir, f'{shard_num}_model_best.pt'))
            else:
                model_para = torch.load(os.path.join(args.original_path, f'{shard_num}_model_best.pt'))
            shard_emb.append(model_para['model_state']['emb.weight'])

        all_emb = torch.stack(shard_emb) 
        all_emb = all_emb.permute(1, 0, 2)


        model = RecEraser(args, embedding_dim=64, num_layers=2).to(device)

        early_stop_cnt = 0

        dr_edge = data.train_pos_edge_index[:, data.dr_mask]
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_recall = early_stop_cnt = 0
        
        start_time = time.time()
        for epoch in range(args.epoch_agg):

            model.train()

            loader = torch.utils.data.DataLoader(
                range(dr_edge.size(1)),
                shuffle=True,
                batch_size=args.batch_size,
            )

            epoch_loss = epoch_time = total_examples = 0
            for step, ind in enumerate(tqdm(loader, desc='Step',position=0, leave=True)):
                start = time.time()
                # Positive and negative sample

                pos_edge_index = dr_edge[:, ind].to(device)

                user_indices, pos_item_indices = pos_edge_index
                neg_item_indices = torch.randint(self.num_users, self.num_users + self.num_items,(ind.numel(), ), device=device)
                indices = [user_indices, pos_item_indices-self.num_users, neg_item_indices-self.num_users]
                
                # user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(batch.size(0), edge_index)

                # neg_edge_index = torch.stack([user_indices, neg_item_indices], dim=0).long()
            
                # edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

                optimizer.zero_grad()

                loss, users_emb, items_emb = model(all_emb, indices,)

                emb = torch.cat([users_emb, items_emb])

                
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
                precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, valid_log= self.eval(args, model, emb, data, stage='val')
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
    def eval(self, args, model, emb, data, group=None, stage='val', score=None, pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        if args.eval_on_cpu:
            model = model.to('cpu')
        
        if hasattr(data, 'dtrain_mask'):
            mask = data.dtrain_mask
        else:
            mask = data.dr_mask
        
        x = None
        if emb == None:
            emb = model(x, data.train_pos_edge_index[:, mask])
            users_emb, items_emb = emb[:args.num_users], emb[args.num_users:]
        else:
            users_emb, items_emb = emb[:args.num_users], emb[args.num_users:]

        dr_edge = data.train_pos_edge_index[:, mask].to(device)
        dr_edge = dr_edge[:, dr_edge[0] < args.num_users]

        precision = recall = ndcg = total_examples = 0

        if stage == 'test':
            rf = torch.zeros(data.num_items), torch.zeros(data.num_items), torch.zeros(data.num_items)
            group_precision = torch.Tensor([0, 0, 0]) # group precision
            group_recall = torch.Tensor([0, 0, 0]) # group recall
            group_ndcg = torch.Tensor([0, 0, 0])# group ndcg
            group_examples = torch.Tensor([0, 0, 0])# group examples
            group3 = group['best'], group['diverse'], group['niche']

        for start in range(0, args.num_users, args.batch_size):

            end = start + args.batch_size

            logits = users_emb[start:end] @ items_emb.t()

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
        if args.unlearning_model in ['original', 'RecEraser_Ori']:
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



        if args.eval_on_cpu:
            model = model.to(device)
        if stage == 'test':
            return precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, log, rf, topk_ind
        else:
            return precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, log

    # @torch.no_grad()
    def test(self, args, model, data, popularity_bias, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):
        
        if ckpt == 'best':    # Load best ckpt
            emb = torch.load(os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
            # ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'model_best.pt'))
            # model.load_state_dict(ckpt['model_state'])

        if  args.dataset == 'AmazonBook':
            pred_all = False
        else:
            pred_all = True
            
        # popularity_bias = torch.load(os.path.join(args.data_dir, args.dataset,'popularity_bias.pt'))
        # group = popularity_bias['best'], popularity_bias['diverse'], popularity_bias['niche']self.eval(args, model, emb, data, stage='val')

        precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, test_log, rf, topk_ind= self.eval(args, model, emb, data, popularity_bias, stage='test', score=None,  pred_all= False)
        
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


        return precision, recall, ndcg, df_auc, df_aup, df_logit, test_log

    # @torch.no_grad()
    # def get_output(self, model, node_embedding, data):
    #     model.eval()
    #     node_embedding = node_embedding.to(device)
    #     edge = data.edge_index.to(device)
    #     output = model.decode(node_embedding, edge, edge_type)

    #     return output

    def save_log(self, args):
        # print(self.trainer_log)
        with open(os.path.join(args.checkpoint_dir, 'trainer_log.json'), 'w') as f:
            json.dump(self.trainer_log, f)
        
        torch.save(self.logit_all_pair, os.path.join(args.checkpoint_dir, 'pred_proba.pt'))


