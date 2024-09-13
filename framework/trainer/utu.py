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


    
class UtUTrainer(Trainer):

    def train(self, model, data, optimizer, args, popularity_bias= None, z_ori=None, attack_model_all=None, attack_model_sub=None):
        
        return self.train_minibatch(model, data, optimizer, args, popularity_bias, z_ori, attack_model_all, attack_model_sub)
    
    def train_minibatch(self, model, data, optimizer, args, popularity_bias= None, z_ori=None, attack_model_all=None, attack_model_sub=None):
        start_time = time.time()
        best_recall = best_precision = best_ndcg = best_metric = 0
        if 'kld' in args.unlearning_model:
            loss_fct = BoundedKLD
        else:
            loss_fct = nn.MSELoss()

        # model = model.to(device)
        data = data.to(device)
        best_metric = 0

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
        dr_edge = data.train_pos_edge_index[:, data.dr_mask].to(device)
        
        z = model(x, dr_edge)
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
        torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
        self.trainer_log['best_metric'] = best_metric

        self.trainer_log['best_training_time'] = time.time()-start_time

        # for epoch in trange(args.epochs, desc='Unlerning'):

        #     model.train()

        #     loader = torch.utils.data.DataLoader(
        #         range(dr_edge.size(1)),
        #         shuffle=True,
        #         batch_size=args.batch_size,
        #     )

        #     # print('current deletion weight', model.deletion1.deletion_weight.sum(), model.deletion2.deletion_weight.sum())

        #     epoch_loss_e, epoch_loss_l = 0, 0
        #     epoch_loss, epoch_time = 0, 0
        #     for step, ind  in enumerate(tqdm(loader, leave=False)):
        #         start_time = time.time()
        #         optimizer.zero_grad()
        #         pos_edge_index = dr_edge[:, ind].to(device)
        #         user_indices, pos_item_indices = pos_edge_index
        #         neg_item_indices = torch.randint(data.num_users, data.num_users + data.num_items,(ind.numel(), ), device=device)
        #         indices = [user_indices, pos_item_indices, neg_item_indices]
        #         neg_edge_index = torch.stack([user_indices, neg_item_indices], dim=0).long()
            
        #         edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

        #         sdf_node_1hop_mask, sdf_node_2hop_mask, two_hop_mask = self.k_hop_batch_neighbor(dr_edge, pos_edge_index, data.num_nodes)
        #         sdf_node_1hop_mask, sdf_node_2hop_mask, two_hop_mask = sdf_node_1hop_mask.to(device), sdf_node_2hop_mask.to(device), two_hop_mask.to(device)
                
        #         z = model(x, dr_edge[:, two_hop_mask], sdf_node_1hop_mask, sdf_node_2hop_mask)
        #         # z_two_hop = z[batch.sdf_node_2hop_mask]

        #         # # Effectiveness and Randomness
        #         neg_size = ind.numel()

        #         df_logits = model.decode(z, edge_label_index).sigmoid()
        #         # print(df_logits[:100])
        #         # print(df_logits[neg_size:neg_size+100])
                
        #         # loss_e = self.loss_fn(df_logits, indices, z)
        #         # mse loss want as small as possible
        #         loss_e = loss_fct(df_logits[:neg_size], df_logits[neg_size:])
                
        #         # Local causality
        #         # Only take the lower triangular part

        #         # edge = batch.edge_index[:, batch.sdf_mask]
        #         # lower_mask = edge[0] < edge[1]
        #         # row, col = edge[0][lower_mask], edge[1][lower_mask]
        #         row, col = dr_edge[:, two_hop_mask]
        #         logits_ni_ori = (z_ori[row] * z_ori[col]).sum(dim=-1).sigmoid().to(device)
        #         logits_ni = (z[row] * z[col]).sum(dim=-1).sigmoid()
        #         # loss_l = loss_fct(logits_ni, logits_ni_ori)
                
        #         row, col = data.train_pos_edge_index[:, data.df_mask]
        #         logits_df_ori = (z_ori[row] * z_ori[col]).sum(dim=-1).sigmoid().to(device)        
        #         logits_df = (z[row] * z[col]).sum(dim=-1).sigmoid()

        #         tau = 0.3
        #         del_logits =torch.exp(loss_fct(logits_df, logits_df_ori)/tau) 
        #         ni_logits = torch.exp(loss_fct(logits_ni, logits_ni_ori)/tau)
        #         loss_l = -torch.log(del_logits/(del_logits + ni_logits))
                

        #         # print(loss_e, loss_l, z_ori.device, z.device)
        #         alpha = 0
        #         if 'ablation_random' in self.args.unlearning_model:
        #             loss_l = torch.tensor(0)
        #             loss = loss_e
        #         elif 'ablation_locality' in self.args.unlearning_model:
        #             loss_e = torch.tensor(0)
        #             loss = loss_l
        #         else:
        #             loss = alpha * loss_e + (1 - alpha) * loss_l
        #             # loss = alpha * loss_e + alpha * loss_l

        #         loss.backward()
        #         # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        #         optimizer.step()

        #         end_time = time.time()
        #         epoch_loss_e += loss_e.item()
        #         epoch_loss_l += loss_l.item()
        #         epoch_loss += loss.item()
        #         epoch_time += end_time - start_time

        #     # epoch_loss_e /= step
        #     # epoch_loss_l /= step
        #     # epoch_loss /= step
        #     # epoch_time /= step

        #     if (epoch+1) % args.valid_freq == 0:
        #         precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, valid_log= self.eval(args, model, data, rw_edge = dr_edge, stage='val')
        #         # valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

        #         train_log = {
        #             'epoch': epoch,
        #             'train_loss': epoch_loss / step,
        #             'train_loss_l': epoch_loss_l / step,
        #             'train_loss_e': epoch_loss_e / step,
        #             'train_time': epoch_time / step,
        #         }
                
        #         for log in [train_log, valid_log]:
        #             # wandb.log(log)
        #             msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
        #             tqdm.write(' | '.join(msg))
        #             self.trainer_log['log'].append(log)

        #         # if dt_auc + df_auc > best_metric:
        #         if best_metric < recall+ ndcg:
        #             best_recall = recall
        #             best_epoch = epoch
        #             best_precision = precision
        #             best_ndcg = ndcg
        #             best_metric = recall + ndcg
        #             best_epoch = epoch

        #             print(f'Save best checkpoint at epoch {epoch:04d}. recall = {recall:.4f}')
                    # ckpt = {
                    #     'model_state': model.state_dict(),
                    #     'optimizer_state': optimizer.state_dict(),
                    # }
                    # torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    # torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
            

        # # Save
        # ckpt = {
        #     'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
        #     'optimizer_state': optimizer.state_dict(),
        # }
        # torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))


