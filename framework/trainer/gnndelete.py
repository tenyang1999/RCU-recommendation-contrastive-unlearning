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


    
class GNNDeleteTrainer(Trainer):

    def train(self, model, data, optimizer, args, z_ori=None, attack_model_all=None, attack_model_sub=None):
        
        return self.train_minibatch(model, data, optimizer, args, z_ori, attack_model_all, attack_model_sub)

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

    def train_fullbatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to('cuda')
        data = data.to('cuda')

        best_metric = 0

        # '''Model naming convention: "gnndelete_random_mse_edgeprob_1_local_mse_edgeprob_1" '''
        # _, _, random_loss_fct, compute_random_on, random_layer, _, local_loss_fct, compute_local_on, local_layer = self.args.unlearning_model.split('_')
        # random_loss_fct = get_loss_fct(random_loss_fct)
        # local_loss_fct = get_loss_fct(local_loss_fct)

        # neg_size = 10

        # MI Attack before unlearning
        # if attack_model_all is not None:
        #     mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
        #     self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
        #     self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        # if attack_model_sub is not None:
        #     mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
        #     self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
        #     self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before

        # All node paris in S_Df without Df
        ## S_Df 1 hop all pair mask
        sdf1_all_pair_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        idx = torch.combinations(torch.arange(data.num_nodes)[data.sdf_node_1hop_mask], with_replacement=True).t()
        sdf1_all_pair_mask[idx[0], idx[1]] = True
        sdf1_all_pair_mask[idx[1], idx[0]] = True

        assert sdf1_all_pair_mask.sum().cpu() == data.sdf_node_1hop_mask.sum().cpu() * data.sdf_node_1hop_mask.sum().cpu()

        ## Remove Df itself
        sdf1_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1]] = False
        sdf1_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][1], data.train_pos_edge_index[:, data.df_mask][0]] = False

        ## S_Df 2 hop all pair mask
        sdf2_all_pair_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        idx = torch.combinations(torch.arange(data.num_nodes)[data.sdf_node_2hop_mask], with_replacement=True).t()
        sdf2_all_pair_mask[idx[0], idx[1]] = True
        sdf2_all_pair_mask[idx[1], idx[0]] = True

        assert sdf2_all_pair_mask.sum().cpu() == data.sdf_node_2hop_mask.sum().cpu() * data.sdf_node_2hop_mask.sum().cpu()

        ## Remove Df itself
        sdf2_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1]] = False
        sdf2_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][1], data.train_pos_edge_index[:, data.df_mask][0]] = False

        ## Lower triangular mask
        idx = torch.tril_indices(data.num_nodes, data.num_nodes, -1)
        lower_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        lower_mask[idx[0], idx[1]] = True

        ## The final mask is the intersection
        sdf1_all_pair_without_df_mask = sdf1_all_pair_mask & lower_mask
        sdf2_all_pair_without_df_mask = sdf2_all_pair_mask & lower_mask

        # print(data.sdf_node_2hop_mask.sum())
        # print(sdf_all_pair_mask.nonzero())
        # print(data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1])
        # print('aaaaaaaaaaaa', data.sdf_node_2hop_mask.sum(), a, sdf_all_pair_mask.sum())
        # print('aaaaaaaaaaaa', lower_mask.sum())
        # print('aaaaaaaaaaaa', sdf_all_pair_without_df_mask.sum())
        # print('aaaaaaaaaaaa', data.sdf_node_2hop_mask.sum())
        # assert sdf_all_pair_without_df_mask.sum() == \
        #         data.sdf_node_2hop_mask.sum().cpu() * (data.sdf_node_2hop_mask.sum().cpu() - 1) // 2 - data.df_mask.sum().cpu()

        # Original node embeddings
        # with torch.no_grad():
        #     z1_ori, z2_ori = model.get_original_embeddings(data.x, data.train_pos_edge_index[:, data.dtrain_mask], return_all_emb=True)
          
        # 新加的   
        df_all_pair_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        df_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1]] = True
        df_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][1], data.train_pos_edge_index[:, data.df_mask][0]] = True
        df_all_pair_mask_low = df_all_pair_mask & lower_mask
        
        loss_fct = nn.MSELoss()

        for epoch in trange(args.epochs, desc='Unlerning'):
            model.train()

            start_time = time.time()
            z = model(data.x, data.train_pos_edge_index[:, data.sdf_mask])
            # z1, z2 = model(data.x, data.train_pos_edge_index[:, data.sdf_mask], return_all_emb=True)
            # print('current deletion weight', model.deletion1.deletion_weight.sum(), model.deletion2.deletion_weight.sum())
            # print('aaaaaa', z[data.sdf_node_2hop_mask].sum())

            # Effectiveness and Randomness
            neg_size = data.df_mask.sum()
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=neg_size)

            df_logits = model.decode(z, data.train_pos_edge_index[:, data.df_mask], neg_edge_index)
            loss_r = loss_fct(df_logits[:neg_size], df_logits[neg_size:])
            # df_logits = model.decode(
            #     z, 
            #     data.train_pos_edge_index[:, data.df_mask].repeat(1, neg_size), 
            #     neg_edge_index).sigmoid()
            
            # loss_e = loss_fct(df_logits[:neg_size], df_logits[neg_size:])
            # print('df_logits', df_logits)
            # raise        
            
            # Local causality
            if sdf2_all_pair_without_df_mask.sum() != 0:
                logits_sdf = (z @ z.t())[sdf2_all_pair_without_df_mask].sigmoid()
                loss_l = loss_fct(logits_sdf, logits_ori[sdf2_all_pair_without_df_mask].sigmoid())
                # print('local proba', logits_sdf.shape, logits_sdf, logits_ori[sdf2_all_pair_without_df_mask].sigmoid())
                
                ############# 本次研究主要新增修改的地方#########
                # tau = 0.3
                # logits_df = (z @ z.t())[df_all_pair_mask].sigmoid()
                # del_logits =torch.exp(loss_fct(logits_df,logits_ori[df_all_pair_mask].sigmoid())/tau) 
                # ni_logits = torch.exp(loss_fct(logits_sdf, logits_ori[sdf2_all_pair_without_df_mask].sigmoid())/tau)
                # loss_l = -torch.log(del_logits/(del_logits + ni_logits))

                ############## new version of df low #########
                # tau = 0.3
                # logits_df = (z @ z.t())[df_all_pair_mask_low].sigmoid()
                # del_logits =torch.exp(loss_fct(logits_df,logits_ori[df_all_pair_mask_low].sigmoid())/tau) 
                # ni_logits = torch.exp(loss_fct(logits_sdf, logits_ori[sdf2_all_pair_without_df_mask].sigmoid())/tau)
                # loss_l = -torch.log(del_logits/(del_logits + ni_logits))

            
            else:
                loss_l = torch.tensor(0)
                print('local proba', 0)


            alpha = 0.5
            loss = alpha * loss_r + (1 - alpha) * loss_l
            #loss = loss_l
            # loss, loss_r, loss_l = self.compute_loss(
            #     model, data, random_loss_fct, compute_random_on, random_layer, local_loss_fct, compute_local_on, local_layer,
            #     z1, z2, z1_ori, z2_ori, logits_ori, sdf1_all_pair_without_df_mask, sdf2_all_pair_without_df_mask)
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            end_time = time.time()
            epoch_time = end_time - start_time

            step_log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
                'loss_r': loss_r.item(),
                'loss_l': loss_l.item(),
                'train_time': epoch_time
            }
            # wandb.log(step_log)
            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in step_log.items()]
            tqdm.write(' | '.join(msg))

            if (epoch + 1) % self.args.valid_freq == 0:
                precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')
                valid_log['epoch'] = epoch

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'train_loss_l': loss_l.item(),
                    'train_loss_r': loss_r.item(),
                    'train_time': epoch_time,
                }
                
                for log in [train_log, valid_log]:
                    # wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))
                    self.trainer_log['log'].append(log)

                if dt_auc + df_auc > best_metric:
                    best_metric = dt_auc + df_auc
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))


    def train_minibatch(self, model, data, optimizer, args, z_ori=None, attack_model_all=None, attack_model_sub=None):
        
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
        sdf_edge = data.train_pos_edge_index[:, data.sdf_mask].to(device)
        df_edge = data.train_pos_edge_index[:, data.df_mask].to(device)
        

        for epoch in trange(args.epochs, desc='Unlerning'):

            model.train()

            loader = torch.utils.data.DataLoader(
                range(df_edge.size(1)),
                shuffle=True,
                batch_size=args.batch_size,
            )

            # print('current deletion weight', model.deletion1.deletion_weight.sum(), model.deletion2.deletion_weight.sum())

            epoch_loss_e, epoch_loss_l = 0, 0
            epoch_loss, epoch_time = 0, 0
            for step, ind  in enumerate(tqdm(loader, leave=False)):
                start_time = time.time()
                optimizer.zero_grad()
                pos_edge_index = df_edge[:, ind].to(device)
                user_indices, pos_item_indices = pos_edge_index
                neg_item_indices = torch.randint(data.num_users, data.num_users + data.num_items,(ind.numel(), ), device=device)
                indices = [user_indices, pos_item_indices, neg_item_indices]
                neg_edge_index = torch.stack([user_indices, neg_item_indices], dim=0).long()
            
                edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

                sdf_node_1hop_mask, sdf_node_2hop_mask, two_hop_mask = self.k_hop_batch_neighbor(dr_edge, pos_edge_index, data.num_nodes)
                sdf_node_1hop_mask, sdf_node_2hop_mask, two_hop_mask = sdf_node_1hop_mask.to(device), sdf_node_2hop_mask.to(device), two_hop_mask.to(device)
                
                z = model(x, dr_edge[:, two_hop_mask], sdf_node_1hop_mask, sdf_node_2hop_mask)
                # z_two_hop = z[batch.sdf_node_2hop_mask]

                # # Effectiveness and Randomness
                neg_size = ind.numel()

                df_logits = model.decode(z, edge_label_index)
                loss_e = loss_fct(df_logits[:neg_size], df_logits[neg_size:])
                
                # Local causality
                # Only take the lower triangular part


                edge = dr_edge[:, two_hop_mask]
                lower_mask = edge[0] < edge[1]
                row, col = edge[0][lower_mask], edge[1][lower_mask]        
                
                all_ni = torch.cat([row, col])
                # ori = z_ori
                loss_l = (F.cosine_similarity(z_ori[all_ni], z[all_ni]).sum())*-1

                # logits_ni_ori = (z_ori[row] * z_ori[col]).sum(dim=-1).to(device)
                # logits_ni = (z[row] * z[col]).sum(dim=-1)
                # loss_l = loss_fct(logits_ni, logits_ni_ori)
                
                # print(loss_e, loss_l, z_ori.device, z.device)
                alpha = 0.5
                if 'ablation_random' in self.args.unlearning_model:
                    loss_l = torch.tensor(0)
                    loss = loss_e
                elif 'ablation_locality' in self.args.unlearning_model:
                    loss_e = torch.tensor(0)
                    loss = loss_l
                else:
                    loss = alpha * loss_e + (1 - alpha) * loss_l
                    # loss *= 10000
                    # loss = alpha * loss_e + alpha * loss_l

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

                end_time = time.time()
                epoch_loss_e += loss_e.item()
                epoch_loss_l += loss_l.item()
                epoch_loss += loss.item()
                epoch_time += end_time - start_time


            if (epoch+1) % args.valid_freq == 0:
                precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, valid_log= self.eval(args, model, data, stage='val')
                # valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': epoch_loss / step,
                    'train_loss_l': epoch_loss_l / step,
                    'train_loss_e': epoch_loss_e / step,
                    'train_time': epoch_time / step,
                }
                
                for log in [train_log, valid_log]:
                    # wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))
                    self.trainer_log['log'].append(log)

                # if dt_auc + df_auc > best_metric:
                if best_metric < recall+ ndcg:
                    best_recall = recall
                    best_epoch = epoch
                    best_precision = precision
                    best_ndcg = ndcg
                    best_metric = recall + ndcg
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. recall = {recall:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
            

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))
        return sdf_edge
