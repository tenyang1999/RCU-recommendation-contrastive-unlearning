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
from ..cgu_utils import *
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CGUTrainer(Trainer):

    def train(self, model, data, optimizer, args, popularity_bias= None, z_ori=None, attack_model_all=None, attack_model_sub=None):
        
        return self.train_minibatch(model, data, optimizer, args, popularity_bias= None, z_ori=None, attack_model_all=None, attack_model_sub=None)
        

    def train_minibatch(self, model, data, optimizer, args, popularity_bias= None, z_ori=None, attack_model_all=None, attack_model_sub=None):
        start_time = time.time()
        best_recall = best_precision = best_ndcg = 0

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
        # data = data.to(device)
        # dr_edge = data.train_pos_edge_index[:, data.dr_mask].to(device)
        w = torch.zeros(b.size()).float().to(device)
        w = ovr_lr_optimize(X_train, y_train, args.lam, weight, b=b, num_steps=args.num_steps, verbose=args.verbose,
                                opt_choice=args.optimizer, lr=args.lr, wd=args.wd)

        train_edge_index = data.train_pos_edge_index[:, data.dr_mask].to(device)
        early_stop_cnt = 0

        num_removes = train_edge_index.size(1)
        trails = args.epochs
        # budget for removal
        c_val = get_c(args.delta)
        b_std = args.std
        if args.compare_gnorm:
            budget = 1e5
        else:
            if args.train_mode == 'ovr':
                budget = get_budget(b_std, args.eps, c_val) * y_train.size(1)
            else:
                budget = get_budget(b_std, args.eps, c_val)
        gamma = 1/4  # pre-computed for -logsigmoid loss
        print('Budget:', budget)
        weight = None
        # our removal
        # num_removes - number of data points to remove
        # trails - Number of repeated trails 
        # grad_norm_approx is the data dependent upper bound of residual gradient norm
        grad_norm_approx = torch.zeros((num_removes, trails)).float()
        removal_times = torch.zeros((num_removes, trails)).float()  # record the time of each removal
        acc_removal = torch.zeros((2, num_removes, trails)).float()  # record the acc after removal, 0 for val, 1 for test
        grad_norm_worst = torch.zeros((num_removes, trails)).float()  # worst case norm bound
        grad_norm_real = torch.zeros((num_removes, trails)).float()  # true norm
        # graph retrain
        removal_times_graph_retrain = torch.zeros((args.num_removes, args.trails)).float()
        acc_graph_retrain = torch.zeros((2, args.num_removes, args.trails)).float()

        for trail_iter in trange(trails, desc='Epoch'):
            print('*'*10, trail_iter, '*'*10)
            model.train()
            
            # get a random permutation for edge indices for each trail
            perm = torch.from_numpy(np.random.permutation(data.train_pos_edge_index.shape[1]))
            # Note that all edges are used in training, so we just need to decide the order to remove edges
            # the number of training samples will always be m
            edge_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)

            X_scaled_copy = x.clone().detach().float()
            w_approx = w.clone().detach()  # copy the parameters to modify
            X_old = x.clone().detach().to(device)

            num_retrain = 0
            grad_norm_approx_sum = 0
            perm_idx = 0
            # start the removal process
            print('='*10 + 'Testing our edge removal' + '='*10)

            epoch_loss = epoch_time = total_examples = 0
                
            for i in range(args.num_removes):
                # First, check if this is a self-loop or an edge already deleted
                while (data.train_pos_edge_index[0, perm[perm_idx]] == data.train_pos_edge_index[1, perm[perm_idx]]) or (not edge_mask[perm[perm_idx]]):
                    perm_idx += 1
                edge_mask[perm[perm_idx]] = False
                source_idx = data.train_pos_edge_index[0, perm[perm_idx]]
                dst_idx = data.train_pos_edge_index[1, perm[perm_idx]]
                # find the other undirected edge
                rev_edge_idx = torch.logical_and(data.edge_index[0] == dst_idx,
                                                data.edge_index[1] == source_idx).nonzero().squeeze(-1)
                if rev_edge_idx.size(0) > 0:
                    edge_mask[rev_edge_idx] = False

                perm_idx += 1
                start = time.time()
                # Get propagated features
                if args.prop_step > 0:
                    X_new = model(X_scaled_copy, data.edge_index[:, edge_mask]).to(device)
                else:
                    X_new = X_scaled_copy.to(device)
                
                K = get_K_matrix(X_new[train_mask]).to(device)
                spec_norm = sqrt_spectral_norm(K)

                # removal from all one-vs-rest models
                X_rem = X_new[train_mask]
                for k in range(y_train.size(1)):
                    assert weight is None
                    y_rem = y_train[:, k]
                    H_inv = lr_hessian_inv(w_approx[:, k], X_rem, y_rem, args.lam)
                    # grad_i is the difference
                    grad_old = lr_grad(w_approx[:, k], X_old[train_mask], y_rem, args.lam)
                    grad_new = lr_grad(w_approx[:, k], X_rem, y_rem, args.lam)
                    grad_i = grad_old - grad_new
                    Delta = H_inv.mv(grad_i)
                    Delta_p = X_rem.mv(Delta)
                    # update w here. If beta exceed the budget, w_approx will be retrained
                    w_approx[:, k] += Delta
                    grad_norm_approx[i, trail_iter] += (Delta.norm() * Delta_p.norm() * spec_norm * gamma).cpu()
                    if args.compare_gnorm:
                        grad_norm_real[i, trail_iter] += lr_grad(w_approx[:, k], X_rem, y_rem, args.lam).norm().cpu()
                        grad_norm_worst[i, trail_iter] += get_worst_Gbound_edge(args.lam, X_rem.shape[0],
                                                                                args.prop_step)
                # decide after all classes
                if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                    # retrain the model
                    grad_norm_approx_sum = 0
                    b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
                    w_approx = ovr_lr_optimize(X_rem, y_train, args.lam, weight, b=b, num_steps=args.num_steps, verbose=args.verbose,
                                            opt_choice=args.optimizer, lr=args.lr, wd=args.wd)
                    num_retrain += 1
                else:
                    grad_norm_approx_sum += grad_norm_approx[i, trail_iter]

                removal_times[i, trail_iter] = time.time() - start
                # Remember to replace X_old with X_new
                X_old = X_new.clone().detach()
                if i % args.disp == 0:
                    print('Iteration %d: time = %.2fs, number of retrain = %d' % (i+1, removal_times[i, trail_iter], num_retrain))
                    print('Val acc = %.4f, Test acc = %.4f' % (acc_removal[0, i, trail_iter], acc_removal[1, i, trail_iter]))

    
        # # save all results
        # if not osp.exists(args.result_dir):
        #     os.makedirs(args.result_dir)
        # save_path = '%s/%s_std_%.0e_lam_%.0e_nr_%d_K_%d_opt_%s_mode_%s_eps_%.1f_delta_%.0e' % (args.result_dir,
        #                                                                                     args.dataset, b_std,
        #                                                                                     args.lam, args.num_removes,
        #                                                                                     args.prop_step,
        #                                                                                     args.optimizer,
        #                                                                                     args.removal_mode,
        #                                                                                     args.eps, args.delta)
        if args.train_mode == 'binary':
            save_path += '_bin_%s' % args.Y_binary
        if args.GPR:
            save_path += '_gpr'
        if args.compare_gnorm:
            save_path += '_gnorm'
        if args.compare_retrain:
            save_path += '_retrain'
        save_path += '.pth'

        torch.save({'grad_norm_approx': grad_norm_approx, 'removal_times': removal_times, 'acc_removal': acc_removal,
                    'grad_norm_worst': grad_norm_worst, 'grad_norm_real': grad_norm_real,
                    'removal_times_graph_retrain': removal_times_graph_retrain,
                    'acc_graph_retrain': acc_graph_retrain}, save_path)
            
                log = {
                    'epoch': trail_iter,
                    'step': step,
                    'train_loss': loss.item(),
                }
                # wandb.log(log)
                # msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                # tqdm.write(' | '.join(msg))

                epoch_loss += float(loss.item()) * (rank.numel()/2)
                total_examples += (rank.numel()/2)
                epoch_time += time.time() - start

            train_loss = epoch_loss/ total_examples

            if (trail_iter+1) % args.valid_freq == 0:
                precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, valid_log= self.eval(args, model, data, stage='val')
                # precision, recall, ndcg = self.test(args, model, data, k=20)
                # valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')
                train_log = {
                    'epoch': trail_iter,
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
                    best_epoch = trail_iter
                    best_precision = precision
                    best_ndcg = ndcg
                    best_time =  time.time()-start_time

                    print(f'Save best checkpoint at epoch {trail_iter:04d}. recall = {recall:.4f}')
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
                print(f'early stop training at epoch {trail_iter}')
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

        return train_edge_index

