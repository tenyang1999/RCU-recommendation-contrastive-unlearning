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

class RecEraser_Ori(Trainer):
    def train(self, model, data, shard_num, optimizer, args):
        if self.args.dataset in ['to_be_continue']:
            pass
            # return self.train_fullbatch(model, data, optimizer, args)

        if self.args.dataset in ['MovieLens100K', 'Douban', 'AmazonBook', 'MovieLens1M']:
            return self.train_minibatch(model, data,  shard_num, optimizer, args)
        

    def train_minibatch(self, args, model, data, shard_num, optimizer):
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
                neg_item_indices = torch.randint(data.num_users, data.num_users + data.num_items,(ind.numel(), ), device=device)
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

                log = {
                    'epoch': epoch,
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
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, f'{shard_num}_model_best.pt'))
                    torch.save(emb, os.path.join(args.checkpoint_dir, f'{shard_num}_node_embeddings.pt'))
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

