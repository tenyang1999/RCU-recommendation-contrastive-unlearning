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
from torch_geometric.data import Data

class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
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


    
class UserTrainer:

    def __init__(self, args):
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()


    def train(self, model, data, optimizer, args, pb):

        if self.args.dataset in ['to_be_continue']:
            pass
            # return self.train_fullbatch(model, data, optimizer, args)

        if self.args.dataset in ['MovieLens100K', 'Douban', 'AmazonBook', 'MovieLens1M']:
            return self.train_minibatch(model, data, optimizer, args, pb)

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

    def train_minibatch(self, args, model, data, optimizer, label):

        train_data = data['user'].x
        model = FullyConnectedNet(train_data.size(1), 32, 3)
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)

        start_time = time.time()
        best_valid_loss = 1000000
        # train_data = Data(x= data['user'].x, edge_index=data.train_pos_edge_index)
        # data.x = data['user'].x
        # data.num_edges = data.train_pos_edge_index.size(0)
        # data.edge_index = data.train_pos_edge_index
        
        # loader = GraphSAINTRandomWalkSampler(
        #     train_data, batch_size=args.batch_size, walk_length=2, num_steps=args.num_steps,
        # )
        loader = torch.utils.data.DataLoader(dataset=train_data, 
                                          batch_size=args.batch_size, 
                                          shuffle=False)

        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()
            # data = data.to(device)
            epoch_loss = 0
            for step, batch in enumerate(tqdm(loader, desc='Step',position=0, leave=True)):
                # Positive and negative sample
                # train_pos_edge_index = batch.edge_index.to(device)
                # x = model(batch.x.to(device), train_pos_edge_index)

                # out = F.softmax(x, dim=1)
                # print(out.size())
                out = model(batch)
                loss = self.loss_fn(out, label)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                print(loss)
                epoch_loss += loss.item()

            if (epoch+1) % args.valid_freq == 0:
                # precision, recall, ndcg, df_auc, df_aup, df_logit, logit_all_pair, valid_log= self.eval(args, model, data, stage='val')
                # valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': epoch_loss / step
                }
                
                for log in [train_log, valid_log]:
                    # wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))
                    self.trainer_log['log'].append(log)

                # if dt_auc + df_auc > best_metric:
                if best_metric < df_auc + df_aup:
                    best_recall = recall
                    best_epoch = epoch
                    best_precision = precision
                    best_ndcg = ndcg
                    best_metric = df_auc + df_aup
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

