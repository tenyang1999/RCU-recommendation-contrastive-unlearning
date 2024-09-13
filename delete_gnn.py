import os
import copy
import json
# import wandb
import pickle
import argparse
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected, to_networkx, k_hop_subgraph, is_undirected
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.seed import seed_everything
# from torchsummary import summary
from framework import get_model, get_trainer
# from framework.models.gcn import GCN
from framework.training_args import parse_args
from framework.utils import get_ii_constraint_mat
# from train_mi import MLPAttacker
from scipy.sparse import dok_matrix
import numpy as np
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_args(path):
    with open(path, 'r') as f:
        d = json.load(f)
    parser = argparse.ArgumentParser()
    for k, v in d.items():
        parser.add_argument('--' + k, default=v)
    try:
        parser.add_argument('--df_size', default=0.5)
    except:
        pass
    args = parser.parse_args()

    for k, v in d.items():
        setattr(args, k, v)

    return args

def main():
    args = parse_args()
    original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'original', str(args.random_seed))
    seed_everything(args.random_seed)

    if args.df_node:
        args.checkpoint_dir = os.path.join(
            args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, 'node',
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]])) # args.add_size, 
    else:
        args.checkpoint_dir = os.path.join(
            args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, 
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]])) # args.add_size, 
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset
    # if args.dataset == 'AmazonBook':
    #     with open(os.path.join(args.data_dir, args.dataset, f'sd_{args.random_seed}.pkl'), 'rb') as f:
    #         dataset, data = pickle.load(f)
    # else:    
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)
    print('Directed dataset:', dataset, data)
    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = dataset.num_features
    popularity_bias = torch.load(os.path.join(args.data_dir, args.dataset,  f'{args.random_seed}_popularity_bias.pt'))
    print('Training args', args)
    # wandb.init(project="GNNDelete", config = args)

    # Df and Dr
    assert args.df != 'none'

    if args.df_node:     # df_size is number of nodes/edges to be deleted
        if 'user' in args.df:
            df_size = int(args.df_size / 100 * data.num_users)
        else:
            df_size = int(args.df_size / 100 * data.num_items)
    else:                       # df_size is the ratio
        df_size = int(args.df_size / 100 * data.train_pos_edge_index.size(1))

    # if args.add_size >= 100:     # add_size is number of nodes/edges to be added
    #     add_size = int(args.add_size)
    # else:                       # add_size is the ratio
    #     add_size = int(10* args.add_size / 100 * data.add_edge_index.shape[1])

    print(f'Original size: {data.train_pos_edge_index.shape[1]:,}')
    print(f'Df size: {df_size:,}')
    # print(f'Add size: {add_size:,}')

    if args.df != 'random':
        tar = torch.load(os.path.join(args.data_dir, args.dataset, f'df_{args.random_seed}.pt'))[args.df]
        tar = tar.nonzero().squeeze()
        # print(tar)
    else:
        tar = torch.arange(data.num_users)
        # if 'user' in args.df:
        #     tar = torch.arange(data.num_users)
        # else:
        #     tar = torch.arange(data.num_items)

    if args.df_node :
        idx = torch.randperm(tar.size(0))[:df_size]
        idx = tar[idx]
        if 'user' in args.df:
            df_mask_all = torch.Tensor([ True if i in idx else False for i in data.train_pos_edge_index[0]]).bool()
        else:
            df_mask_all = torch.Tensor([ True if i in idx else False for i in data.train_pos_edge_index[1] - data.num_users]).bool()
        df_global_idx = df_mask_all.nonzero().squeeze()
        print(f'Deleting the following {args.df} :', idx)
        data.df_node = idx
    else:
        df_mask_all = torch.Tensor([ True if i in tar else False for i in data.train_pos_edge_index[0]])   
        df_nonzero = df_mask_all.nonzero().squeeze()

        idx = torch.randperm(df_nonzero.size(0))[:df_size]
        df_global_idx = df_nonzero[idx]

    print('Deleting the following edges:', df_global_idx, ' delete edge size:', df_global_idx.size(0))

    # add edge index 接在後段
    # add_idx = torch.randperm(data.add_edge_index.size(1))[:add_size]
    # add_edge_index = data.add_edge_index[:, add_idx]
    # data.train_pos_edge_index = torch.hstack((data.train_pos_edge_index, add_edge_index))
    
    dr_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    dr_mask[df_global_idx] = False

    df_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    df_mask[df_global_idx] = True

    # For testing
    data.directed_df_edge_index = data.train_pos_edge_index[:, df_mask]

    # To undirected for message passing
    # assert not is_undirected(data.train_pos_edge_index)
    # if 'x' in data.keys():   
    #     args.num_users, args.dim_users  = data['user'].x.size()
    #     args.num_items, args.dim_items  = data['item'].x.size()
    # else:
    #     args.num_users = data.num_users
    #     args.num_items = data.num_items
    args.num_users = data.num_users
    args.num_items = data.num_items

    if args.gnn == 'UltraGCN':

        dr_edge =  data.train_pos_edge_index[:, dr_mask]        
        train_mat = dok_matrix((data.num_users, data.num_items), dtype=np.float32)
        for x in dr_edge.T:   
                train_mat[x[0], x[1]-data.num_users] = 1.0    
        ii_neighbor_mat, ii_constraint_mat = get_ii_constraint_mat(train_mat, num_neighbors =args.num_neighbors) 
        ii_neighbor_mat, ii_constraint_mat =ii_neighbor_mat.to(device), ii_constraint_mat.to(device)

        items_D = np.sum(train_mat, axis = 0).reshape(-1)
        users_D = np.sum(train_mat, axis = 1).reshape(-1)

        beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
        beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

        constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1).to(device),
                        "beta_iD": torch.from_numpy(beta_iD).reshape(-1).to(device)}
    
        data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
        data = data.to(device)
        model = get_model(args,  constraint_mat=constraint_mat, ii_constraint_mat=ii_constraint_mat, ii_neighbor_mat=ii_neighbor_mat).to(device)
    
    else:
        train_pos_edge_index, [df_mask] = to_undirected(data.train_pos_edge_index, [df_mask.int()])
        data.train_pos_edge_index = train_pos_edge_index
        df_mask = df_mask.bool()
        dr_mask = ~df_mask
        # Model
        model = get_model(args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type) # sdf_node_1hop, sdf_node_2hop, 
        
    data.df_mask = df_mask
    data.dr_mask = dr_mask




    if args.unlearning_model != 'retrain':  # Start from trained GNN model
        if os.path.exists(os.path.join(original_path, 'node_embeddings.pt')):
            z_ori = torch.load(os.path.join(original_path, 'node_embeddings.pt'))
            if z_ori is not None:
                z_ori = z_ori.to(device)
        else:
            z_ori = None

        model_ckpt = torch.load(os.path.join(original_path, 'model_best.pt'), map_location=device)
        model.load_state_dict(model_ckpt['model_state'], strict=False)
        
    else:       # Initialize a new GNN model
        retrain = None
        z_ori = None

    model = model.to(device)
    data = data.to(device)

    # print(model.named_parameters())
    if 'gnndelete' in args.unlearning_model and 'nodeemb' in args.unlearning_model:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters() if 'del' in n])

        if 'layerwise' in args.loss_type:
            optimizer1 = torch.optim.Adam(model.deletion1.parameters(), lr=args.lr)
            optimizer2 = torch.optim.Adam(model.deletion2.parameters(), lr=args.lr)
            optimizer = [optimizer1, optimizer2]
        else:
            optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)

    else:
        if 'gnndelete' in args.unlearning_model:
            parameters_to_optimize = [
                {'params': [p for n, p in model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
            ]
            print('parameters_to_optimize', [n for n, p in model.named_parameters() if 'del' in n])
        
        else:
            parameters_to_optimize = [
                {'params': [p for n, p in model.named_parameters()], 'weight_decay': 0.0}
            ]
            print('parameters_to_optimize', [n for n, p in model.named_parameters()])
        
        optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    
    # wandb.watch(model, log_freq=100)

    # # MI attack model
    attack_model_all = None
    attack_model_sub = None

    # Train
    trainer = get_trainer(args)
    # trainer.train(args, model, data, optimizer)

    trainer.train(model, data, optimizer, args, popularity_bias, z_ori, attack_model_all, attack_model_sub)

    # Test
    # retrain_path = os.path.join(
    #     'checkpoint', args.dataset, args.gnn, 'retrain', 
    #     '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]), 
    #     'model_best.pt')
    # if os.path.exists(retrain_path):
    #     retrain_ckpt = torch.load(retrain_path, map_location=device)
    #     retrain_args = copy.deepcopy(args)
    #     retrain_args.unlearning_model = 'retrain'
    #     retrain = get_model(retrain_args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
    #     retrain.load_state_dict(retrain_ckpt['model_state'])
    #     retrain = retrain.to(device)
    #     retrain.eval()
    # else:
    #     retrain = None
    
    test_results = trainer.test(args, model, data, popularity_bias, attack_model_all=attack_model_all, attack_model_sub=attack_model_sub)
    print(test_results[:9])
    print(test_results[-1])
    result = test_results[-1]
    with open('output.txt', 'a') as f:
        f.write('\n')
        f.write(f'{args.dataset}, {args.unlearning_model}, {args.gnn},  {args.random_seed}, {args.batch_size}, {args.epochs}, {args.df}, {args.df_size}, ')
        f.write(f"{trainer.trainer_log['best_training_time']}, ")
        # f.write(f'{test_results[-1]}')  #文件的写操作
        # f.write(json.dumps(test_results[-1]))
        for i in test_results[:5]:
            f.write(f'{i}, ')

        f.write(f"{args.i1}, {args.i2}, {args.i3}, ")
        f.write(f"{args.alpha}, ")

        f.write(f"{args.df_node}")

        # for i in  result["test_group_precision"].tolist():
        #     f.write(f'{round(i,4)}, ')
        # for i in  result["test_group_recall"].tolist():
        #     f.write(f'{round(i,4)}, ')  
        # for i in  result["test_group_nDCG"].tolist():
        #     f.write(f'{round(i,4)}, ')        
        
    
    trainer.save_log()


if __name__ == "__main__":
    main()
