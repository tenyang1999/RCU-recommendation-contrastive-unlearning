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
# from framework.utils import *
# from train_mi import MLPAttacker


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
    # start from the origianl sub model 
    original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'RecEraser_Ori', str(args.random_seed))
    seed_everything(args.random_seed)
    args.original_path = original_path

    if 'gnndelete' in args.unlearning_model:
        args.checkpoint_dir = os.path.join(
            args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, 
            '-'.join([str(i) for i in [args.loss_fct, args.loss_type, args.alpha, args.neg_sample_random]]),
            '-'.join([str(i) for i in [args.df, args.df_size,  args.random_seed]])) #  args.add_size,
    else:
        args.checkpoint_dir = os.path.join(
            args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, 
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]])) # args.add_size, 
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
            dataset, data = pickle.load(f)
    print('Directed dataset:', dataset, data)
    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = dataset.num_features
    popularity_bias = torch.load(os.path.join(args.data_dir, args.dataset, f'popularity_bias.pt'))
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

    
    # train_pos_edge_index, [df_mask] = to_undirected(data.train_pos_edge_index, [df_mask.int()])
    # two_hop_mask = two_hop_mask.bool()
    df_mask = df_mask.bool()
    dr_mask = ~df_mask
    
    # data.train_pos_edge_index = train_pos_edge_index
    # data.edge_index = train_pos_edge_index
    # assert is_undirected(data.train_pos_edge_index)

    # Edges in S_Df
    # _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
    #     data.train_pos_edge_index[:, df_mask].flatten().unique(), 
    #     2, 
    #     data.train_pos_edge_index,
    #     num_nodes=data.num_nodes)
    # data.sdf_mask = two_hop_mask
    # print('Undirected dataset:', data)

    # data.sdf_mask = two_hop_mask
    data.df_mask = df_mask
    data.dr_mask = dr_mask

    if 'x' in data.keys():   
        args.num_users, args.dim_users  = data['user'].x.size()
        args.num_items, args.dim_items  = data['item'].x.size()
    else:
        args.num_users = data.num_users
        args.num_items = data.num_items

    shard = torch.load(os.path.join(args.data_dir, args.dataset, f'{args.random_seed}_shards.pt'))
    # model_para = []
    # for i in range(len(shard)):
    #     model_para.append(torch.load(os.path.join(original_path, f'{i}_model_best.pt')))

    # Model
    model = get_model(args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type) # sdf_node_1hop, sdf_node_2hop, 

    model = model.to(device)
    data = data.to(device)
    
    z_ori = None

    # # MI attack model
    attack_model_all = None
    attack_model_sub = None

    # Train
    trainer = get_trainer(args)
    # trainer.train(args, model, data, optimizer)

    trainer.train(model, data, args, popularity_bias, z_ori, shard, attack_model_all, attack_model_sub)

    # Test
    # if args.unlearning_model != 'retrain':
    #     retrain_path = os.path.join(
    #         'checkpoint', args.dataset, args.gnn, 'retrain', 
    #         '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]), 
    #         'model_best.pt')
    #     if os.path.exists(retrain_path):
    #         retrain_ckpt = torch.load(retrain_path, map_location=device)
    #         retrain_args = copy.deepcopy(args)
    #         retrain_args.unlearning_model = 'retrain'
    #         retrain = get_model(retrain_args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
    #         retrain.load_state_dict(retrain_ckpt['model_state'])
    #         retrain = retrain.to(device)
    #         retrain.eval()
    #     else:
    #         retrain = None
    # else:
    #     retrain = None
    retrain = None
    
    test_results = trainer.test(args, model, data, popularity_bias, model_retrain=retrain, attack_model_all=attack_model_all, attack_model_sub=attack_model_sub)
    print(test_results[:9])
    print(test_results[-1])
    result = test_results[-1]
    with open('output.txt', 'a') as f:
        f.write('\n')
        f.write(f'{args.dataset}, {args.unlearning_model}, {args.gnn},  {args.random_seed}, {args.batch_size}, {args.epochs}, {args.df}, {args.df_size}, ')
        # f.write(f'{test_results[-1]}')  #文件的写操作
        # f.write(json.dumps(test_results[-1]))
        f.write(f"{trainer.trainer_log['shard_retrain_time']+trainer.trainer_log['best_training_time']}, ")

        for i in test_results[:5]:
            f.write(f'{i}, ')

        # f.write(f"{trainer.trainer_log['shard_retrain_time']}, ")
        # f.write(f"{trainer.trainer_log['best_training_time']}, ")
        
        # for i in  result["test_group_precision"].tolist():
        #     f.write(f'{round(i,4)}, ')
        # for i in  result["test_group_recall"].tolist():
        #     f.write(f'{round(i,4)}, ')  
        # for i in  result["test_group_nDCG"].tolist():
        #     f.write(f'{round(i,4)}, ')        

    
    trainer.save_log(args)


if __name__ == "__main__":
    main()
