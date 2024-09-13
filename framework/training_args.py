import argparse


num_edge_type_mapping = {
    'FB15k-237': 237,
    'WordNet18': 18,
    'WordNet18RR': 11,
    'ogbl-biokg': 51
}

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument('--unlearning_model', type=str, default='original',
                        help='unlearning method')
    parser.add_argument('--gnn', type=str, default='LightGCN', 
                        help='GNN architecture')
    parser.add_argument('--in_dim', type=int, default=128, 
                        help='input dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, 
                        help='hidden dimension')
    parser.add_argument('--out_dim', type=int, default=64, 
                        help='output dimension')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='data dir')
    parser.add_argument('--df', type=str, default='high_user',
                        help='Df set to use: high degrees users or low degrees users, ["high_user", "low_user", "random"]')
    parser.add_argument('--df_node', type=str, default=False,
                        help='data deleted nodes or edges') 
    
    parser.add_argument('--df_idx', type=str, default='none',
                        help='indices of data to be deleted')
    parser.add_argument('--df_size', type=float, default=0.5,
                        help='Df size')
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='dataset')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=512, 
                        help='batch size')
    parser.add_argument('--walk_length', type=int, default=2,
                        help='random walk length for GraphSAINTRandomWalk sampler')
    parser.add_argument('--num_steps', type=int, default=32,
                        help='number of steps for GraphSAINTRandomWalk sampler')

    # Training
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, 
                        help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', 
                        help='optimizer to use')
    parser.add_argument('--epochs', type=int, default=3000, 
                        help='number of epochs to train')
    parser.add_argument('--valid_freq', type=int, default=10,
                        help='# of epochs to do validation')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                        help='checkpoint folder')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='alpha in loss function')
    parser.add_argument('--neg_sample_random', type=str, default='non_connected',
                        help='type of negative samples for randomness')
    parser.add_argument('--loss_fct', type=str, default='mse_mean',
                        help='loss function. one of {mse, kld, cosine}')
    parser.add_argument('--loss_type', type=str, default='both_layerwise',
                        help='type of loss. one of {both_all, both_layerwise, only2_layerwise, only2_all, only1}')
    parser.add_argument('--retrain_auc', type=float, default=0.05,
                        help='metric avoid over-unlearning')

    # # GraphEraser
    parser.add_argument('--num_clusters', type=int, default=10, 
                        help='top k for evaluation')
    parser.add_argument('--kmeans_max_iters', type=int, default=1, 
                        help='top k for evaluation')
    parser.add_argument('--shard_size_delta', type=float, default=0.005)
    parser.add_argument('--terminate_delta', type=int, default=0)
    parser.add_argument('--epoch_agg', type=int, default=500)

    # RecEraser
    parser.add_argument('--num_local', type=int, default=10)
    

    # # GraphEditor
    # parser.add_argument('--eval_steps', type=int, default=1)
    # parser.add_argument('--runs', type=int, default=1)

    # parser.add_argument('--num_remove_links', type=int, default=11)
    # parser.add_argument('--parallel_unlearning', type=int, default=4)

    # parser.add_argument('--lam', type=float, default=0)
    # parser.add_argument('--regen_feats', action='store_true')
    # parser.add_argument('--regen_neighbors', action='store_true')
    # parser.add_argument('--regen_links', action='store_true')
    # parser.add_argument('--regen_subgraphs', action='store_true')
    # parser.add_argument('--hop_neighbors', type=int, default=20)

    # contrastive loss
    parser.add_argument('--tau', type=float, default=0.3, 
                        help='tau for contrastive loss')


    # Nora
    parser.add_argument('--k1', type=float, default=1,
                        help="For method 'nora': k1, within [0, 1]")
    parser.add_argument('--k2', type=float, default=[1, 0], nargs='+',
                        help="For method 'nora': k2 and k2', within [0, 1]")
    parser.add_argument('--k3', type=float, default=1,
                        help="For method 'nora': k3', usuall within [0.5, 5]")
    parser.add_argument('--self_buff', type=float, default=3.0, 
                        help="For method 'nora': beta")
    parser.add_argument('--grad_norm', type=float, default=1, 
                        help="For method 'nora': p-norm")
    
    
    

    # # Item score 
    parser.add_argument('--i1', type=float, default=0.2,
                        help="Use in different user group weight, usuall within LightGCN [0.2, 0.4, 1]")   #[0, 0.1, 0.4]
    parser.add_argument('--i2', type=float, default=0.4)
    parser.add_argument('--i3', type=float, default=1)
    
    
    # UltraGCN
    #L = -(w1 + w2*\beta)) * log(sigmoid(e_u e_i)) - \sum_{N-} (w3 + w4*\beta) * log(sigmoid(e_u e_i'))

    # # MovieLens1M
    parser.add_argument('--initial_weight', type=float, default=1e-3)
    
    parser.add_argument('--w1', type=float, default=1e-7)
    parser.add_argument('--w2', type=float, default=1)
    parser.add_argument('--w3', type=float, default=1e-7)
    parser.add_argument('--w4', type=float, default=1)
    parser.add_argument('--negative_num', type=float, default=200)
    parser.add_argument('--negative_weight', type=float, default=200)
    
    parser.add_argument('--gamma', type=float, default=1e-4,
                        help="weight of l2 normalization")
    parser.add_argument('--lambda_', type=float, default=1e-3,
                        help="weight of L_I")
    parser.add_argument('--num_neighbors', type=float, default=10)

    
    # SimGCL
    # 𝜆 is 0.2 on Douban-Book,0.5 on Yelp2018,and 2 onAmazon-Book
    # parser.add_argument('--cl_rate', type=float, default=2)
    # parser.add_argument('--eps', type=float, default=0.1)
    
    # CGU
    parser.add_argument('--prop_step', type=int, default=2, help='number of steps of graph propagation/convolution')
    parser.add_argument('--delta', type=float, default=1e-4, help='Delta coefficient for certified removal.')
    parser.add_argument('--std', type=float, default=1e-2, help='standard deviation for objective perturbation')
    parser.add_argument('--compare_gnorm', action='store_true', default=False,
                        help='Compute norm of worst case and real gradient each round.')
    parser.add_argument('--lam', type=float, default=1e-2, help='L2 regularization')


    # Evaluation
    parser.add_argument('--topk', type=int, default=10, 
                        help='top k for evaluation')
    parser.add_argument('--eval_on_cpu', type=bool, default=False, 
                        help='whether to evaluate on CPU')

    # KG
    parser.add_argument('--num_edge_type', type=int, default=None, 
                        help='number of edges types')

    args = parser.parse_args()

    # if args.unlearning_model in ['original', 'retrain']:
    #     args.valid_freq = 10

    if 'gnndelete' in args.unlearning_model:
        if args.gnn not in ['rgcn', 'rgat'] and 'ogbl' in args.dataset:
            args.epochs = 600
            args.valid_freq = 100
        if args.gnn in ['rgcn', 'rgat']:
            if args.dataset == 'WordNet18':
                args.epochs = 50
                args.valid_freq = 2
                args.batch_size = 1024
            if args.dataset == 'ogbl-biokg':
                args.epochs = 50
                args.valid_freq = 10
                args.batch_size = 64

    elif args.unlearning_model == 'gradient_ascent':
        args.epochs = 10
        args.valid_freq = 1
    
    elif args.unlearning_model == 'descent_to_delete':
        args.epochs = 1

    elif args.unlearning_model == 'graph_editor':
        args.epochs = 400
        args.valid_freq = 200

    elif args.unlearning_model == 'RecEraser':
        args.batch_size = 512
        args.lr=0.001
        

    return args
