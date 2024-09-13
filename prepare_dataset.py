import os
import math
import pickle
import torch
import pandas as pd
# import networkx as nx
from tqdm import tqdm
from torch_geometric.seed import seed_everything
import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens100K, MovieLens, IGMCDataset, AmazonBook, MovieLens1M
from torch_geometric.utils import k_hop_subgraph, negative_sampling, is_undirected, to_networkx, degree
from torch_geometric.data import Data, HeteroData

data_dir = './data'
# df_size = [i / 100 for i in range(10)] + [i / 10 for i in range(10)] + [i for i in range(10)]       # Df_size in percentage
seeds = [42, 63, 13, 87, 100]
# seeds = [63]

# graph_datasets = ['Cora', 'PubMed', 'DBLP', 'CS', 'ogbl-citation2', 'ogbl-collab'][:4]

rec_datasets = ['MovieLens100K', 'MovieLens1M', 'Douban']#, 'AmazonBook']
# rec_datasets = ['Douban']
os.makedirs(data_dir, exist_ok=True)

def train_val_split_edges_no_neg_adj_mask(data, val_ratio: float = 0.1, test_ratio: float = 0, two_hop_degree=None):
    '''Avoid adding neg_adj_mask'''
    
    num_nodes = data.num_nodes
    row, col = data.edge_index
    
    data.edge_index = None
    
    # mask = row < col
    # row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    if two_hop_degree is not None:          # Use low degree edges for test sets
        low_degree_mask = two_hop_degree < 50

        low = low_degree_mask.nonzero().squeeze()
        high = (~low_degree_mask).nonzero().squeeze()

        low = low[torch.randperm(low.size(0))]
        high = high[torch.randperm(high.size(0))]

        perm = torch.cat([low, high])

    else:
        perm = torch.randperm(row.size(0))

    row = row[perm]
    col = col[perm]
     

    # Train # choose edge after val and test
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    
    assert not is_undirected(data.train_pos_edge_index)

    
    # Test
    if test_ratio != 0:
        r, c = row[:n_t], col[:n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)

    neg_edge_index = negative_sampling(
        edge_index=data.test_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.test_pos_edge_index.shape[1])

    data.test_neg_edge_index = neg_edge_index

    # Valid
    r, c = row[n_t:n_t+n_v], col[n_t:n_t+n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)

    neg_edge_index = negative_sampling(
        edge_index=data.val_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.val_pos_edge_index.shape[1])

    data.val_neg_edge_index = neg_edge_index

    return data

def process_graph():
    data_dir = './data'
    # d = 'MovieLens100K'
    for d in rec_datasets:
        if d in 'MovieLens':
            dataset = MovieLens(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
        elif d == 'MovieLens100K':
            dataset = MovieLens100K(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
        elif d == 'MovieLens1M':
            dataset = MovieLens1M(os.path.join(data_dir, d), transform=T.NormalizeFeatures())    
        elif d == 'Douban':
            dataset = IGMCDataset(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
        elif d == 'AmazonBook':
            dataset = AmazonBook(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
        else:
                raise NotImplementedError
        
        print('Processing:', d)
        print(dataset)
        data = dataset[0]

        # data = data.to(device)

        if d in ['MovieLens100K', 'MovieLens', 'MovieLens1M']:
            item = 'movie'
        elif d in ['Douban']:
            item = 'item'
        elif d in ['AmazonBook']:
            item = 'book'

        try:    
            num_users, dim_users  = data['user'].x.size()
            num_items, dim_items  = data[item].x.size()
        except:
            num_users, num_items = data['user'].num_nodes, data[item].num_nodes

        train_edge_index = data["user", "rates", item].edge_index
        # train_label = data["user", "rates", item].rating
        train_edge_index[1] = train_edge_index[1]+ num_users

        if 'edge_label_index' in data.keys():
            test_edge_index = data["user", "rates", item].edge_label_index
            # test_label = data["user", "rates", item].edge_label    
            test_edge_index[1] = test_edge_index[1]+ num_users

        if 'ogbl' in d:
            # fine 2 hop neighbor degree
            train_data = Data(edge_index=train_edge_index, num_nodes= num_items+num_users)
            graph = to_networkx(train_data).to_undirected()

            node_to_neighbors = {}
            for n in tqdm(graph.nodes(), desc='Two hop neighbors'):
                neighbor_1 = set(graph.neighbors(n))
                neighbor_2 = sum([list(graph.neighbors(i)) for i in neighbor_1], [])
                neighbor_2 = set(neighbor_2)
                neighbor = neighbor_1 | neighbor_2
                
                node_to_neighbors[n] = neighbor

            two_hop_degree = []
            row, col = train_data.edge_index
            # mask = row < col
            # row, col = row[mask], col[mask]
            for r, c in tqdm(zip(row, col), total=len(row)):
                neighbor_row = node_to_neighbors[r.item()]
                neighbor_col = node_to_neighbors[c.item()]
                neighbor = neighbor_row | neighbor_col
                
                num = len(neighbor)
                
                two_hop_degree.append(num)

            two_hop_degree = torch.tensor(two_hop_degree)

        

    

        for s in seeds:
            seed_everything(s)
            if 'edge_label_index' in data.keys():
                train_data = HeteroData(edge_index=train_edge_index, test_pos_edge_index= test_edge_index, num_nodes= num_items+num_users, )
            else:
                train_data = HeteroData(edge_index=train_edge_index, num_nodes= num_items+num_users, )

            if 'x' in data.keys():
                train_data['user'].x = data['user'].x
                train_data['item'].x = data[item].x
            # D
            # data = dataset[0]
            # add_edge_index = data.edge_index[:, add_mask]

            # data.edge_index = data.edge_index[:, ~add_mask]
            if 'ogbl' in d:
                train_data = train_val_split_edges_no_neg_adj_mask(train_data, val_ratio=0.1, two_hop_degree=two_hop_degree)
            elif 'edge_label_index' in data.keys():
                train_data = train_val_split_edges_no_neg_adj_mask(train_data, val_ratio=0.1)
            else:
                train_data = train_val_split_edges_no_neg_adj_mask(train_data, val_ratio=0.1, test_ratio=0.2)

            # data.add_edge_index = add_edge_index
            print(s, train_data)
            # print(data.keys())

            train_data.num_users = num_users
            train_data.num_items = num_items
            

            with open(os.path.join(data_dir, d, f'd_{s}.pkl'), 'wb') as f:
                pickle.dump((dataset, train_data), f)

            # Two ways to sample Df from the training set
            # high: degree > 50, low: degree<= 50
            
                
            node_count = degree(train_data.test_pos_edge_index[0], num_nodes=num_users)
            cut_point= round(num_users*0.2)

            val,ind = node_count.sort(descending=True)
            high_user = torch.zeros(num_users, dtype=torch.bool)
            high_user[ind[:cut_point]] = True
            low_user = ~high_user

            node_count = degree(train_data.test_pos_edge_index[1]-num_users, num_nodes=num_items)
            cut_point= round(num_items*0.2)

            val,ind = node_count.sort(descending=True)
            high_item = torch.zeros(num_items, dtype=torch.bool)
            high_item[ind[:cut_point]] = True
            low_item = ~high_item


            torch.save(
                    {'high_user': high_user, 'low_user': low_user,
                     'high_item': high_item, 'low_item': low_item,
                     },
                    os.path.join(data_dir, d, f'df_{s}.pt')
                )
            


def main():
    process_graph()

if __name__ == "__main__":
    main()
