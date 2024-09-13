import os
import pickle
import torch
from torch_geometric.utils import degree
# import networkx as nx

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
data_dir = './data'
rec_datasets = ['MovieLens100K', 'MovieLens1M', 'Douban', 'AmazonBook']
# rec_datasets = ['Douban']
random_seeds = [13, 63, 42, 87, 100]
# random_seed = 63
for random_seed in random_seeds:
    for dataset in rec_datasets:
        with open(os.path.join(data_dir, dataset, f'd_{random_seed}.pkl'), 'rb') as f:
            datasets, data = pickle.load(f)
        print('Directed dataset:', datasets, data)

        degree_of_user = degree(data.train_pos_edge_index[0], dtype=torch.long, num_nodes=data.num_users)
        degree_of_item = degree(data.train_pos_edge_index[1]-data.num_users, dtype=torch.long, num_nodes=data.num_items)

        val, ind = degree_of_item.sort(descending=True)
        # popular_item index 
        popular_item = ind[:data.num_items//5]
        popularity_of_item = degree_of_item/data.num_users

        user_rated_pop_item = torch.zeros(data.num_users)
        sum_popularity_of_item_per_user = torch.zeros(data.num_users)

        for edge in range(data.train_pos_edge_index.size(1)):
            u, i = data.train_pos_edge_index[:,edge]
            i -= data.num_users
            if i in popular_item:
                user_rated_pop_item[u] += 1
            sum_popularity_of_item_per_user[u] += popularity_of_item[i]

        avg_popularity_of_item_per_user = sum_popularity_of_item_per_user/degree_of_user

        percent_of_user_rated_pop_item =  user_rated_pop_item/degree_of_user
        val, ind = percent_of_user_rated_pop_item.sort(descending=True)
        top20 = round(data.num_users*0.2)
        best = ind[:top20]
        diverse = ind[top20:][:-top20]
        niche = ind[-top20:]

        torch.save({'popular_item':popular_item, 
                    'popularity_of_item':popularity_of_item, 
                    'percent_of_user_rated_pop_item':percent_of_user_rated_pop_item, 
                    'avg_popularity_of_item_per_user':avg_popularity_of_item_per_user,
                    'best': best, 'diverse': diverse, 'niche': niche},
                    os.path.join(data_dir, dataset, f'{random_seed}_popularity_bias.pt') 
                        )