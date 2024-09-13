import torch
import os
import pickle
import time

def E_score2(a,b):
    return (torch.pow(a-b, 2)).sum()
seeds = [63, 13, 87, 100, 42]
for dataset in [ 'AmazonBook']:
   for random_seed in seeds:
      start_time = time.time()
      data_dir = './data'
      # dataset = 'MovieLens100K'
      # dataset = 'DBLP'
      # random_seed = 42

      with open(os.path.join(data_dir, dataset, f'd_{random_seed}.pkl'), 'rb') as f:
         datasets, data = pickle.load(f)
      print('Directed dataset:', datasets, data)
      ori_path = f'checkpoint/{dataset}/LightGCN/original/{random_seed}'
      z = torch.load(os.path.join(ori_path, 'node_embeddings.pt')).cpu()

      edge_index = data.train_pos_edge_index
      k = 10
      # Randomly select k centroids
      max_data = 1.2 * edge_index.size(1) / k
      idx = torch.randperm(edge_index.size(1))[:k]
      centroids = edge_index[:,idx].T
      edge_index = edge_index.T

      centroembs = []
      for i in range(k):
         temp_u = z[centroids[i][0]]
         temp_i = z[centroids[i][1]]
         centroembs.append([temp_u, temp_i])

      for _ in range(50):
         C = [{} for i in range(k)]
         C_num = [0 for i in range(k)]
         Scores = {}
         shards = [[] for i in range(k)]
         
         for i in range(len(edge_index)):
            for j in range(k):
                  # print(z.size())
                  score_u = E_score2(z[edge_index[i][0]],centroembs[j][0])
                  score_i = E_score2(z[edge_index[i][1]],centroembs[j][1])          
                  Scores[i, j] = (-score_u * score_i).item()

         Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)

         fl = set()
         for i in range(len(Scores)):
            tar_edge = Scores[i][0][0]

            if tar_edge not in fl:
               tar_shard = Scores[i][0][1]

               if C_num[tar_shard] < max_data:
                  if edge_index[tar_edge][0] not in C[tar_shard]:
                     C[tar_shard][edge_index[tar_edge][0].item()]=[edge_index[tar_edge][1].item()]
                  else:
                     C[tar_shard][edge_index[tar_edge][0].item()].append(edge_index[tar_edge][1].item())
                  fl.add(tar_edge)
                  shards[tar_shard].append(edge_index[tar_edge])
                  C_num[tar_shard] +=1
                  
         centroembs_next = []
         for i in range(k):
            temp_u = torch.zeros(z.size(1))
            temp_i = torch.zeros(z.size(1))
            cnt = 0
            for j in C[i].keys():
                  for l in C[i][j]:
                     temp_u+= z[j]
                     temp_i+= z[l]
                     cnt +=1
            centroembs_next.append([temp_u/cnt, temp_i/cnt])

         loss = 0.0

         for i in range(k):
            score_u = E_score2(centroembs_next[i][0],centroembs[i][0])
            score_i = E_score2(centroembs_next[i][1],centroembs[i][1])

            loss += (score_u * score_i)

         centroembs = centroembs_next

         # for i in range(k): 
         #    print(C_num[i])

         print( _, loss)

      for i in range(k): 
         shards[i] = torch.stack(shards[i]).T

      print(time.time()-start_time)
      torch.save(shards, os.path.join(data_dir, dataset, f'{random_seed}_shards.pt'))