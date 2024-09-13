import torch
import numpy as np




def gap(num_users, pb, topk_ind):
    topk_ind = topk_ind.cpu()

    best, diverse, niche = pb['best'], pb['diverse'], pb['niche']

    avg_popularity_of_item_per_user = pb['avg_popularity_of_item_per_user']
    popularity_of_item = pb['popularity_of_item']

    gap_best = avg_popularity_of_item_per_user[best].mean()
    gap_diverse = avg_popularity_of_item_per_user[diverse].mean()
    gap_niche = avg_popularity_of_item_per_user[niche].mean()
    
    rec_popularity_of_item = torch.zeros(num_users)
    for i in range(topk_ind.size(0)):
        rec_popularity_of_item[i] = popularity_of_item[topk_ind[i]].mean()

    gap_best_R = rec_popularity_of_item[best].mean()
    gap_diverse_R = rec_popularity_of_item[diverse].mean()
    gap_niche_R = rec_popularity_of_item[niche].mean()

    delta_gap_best = (gap_best_R - gap_best) / gap_best
    delta_gap_diverse = (gap_diverse_R - gap_diverse) / gap_diverse
    delta_gap_niche = (gap_niche_R - gap_niche) / gap_niche

    gap_R = rec_popularity_of_item.mean()
    gap_G = avg_popularity_of_item_per_user.mean()
    delta_gap_ALL = (gap_R - gap_G) / gap_G

    return  round(delta_gap_ALL.item(),4), round(delta_gap_best.item(),4), round(delta_gap_diverse.item(),4), round(delta_gap_niche.item(),4)