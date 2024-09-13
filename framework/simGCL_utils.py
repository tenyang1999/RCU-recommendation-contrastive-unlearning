import torch
import torch.nn.functional as F

def simGCL_cl_loss(indices, perturbed_emb1, perturbed_emb2):

        f = lambda x: torch.exp(x / 0.2)
        norm = lambda x: F.normalize(x , p=2, dim=1)

        user, pos_item, neg_item = indices

        p_user_emb1, p_item_emb1 = perturbed_emb1[user], perturbed_emb1[pos_item]
        p_user_emb2, p_item_emb2 = perturbed_emb2[user], perturbed_emb2[pos_item]

        norm_emb_user1, norm_emb_item1 = norm(p_user_emb1), norm(p_item_emb1)
        norm_emb_user2, norm_emb_item2 = norm(p_user_emb2), norm(p_item_emb2)

        pos_score_u = f(torch.sum(norm_emb_user1 * norm_emb_user2, dim=1))

        pos_score_i = f(torch.sum(norm_emb_item1 * norm_emb_item2, dim=1))

        ttl_score_u = torch.matmul(norm_emb_user1, norm_emb_user2.t())
        ttl_score_u = torch.sum(f(ttl_score_u), dim=1)

        ttl_score_i = torch.matmul(norm_emb_item1, norm_emb_item2.t())
        ttl_score_i = torch.sum(f(ttl_score_i), dim=1)
        
        cl_loss = -torch.sum(torch.log(pos_score_u / ttl_score_u)) - torch.sum(torch.log(pos_score_i / ttl_score_i))

        return cl_loss