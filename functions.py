import torch
import torch.nn as nn
import torch.nn.functional as F

import time, os, random
import numpy as np

def negative_sampling(train_u, train_i, u_cnt, neg_candidates, n_negs):
    idxlist = np.array(range(len(train_u)))
    
    new_users, new_items = [], []
    new_labels = np.ones(len(new_users), dtype=np.float32)
    
    neg_users, neg_items = [], []
    for u, cnt in enumerate(u_cnt):
        sampled_users = [u] * (cnt * n_negs)
        sampled_items = random.choices(neg_candidates[u], k=cnt*n_negs)
        neg_users += sampled_users
        neg_items += sampled_items
    '''
    new_users = np.hstack([train_u, np.array(neg_users)])
    new_items = np.hstack([train_i, np.array(neg_items)])
    new_labels = np.ones(len(train_u), dtype=np.float32)

    neg_labels = np.zeros(len(neg_users), dtype=np.float32)
    new_labels = np.hstack([new_labels, neg_labels])
    #print('Negative Samples', len(new_users), len(new_items), len(new_labels))
    '''
    return np.array(neg_users), np.array(neg_items)

def vaepp_loss_function(recon_x, x, mu, logvar):
    #BCE = nn.BCEWithLogitsLoss(recon_x, x.view(-1, 784), reduction='sum')
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + KLD

def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0

def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0

'''
new_users = np.take(train_u, idxlist)
new_items = np.take(train_i, idxlist)
new_labels = np.ones(len(new_users), dtype=np.float32)

time_neg = time.time()
neg_users, neg_items = [], []
for u in range(0, n_users):
    cnt = u_cnt[u]
    sampled_users = [u] * (cnt * n_negs)
    sampled_items = random.choices(neg_dict[u], k=cnt*n_negs)
    neg_users += sampled_users
    neg_items += sampled_items

new_users = np.hstack([train_u, np.array(neg_users)])
new_items = np.hstack([train_i, np.array(neg_items)])
new_labels = np.ones(len(train_u), dtype=np.float32)

neg_labels = np.zeros(len(neg_users), dtype=np.float32)
new_labels = np.hstack([new_labels, neg_labels])
print('Negative Samples', len(new_users), len(new_items), len(new_labels))
print('Negative Sampling time:\t{:.4f}'.format(time.time() - time_neg))
'''