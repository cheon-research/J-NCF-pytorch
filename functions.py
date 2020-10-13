import torch
import torch.nn.functional as F

import random
import numpy as np


def negative_sampling(u_cnt, neg_candidates):
    neg_items = []
    for u, cnt in enumerate(u_cnt):
        sampled_items = random.choices(neg_candidates[u], k=cnt)
        neg_items += sampled_items
    return np.array(neg_items)


def dmf_negative_sampling(train_u, train_i, train_r, u_cnt, neg_candidates, n_negs):
    neg_users, neg_items = [], []
    for u, cnt in enumerate(u_cnt):
        sampled_users = [u] * (cnt * n_negs)
        sampled_items = random.choices(neg_candidates[u], k=cnt*n_negs)
        neg_users += sampled_users
        neg_items += sampled_items

    new_users = np.hstack([train_u, np.array(neg_users)])
    new_items = np.hstack([train_i, np.array(neg_items)])

    neg_labels = np.zeros(len(neg_users), dtype=np.float32)
    new_labels = np.hstack([train_r, neg_labels])
    
    return new_users, new_items, new_labels


def TOP1(pos, neg):
    diff = neg - pos
    loss = torch.sigmoid(diff) + torch.sigmoid(torch.pow(neg, 2))
    return torch.mean(loss)


def BPR(pos, neg):
    diff = neg - pos
    return -torch.mean(torch.logsigmoid(diff))


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0