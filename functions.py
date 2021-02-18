import torch
import torch.nn.functional as F

import random
import numpy as np


def one_negative_sampling(u_ids, neg_candidates):
    neg_items = []
    for u in u_ids:
        sampled_items = random.choice(neg_candidates[u])
        neg_items.append(sampled_items)
    return np.array(neg_items)


def bpr_negative_sampling(train_u, train_i, train_r, u_cnt, neg_candidates, n_negs):
    new_users, new_item_i, new_item_j, new_labels = [], [], [], []

    start_idx = 0
    for u, cnt in enumerate(u_cnt):
        end_idx = start_idx + cnt

        users = [u] * (cnt * n_negs)
        item_i = list(train_i[start_idx:end_idx]) * n_negs
        labels = list(train_r[start_idx:end_idx]) * n_negs
        item_j = random.choices(neg_candidates[u], k=cnt*n_negs)

        new_users += users
        new_item_i += item_i
        new_labels += labels
        new_item_j += item_j

        end_idx = start_idx + cnt

    return np.array(new_users), np.array(new_item_i), np.array(new_item_j), np.array(new_labels)

def TOP1(item_i, item_j, n_negs):
    diff = item_j - item_i
    loss = (torch.sigmoid(diff) + torch.sigmoid(torch.pow(item_j, 2)))
    return torch.mean(loss)


def BPR(item_i, item_j):
    diff = item_i - item_j
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


if __name__ == "__main__":
    neg_candidates = {0: [1, 2, 3], 1: [10, 20, 30], 2: [100, 200, 300]}
    print(one_negative_sampling([0, 2], neg_candidates))