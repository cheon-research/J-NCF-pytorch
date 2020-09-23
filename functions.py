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


def TOP1(pos, neg, num_ng):
    diff = neg - pos
    loss = torch.sigmoid(diff) + torch.sigmoid(torch.pow(neg, 2))
    return torch.mean(loss)


def BPR(pos, neg):
    diff = neg - pos
    return -torch.mean(torch.logsigmoid(diff))


def explicit_log(y_hat, y, y_max):
    Y_ui = y / y_max
    loss = - Y_ui * torch.log(y_hat) - (1 - Y_ui) * torch.log(1 - y_hat)
    return torch.sum(loss)


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0