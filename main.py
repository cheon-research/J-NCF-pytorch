"""
Implementation for "Joint Neural Collaborative Filtering for Recommender Systems"
ACM Transactions on Information Systems, Vol. 37, No. 4, Article 39. (August 2019)

https://dl.acm.org/doi/10.1145/3343117
https://arxiv.org/pdf/1907.03459.pdf

by Sangjin Cheon (cheon.research @ gmail.com)
University of Seoul, Korea
"""

import torch
import torch.optim as optim
import numpy as np

import time

from model import JNCF
import data_utils
from functions import *


def run(dataset, DF_layers, DI_layers, n_negs, alpha, gpu='0'):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print("##### {} Negative Samples experiment on {} DF: {} DI: {}".format(n_negs, dataset, DF_layers, DI_layers))
    
    learning_rate = 0.0001
    batch_size = 256
    #embed_dim = 256
    #factor_dim = 64

    if torch.cuda.is_available():
        device = torch.device('cuda')
        FloatTensor = torch.cuda.FloatTensor
    else:
        device = torch.device('cpu')
        FloatTensor = torch.FloatTensor
    manualSeed = 706
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    print('CUDA Available:', torch.cuda.is_available())

    file_name = 'output/' + dataset + '_J-NCF_' + str(DF_layers) + '_' + str(DI_layers) + '_n_' + str(n_negs) + '.txt'
    output = open(file_name, 'w')

    # Datasets
    user_matrix, item_matrix, train_u, train_i, train_r, neg_candidates, u_cnt, user_rating_max = data_utils.load_train_data(dataset)
    if dataset == 'ml1m':
        epochs = 100
        eval_batch_size = 100 * 151
        test_users, test_items = data_utils.load_test_ml1m()
    elif dataset == 'ml100k':
        epochs = 100
        eval_batch_size = 100 * 41
        test_users, test_items = data_utils.load_test_data(dataset)
    elif dataset == 'yelp':
        epochs = 50
        eval_batch_size = 100 * 81
        test_users, test_items = data_utils.load_test_data(dataset)
    elif dataset == 'amusic':
        epochs = 100
        eval_batch_size = 100 * 3
        test_users, test_items = data_utils.load_test_data(dataset)
    elif dataset == 'agames':
        epochs = 100
        eval_batch_size = 100 * 34
        test_users, test_items = data_utils.load_test_data(dataset)

    n_users, n_items = user_matrix.shape[0], user_matrix.shape[1]

    user_array = user_matrix.toarray()
    item_array = item_matrix.toarray()
    user_idxlist, item_idxlist = list(range(n_users)), list(range(n_items))

    # Model
    model = JNCF(DF_layers, DI_layers, n_users, n_items, 'concat').to(device)  # 'multi' or 'concat'
    pair_loss_function = TOP1  # TOP1 or BPR
    point_loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_hr = 0.0
    for epoch in range(epochs):
        # Train
        model.train()  # Enable dropout (if have).

        idxlist = np.array(range(len(train_u)))
        np.random.shuffle(idxlist)
        epoch_loss, epoch_pair_loss, epoch_point_loss, epoch_i_point_loss, epoch_j_point_loss = .0, .0, .0, .0, .0

        start_time = time.time()
        for batch_idx, start_idx in enumerate(range(0, len(idxlist), batch_size)):
            end_idx = min(start_idx + batch_size, len(idxlist))
            idx = idxlist[start_idx:end_idx]

            u_ids = train_u.take(idx)
            i_ids = train_i.take(idx)
            i_ratings = train_r.take(idx)
            
            users = FloatTensor(user_array.take(u_ids, axis=0))
            items = FloatTensor(item_array.take(i_ids, axis=0))
            labels = FloatTensor(i_ratings)

            rating_max = FloatTensor(user_rating_max.take(u_ids, axis=0))
            Y_ui = labels / rating_max  # for Normalized BCE
            Y_uj = torch.zeros_like(Y_ui)  # for Negative samples point-wise loss

            optimizer.zero_grad()
            point_loss, pair_loss = 0., 0.

            # Negative Sampling
            neg_items_list = []
            for _ in range(0, n_negs):
                neg_items = one_negative_sampling(u_ids, neg_candidates)
                neg_items_list.append(neg_items)

            for ng_idx in range(0, n_negs):
                neg_ids = neg_items_list[ng_idx]
                items_j = FloatTensor(item_array.take(neg_ids, axis=0))

                y_i, y_j = model(users, items, items_j)

                i_point_loss = point_loss_function(y_i, Y_ui)  # positive items i
                j_point_loss = point_loss_function(y_j, Y_uj)  # negative items j
                point_loss = i_point_loss + j_point_loss
                pair_loss = pair_loss_function(y_i, y_j, n_negs)

                loss = alpha * pair_loss + (1 - alpha) * point_loss

                epoch_loss += loss.item()
                epoch_pair_loss += pair_loss.item()
                epoch_point_loss += point_loss.item()
                epoch_i_point_loss += i_point_loss.item()
                epoch_j_point_loss += j_point_loss.item()

                loss.backward()
                optimizer.step()
        train_time = time.time() - start_time

        # Evaluate
        model.eval()
        HR, NDCG = [], []

        time_E = time.time()
        for start_idx in range(0, len(test_users), eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, len(test_users))
            u_ids = test_users[start_idx:end_idx]
            i_ids = test_items[start_idx:end_idx]

            users = FloatTensor(user_array.take(u_ids, axis=0))
            items = FloatTensor(item_array.take(i_ids, axis=0))

            preds, _ = model(users, items, items)

            e_batch_size = eval_batch_size // 100  # faster eval
            preds = torch.chunk(preds.detach().cpu(), e_batch_size)
            chunked_items = torch.chunk(torch.IntTensor(i_ids), e_batch_size)

            for i, pred in enumerate(preds):
                _, indices = torch.topk(pred, 10)
                recommends = torch.take(chunked_items[i], indices).numpy().tolist()

                gt_item = chunked_items[i][0].item()
                HR.append(hit(gt_item, recommends))
                NDCG.append(ndcg(gt_item, recommends))

        eval_time = time.time() - time_E
        #if epoch % 10 == 0:
        e_loss = epoch_loss / (batch_idx + 1)
        e_pair = epoch_pair_loss / (batch_idx + 1)
        e_point = epoch_point_loss / (batch_idx + 1)
        e_i_point = epoch_i_point_loss / (batch_idx + 1)
        e_j_point = epoch_j_point_loss / (batch_idx + 1)
        text_1 = '[Epoch {:03d}]'.format(epoch) + '\ttrain: ' + time.strftime('%M: %S', time.gmtime(train_time)) + '\tHR: {:.4f}\tNDCG: {:.4f}\n'.format(np.mean(HR), np.mean(NDCG))
        text_2 = 'Loss: {:.6f}\tPair: {:.4f}\tPoint: {:.4f}\ti_point: {:.4f}\tj_point: {:.4f}\n'.format(e_loss, e_pair, e_point, e_i_point, e_j_point)
        print(text_1[:-1])
        print(text_2[:-1])
        output.write(text_1)
        output.write(text_2)

        if np.mean(HR) > best_hr:
            best_hr, best_ndcg, best_epoch = np.mean(HR), np.mean(NDCG), epoch
    result = 'DF: {} DI: {}. Best epoch {:02d}: HR = {:.4f}, NDCG = {:.4f}\n'.format(DF_layers, DI_layers, best_epoch, best_hr, best_ndcg)
    print(result[:-1])
    output.write(result)
    output.close()


if __name__ == "__main__":
    DF_layers = [512, 256]
    DI_layers = [256, 128, 64]

    run("ml100k", DF_layers, DI_layers, 7, 0.9)
    run("amusic", DF_layers, DI_layers, 7, 0.9)
    run("ml1m", DF_layers, DI_layers, 7, 0.9)
    run("yelp", DF_layers, DI_layers, 7, 0.9)
    