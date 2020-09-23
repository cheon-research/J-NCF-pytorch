import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_train_data(dataset):
    # modified for J-NCF, explicit feedback
    if dataset == 'ml-1m':
        data = pd.read_csv('dataset/ml-1m.train.rating', sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2],
                        dtype={0: np.int16, 1: np.int16, 2:np.int8})
    elif dataset == 'ml-100k':
        data = pd.read_csv('dataset/ml-100k.train.rating.csv', sep=',', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2],
                        dtype={0: np.int16, 1: np.int16, 2:np.int8})

    n_user, n_item = data['user'].max() + 1, data['item'].max() + 1
    user_count = data.groupby('user').count()['item'].values.reshape(-1) # count interacted items for each user
    user_rating_max = data.groupby('user').max()['rating'].values.reshape(-1)

    item_sort_data = data.sort_values(by=['item', 'user'])

    user_rows, user_cols, user_ratings = data['user'], data['item'], data['rating']
    item_rows, item_cols, item_ratings = item_sort_data['item'], item_sort_data['user'], item_sort_data['rating']
    users, items, ratings = data['user'].values, data['item'].values, data['rating'].values

    # user-item mat for user_embedding / item-user mat for item_embedding
    user_item_matrix = sp.csr_matrix((user_ratings, (user_rows, user_cols)), dtype=np.int8, shape=(n_user, n_item))
    print('user_item_matrix', user_item_matrix.shape)
    item_user_matrix = sp.csr_matrix((item_ratings, (item_rows, item_cols)), dtype=np.int8, shape=(n_item, n_user))
    print('item_user_matrix', item_user_matrix.shape)

    # Negative sample candidates
    neg_candidates = dict()
    item_list = np.array(range(0, n_item))
    for u in range(0, n_user):
        pos_items = data.loc[data['user'] == u]['item'].values # select 'item's that 'user' == u
        candidates = np.setdiff1d(item_list, pos_items) # items - pos_items
        neg_candidates[u] = candidates

    return user_item_matrix, item_user_matrix, users, items, ratings, neg_candidates, user_count, user_rating_max


def load_test_ml_1m():
    test_users, test_items = [], []
    with open('dataset/ml-1m.test.negative', 'r') as fd:
        line = fd.readline()
        while line is not None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_users.append(u)
            test_items.append(eval(arr[0])[1])
            for i in arr[1:]:
                test_users.append(u)
                test_items.append(int(i))
            line = fd.readline()
    return np.array(test_users), np.array(test_items)


def load_test_ml_100k():
    test_users, test_items = [], []
    data = pd.read_csv('dataset/ml-100k.test.negative.csv', sep=',', header=None, names=['user', 'item'], 
        usecols=[0, 1], dtype={0: np.int16, 1: np.int16})

    n_user, n_item = data['user'].max() + 1, data['item'].max() + 1
    users, items = data['user'].values, data['item'].values

    return np.array(users), np.array(items)