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

import JNCF
import dataset
from functions import *

learning_rate = 0.0001
batch_size = 256
epochs = 100
num_ng = 5

if torch.cuda.is_available():
	device = torch.device('cuda')
	torch.cuda.manual_seed_all(706)
	FloatTensor = torch.cuda.FloatTensor
else:
	device = torch.device('cpu')
	torch.manual_seed(706)
	FloatTensor = torch.FloatTensor

# Datasets
user_matrix, item_matrix, train_u, train_i, train_r, neg_candidates, u_cnt, user_rating_max = dataset.load_train_ml_1m()
test_users, test_items = dataset.load_test_ml_1m()
n_users, n_items = user_matrix.shape[0], user_matrix.shape[1]

user_array = user_matrix.toarray()
item_array = item_matrix.toarray()
user_idxlist, item_idxlist = list(range(n_users)), list(range(n_items))

# Model
model = JNCF.JNCF(n_users, n_items, 'concat').to(device)  # 'multi' or 'concat'
point_loss = explicit_log
pair_loss = TOP1
a = 0.7  # a * pair_loss + (1 - a) * point_loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_hr = 0.0
for epoch in range(epochs):
	# Train
	model.train()  # Enable dropout (if have).

	# Negative Sampling
	train_j_list = []
	for _ in range(0, num_ng):
		train_j = negative_sampling(u_cnt, neg_candidates)
		train_j_list.append(train_j)

	idxlist = np.array(range(len(train_u)))
	np.random.shuffle(idxlist)
	epoch_loss = .0

	start_time = time.time()
	for batch_idx, start_idx in enumerate(range(0, len(idxlist), batch_size)):
		end_idx = min(start_idx + batch_size, len(idxlist))
		idx = idxlist[start_idx:end_idx]

		u_ids = train_u.take(idx)
		pos_i_ids = train_i.take(idx)
		pos_i_ratings = train_r.take(idx)
		rating_max = FloatTensor(user_rating_max.take(u_ids, axis=0))

		users = FloatTensor(user_array.take(u_ids, axis=0))
		pos_items = FloatTensor(item_array.take(pos_i_ids, axis=0))
		labels = FloatTensor(pos_i_ratings).to(device)

		for ng_idx in range(0, num_ng):
			neg_i_ids = train_j_list[ng_idx].take(idx, axis=0)
			neg_items = FloatTensor(item_array.take(neg_i_ids, axis=0))

			optimizer.zero_grad()

			pos_preds = model(users, pos_items)
			neg_preds = model(users, neg_items)

			pair_loss = TOP1(pos_preds, neg_preds)
			point_loss = explicit_log(pos_preds, labels, rating_max)

			loss = a * pair_loss + (1 - a) * point_loss
			epoch_loss += (loss.item() / num_ng)
			loss.backward()
			optimizer.step()
	train_time = time.time() - start_time

	# Evaluate
	model.eval()
	HR, NDCG = [], []
	eval_batch_size = 100 * 151

	time_E = time.time()
	for start_idx in range(0, len(test_users), eval_batch_size):
		end_idx = min(start_idx + eval_batch_size, len(test_users))
		u_ids = test_users[start_idx:end_idx]
		i_ids = test_items[start_idx:end_idx]

		users = FloatTensor(user_array.take(u_ids, axis=0))
		items = FloatTensor(item_array.take(i_ids, axis=0))

		preds = model(users, items).detach().cpu()

		e_batch_size = eval_batch_size // 100  # faster eval
		preds = torch.chunk(preds, e_batch_size)
		chunked_items = torch.chunk(torch.IntTensor(i_ids), e_batch_size)
		for i, pred in enumerate(preds):
			_, indices = torch.topk(pred, 10)
			recommends = torch.take(chunked_items[i], indices).numpy().tolist()

			gt_item = chunked_items[i][0].item()
			HR.append(hit(gt_item, recommends))
			NDCG.append(ndcg(gt_item, recommends))

	eval_time = time.time() - time_E
	print('[Epoch {:03d}]'.format(epoch) + '\ttrain: ' + time.strftime('%M: %S', time.gmtime(train_time)) + '\ttest: ' + time.strftime('%M: %S', time.gmtime(eval_time)))
	print('Loss: {:.6f}\tHR: {:.4f}\tNDCG: {:.4f}'.format(epoch_loss / (batch_idx + 1), np.mean(HR), np.mean(NDCG)))

	if np.mean(HR) > best_hr:
		best_hr, best_ndcg, best_epoch = np.mean(HR), np.mean(NDCG), epoch

print('End. Best epoch {:03d}: HR = {:.4f}, NDCG = {:.4f}'.format(best_epoch, best_hr, best_ndcg))
