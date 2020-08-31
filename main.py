import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import JNCF
import config
import evaluate
import torch_dataset, torch_functions


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def TOP1(pos, neg):
	diff = neg - pos
	loss = torch.sigmoid(diff) + torch.sigmoid(torch.pow(neg, 2))
	return loss


parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.0001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=256, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=50,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=4, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
args = parser.parse_args()

if torch.cuda.is_available():
	device = torch.device('cuda')
	torch.cuda.manual_seed_all(706)
	FloatTensor = torch.cuda.FloatTensor
else:
	device = torch.device('cpu')
	torch.manual_seed(706)
	FloatTensor = torch.FloatTensor

############################## PREPARE DATASET ##########################
#user_matrix, item_matrix, train_u, train_i, neg_dict, u_cnt = torch_dataset.load_train_ml_1m()
#test_users, test_items = torch_dataset.load_test_ml_1m()

user_matrix, item_matrix, train_u, train_i, neg_dict, u_cnt = torch_dataset.load_train_ml_100k()
test_users, test_items = torch_dataset.load_test_ml_100k()
n_users, n_items = user_matrix.shape[0], user_matrix.shape[1]

user_array = user_matrix.toarray()
item_array = item_matrix.toarray()

user_idxlist, item_idxlist = list(range(n_users)), list(range(n_items))
train_idxlist = np.array(range(len(train_u)))

########################### CREATE MODEL #################################
model = JNCF.JNCF(n_users, n_items, 'concat').to(device)
#loss_function = nn.BCEWithLogitsLoss()
loss_function = TOP1
optimizer = optim.Adam(model.parameters(), lr=args.lr)

########################### TRAINING #####################################
count, best_hr = 0, 0
for epoch in range(args.epochs):
	model.train() # Enable dropout (if have).
	
	if args.num_ng > 0:
		_, neg_items = torch_functions.negative_sampling(train_u, train_i, u_cnt, neg_dict, args.num_ng)
	pos_labels = np.ones(len(train_u))

	# TRAIN
	idxlist = np.array(range(len(train_u)))
	np.random.shuffle(idxlist)
	epoch_loss = .0

	start_time = time.time()
	for r_batch_idx, start_idx in enumerate(range(0, len(idxlist), args.batch_size)):
		end_idx = min(start_idx + args.batch_size, len(idxlist))
		idx = idxlist[start_idx:end_idx]

		u_ids = train_u.take(idx)
		pos_i_ids = pos_items.take(idx)
		neg_i_ids = neg_items.take(idx)
		#labels = new_labels.take(idx)

		users = FloatTensor(user_array.take(u_ids, axis=0))
		pos_items = FloatTensor(item_array.take(pos_i_ids, axis=0))
		neg_items = FloatTensor(item_array.take(neg_i_ids, axis=0))

		optimizer.zero_grad()

		pos_preds = model(users, pos_items)
		neg_preds = model(users, neg_items)

		loss = loss_function(pos_preds, neg_preds)
		epoch_loss += loss.item()
		loss.backward()
		optimizer.step()

	# EVALUATE
	time_E = time.time()
	model.eval()
	HR, NDCG = [], []
	eval_batch_size = 100 * 151
	for e_batch_idx, start_idx in enumerate(range(0, len(test_users), eval_batch_size)):
		end_idx = min(start_idx + eval_batch_size, len(test_users))
		u_ids = test_users[start_idx:end_idx]
		i_ids = test_items[start_idx:end_idx]

		users = FloatTensor(user_array.take(u_ids, axis=0))
		items = FloatTensor(item_array.take(i_ids, axis=0))

		preds = model(users, items).detach().cpu()

		e_batch_size = eval_batch_size // 100
		preds = torch.chunk(preds, e_batch_size)
		chunked_items = torch.chunk(torch.IntTensor(i_ids), e_batch_size)
		for i, pred in enumerate(preds):
			_, indices = torch.topk(pred, 10)
			recommends = torch.take(chunked_items[i], indices).numpy().tolist()

			gt_item = chunked_items[i][0].item()
			HR.append(hit(gt_item, recommends))
			NDCG.append(ndcg(gt_item, recommends))

	train_time = time_E - start_time
	test_time = time.time() - time_E
	print("The time elapse of epoch {:03d}".format(epoch) + " is for train: " + 
			time.strftime("%M: %S", time.gmtime(train_time)) + " // for test: " + time.strftime("%M: %S", time.gmtime(test_time)))
	print("Loss: {:.6f}\tHR: {:.4f}\tNDCG: {:.4f}".format(epoch_loss/(r_batch_idx+1), np.mean(HR), np.mean(NDCG)))

	if np.mean(HR) > best_hr:
		best_hr, best_ndcg, best_epoch = np.mean(HR), np.mean(NDCG), epoch

print("End. Best epoch {:03d}: HR = {:.4f}, NDCG = {:.4f}".format(best_epoch, best_hr, best_ndcg))
