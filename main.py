import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import JCNF
import config
import evaluate
import data_utils


parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
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
	default=30,  
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
train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()
#train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all_100k()

# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, args.num_ng, True)
test_dataset = data_utils.NCFData(
		test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = data.DataLoader(test_dataset,
		batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

########################### CREATE MODEL #################################


model = JCNF(user_num, item_num, 'concat').to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


########################### TRAINING #####################################
count, best_hr = 0, 0
for epoch in range(args.epochs):
	model.train() # Enable dropout (if have).
	start_time = time.time()
	train_loader.dataset.ng_sample()
	epoch_loss = 0
	time_t = time.time()
	end_tt = time.time()
	for batch_idx, (user, item, label) in enumerate(train_loader):
		user = user.cuda()
		item = item.cuda()
		label = label.float().cuda()

		model.zero_grad()
		prediction = model(user, item)

		loss = loss_function(prediction, label)
		loss.backward()

		optimizer.step()
		epoch_loss += loss.item()

	model.eval()
	HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)

	elapsed_time = time.time() - start_time
	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
	print("Loss: {:.6f}\tHR: {:.4f}\tNDCG: {:.4f}".format(epoch_loss/(batch_idx+1), np.mean(HR), np.mean(NDCG)))

	if HR > best_hr:
		best_hr, best_ndcg, best_epoch = HR, NDCG, epoch

print("End. Best epoch {:03d}: HR = {:.4f}, NDCG = {:.4f}".format(
									best_epoch, best_hr, best_ndcg))
