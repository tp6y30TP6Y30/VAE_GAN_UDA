import os
import os.path
from os.path import join
import argparse
import torch
import torch.nn as nn
from torch import optim
from dataloader import dataloader
from torch.utils.data import DataLoader
from model import Classifier
import numpy as np
import time
import torchvision.utils as utils
from hw3_eval import evaluate

def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	parser.add_argument('--load', type = int, default = -1)
	parser.add_argument('--epochs', type = int, default = 0)
	parser.add_argument('--img_path', type = str, default = '../../hw3_data/digits/')
	parser.add_argument('--source', type = str)
	parser.add_argument('--target', type = str)
	parser.add_argument('--pred_path', type = str, default = './predict/')
	parser.add_argument('--gt_path', type = str, default = '../../hw3_data/digits/')
	return parser.parse_args()

def load_data(mode, img_path, folder, batch_size, usps):
	loader = dataloader(mode, img_path, folder, usps)
	data = DataLoader(loader, batch_size = batch_size if mode == 'train' else 1, shuffle = (mode == 'train'), num_workers = 6 * (mode == 'train'), pin_memory = True)
	return data

def make_saving_path(pred_path, source, target):
	save_path = join('./models/', source + '_' + target)
	os.makedirs(save_path, exist_ok = True)
	os.makedirs(pred_path, exist_ok = True)
	return save_path

def load_model_optim(load, save_path):
	print('loading Classifier...')
	classifier = Classifier()
	print(classifier)
	total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
	print("Total number of params = ", total_params)
	classifier.cuda().float()
	if load != -1:
		classifier.load_state_dict(torch.load(join(save_path, 'classifier_' + str(load) + '.ckpt')))
	optimizer = optim.Adam(classifier.parameters(), lr = 1e-4, betas = (0.5, 0.9))
	return classifier, optimizer

def save_model(best_loss, avg_loss, classifier, save_path, epoch, source, target):
	os.makedirs(join(save_path, 'train_on_source' if source != target else 'train_on_target'), exist_ok = True)
	torch.save(classifier.state_dict(), join(save_path, 'train_on_source' if source != target else 'train_on_target', 'classifier_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
	if avg_loss < best_loss:
		best_loss = avg_loss
		torch.save(classifier.state_dict(), join(save_path, 'train_on_source' if source != target else 'train_on_target', 'classifier_best.ckpt'), _use_new_zipfile_serialization = False)
	return best_loss

def write_csv(filenames, predictions, pred_path):
	with open(pred_path, "w+") as f:
		first_row = ['image_name', 'label']
		f.write("%s,%s\n" %(first_row[0], first_row[1]))
		for index, filename in enumerate(filenames):
			f.write("%s,%s\n" %(filenames[index], predictions[index]))

def train(args, device):
	torch.multiprocessing.freeze_support()
	from tqdm import tqdm
	if args.mode == 'train':
		batch_size = 128
		train_data = load_data(args.mode, args.img_path, args.source, batch_size, args.source == 'usps' or args.target == 'usps')
		valid_data = load_data('valid', args.img_path, args.target, batch_size,  args.source == 'usps' or args.target == 'usps')
		save_path = make_saving_path(args.pred_path, args.source, args.target)
		classifier, optimizer = load_model_optim(args.load, save_path)

		NLLLoss = nn.CrossEntropyLoss()
		NLLLoss.cuda()
		best_loss = 100.0

		for epoch in range(args.load + 1, args.epochs):
			total_loss = 0
			classifier.train()
			for index, (image, label, filename) in enumerate(tqdm(train_data, ncols = 70)):
				batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
				prediction = classifier(batch_images)
				loss = NLLLoss(prediction, batch_labels)
				total_loss += loss.item()
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			avg_loss = total_loss / len(train_data)
			print('epoch:', epoch)
			print('train_avg_loss: {:.5f}'.format(avg_loss))
			with torch.no_grad():
				total_loss = 0
				filenames, predictions = [], []
				classifier.eval()
				for index, (image, label, filename) in enumerate(tqdm(valid_data, ncols = 70)):
					batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
					filenames.append(filename[0])
					prediction = classifier(batch_images)
					predictions.append(torch.argmax(prediction, dim = 1).item())
					loss = NLLLoss(prediction, batch_labels)
					total_loss += loss.item()
				avg_loss = total_loss / len(valid_data)
				best_loss = save_model(best_loss, avg_loss, classifier, save_path, epoch, args.source, args.target)
				print('valid_avg_loss: {:.5f}'.format(avg_loss))
				write_csv(filenames, predictions, join(args.pred_path, 'test_pred.csv'))
				evaluate(join(args.pred_path, 'test_pred.csv'), join(args.gt_path, args.target, 'test.csv'))
				print()
				print()
	elif args.mode == 'valid':
		batch_size = 1
		valid_data = load_data('valid', args.img_path, args.source, args.target, batch_size, args.source == 'usps' or args.target == 'usps')
		save_path = make_saving_path(args.pred_path, args.source, args.target)
		classifier, optimizer = load_model_optim(args.load, save_path)
		NLLLoss = nn.CrossEntropyLoss()
		NLLLoss.cuda()
		with torch.no_grad():
			total_loss = 0
			filenames, predictions = [], []
			classifier.eval()
			for index, (image, label, filename) in enumerate(tqdm(valid_data, ncols = 70)):
				batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
				filenames.append(filename[0])
				prediction = classifier(batch_images)
				predictions.append(torch.argmax(prediction, dim = 1).item())
				loss = NLLLoss(prediction, batch_labels)
				total_loss += loss.item()
			avg_loss = total_loss / len(valid_data)
			print('valid_avg_loss: {:.5f}'.format(avg_loss))
			write_csv(filenames, predictions, join(args.pred_path, 'test_pred.csv'))
			evaluate(join(args.pred_path, 'test_pred.csv'), join(args.gt_path, args.target, 'test.csv'))
			print()
			print()

if __name__ == '__main__':
	args = _parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train(args, device)