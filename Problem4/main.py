import os
import os.path
from os.path import join
import argparse
import torch
import torch.nn as nn
from torch import optim
from dataloader import dataloader
from torch.utils.data import DataLoader
from model import Extractor, Classifier, Discriminator
import numpy as np
import time
import torchvision.utils as utils
from torch.utils.data import random_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	parser.add_argument('--load', type = int, default = -1)
	parser.add_argument('--load_pretrain', type = int, default = -1)
	parser.add_argument('--epochs', type = int, default = 0)
	parser.add_argument('--img_path', type = str, default = '../hw3_data/digits/')
	parser.add_argument('--pretrain_path', type = str, default = './pretrain_models/')
	parser.add_argument('--source', type = str)
	parser.add_argument('--target', type = str)
	parser.add_argument('--pred_path', type = str, default = './predict/')
	parser.add_argument('--gt_path', type = str, default = '../hw3_data/digits/')
	return parser.parse_args()

def load_data(mode, img_path, folder, batch_size, source, target, split = False):
	loader = dataloader(mode, img_path, folder, source, target)
	if split:
		train_split_len = int(0.7 * len(loader))
		valid_split_len = len(loader) - train_split_len
		print('train_split_len: ', train_split_len)
		print('valid_split_len: ', valid_split_len)
		train_split, valid_split = random_split(loader, [train_split_len, valid_split_len])
		train_split_data = DataLoader(train_split, batch_size = batch_size, shuffle = True, num_workers = 6, pin_memory = True)
		valid_split_data = DataLoader(valid_split, batch_size = 1, shuffle = False, num_workers = 0, pin_memory = True)
		return train_split_data, valid_split_data
	else:
		data = DataLoader(loader, batch_size = batch_size if mode == 'train' else 1, shuffle = (mode == 'train'), num_workers = 6 * (mode == 'train'), pin_memory = True)
		return data

def make_saving_path(args):
	os.makedirs(args.pred_path, exist_ok = True)
	if args.mode == 'train' or args.mode == 'valid' or args.mode == 'tsne':
		train_save_path = join('./models/', args.source + '_' + args.target)
		pretrain_save_path = join('./pretrain_models/', args.source + '_' + args.source)
		os.makedirs(train_save_path, exist_ok = True)
	elif args.mode == 'pretrain':
		train_save_path = None
		pretrain_save_path = join('./pretrain_models/', args.source + '_' + args.target)
		os.makedirs(pretrain_save_path, exist_ok = True)
	print('train_save_path: ', train_save_path)
	print('pretrain_save_path: ', pretrain_save_path)
	return train_save_path, pretrain_save_path

def freeze(model):
	for param in model.parameters():
		param.requires_grad = False

def load_models_optims(load, load_pretrain, train_save_path, pretrain_save_path, mode, source, target):
	if mode == 'pretrain':
		print('loading Extractor...')
		extractor_pretrain = Extractor()
		print(extractor_pretrain)
		total_params = sum(p.numel() for p in extractor_pretrain.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		extractor_pretrain.cuda().float()

		print('loading Classifier...')
		classifier_pretrain = Classifier()
		print(classifier_pretrain)
		total_params = sum(p.numel() for p in classifier_pretrain.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		classifier_pretrain.cuda().float()

		if load_pretrain != -1:
			extractor_pretrain.load_state_dict(torch.load(join(pretrain_save_path, 'extractor_' + str(load_pretrain) + '.ckpt')))
			classifier_pretrain.load_state_dict(torch.load(join(pretrain_save_path, 'classifier_' + str(load_pretrain) + '.ckpt')))
		optimizer = optim.Adam(list(list(extractor_pretrain.parameters()) + list(classifier_pretrain.parameters())), lr = 1e-3, betas = (0.5, 0.9))
		return extractor_pretrain, classifier_pretrain, optimizer

	elif mode == 'train' or mode == 'valid' or mode == 'tsne' or mode == 'test':
		print('loading Extractor_pretrain...')
		extractor_pretrain = Extractor()
		print(extractor_pretrain)
		total_params = sum(p.numel() for p in extractor_pretrain.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		extractor_pretrain.cuda().float()

		print('loading Classifier_pretrain...')
		classifier_pretrain = Classifier()
		print(classifier_pretrain)
		total_params = sum(p.numel() for p in classifier_pretrain.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		classifier_pretrain.cuda().float()

		extractor_pretrain.load_state_dict(torch.load(join(pretrain_save_path, 'extractor_' + str(load_pretrain) + '.ckpt')))
		classifier_pretrain.load_state_dict(torch.load(join(pretrain_save_path, 'classifier_' + str(load_pretrain) + '.ckpt')))
		print('freezing Extractor_pretrain...')
		freeze(extractor_pretrain)
		print('freezing Classifier_pretrain...')
		freeze(classifier_pretrain)

		print('loading Extractor_train...')
		extractor_train = Extractor()
		print(extractor_train)
		total_params = sum(p.numel() for p in extractor_train.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		extractor_train.cuda().float()
		extractor_train.load_state_dict(torch.load(join(pretrain_save_path, 'extractor_' + str(load_pretrain) + '.ckpt')))

		print('loading Discriminator...')
		discriminator = Discriminator()
		print(discriminator)
		total_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		discriminator.cuda().float()

		if load != -1:
			extractor_train.load_state_dict(torch.load(join(train_save_path, 'extractor_' + str(load) + '.ckpt')))
			discriminator.load_state_dict(torch.load(join(train_save_path, 'discriminator_' + str(load) + '.ckpt')))
		lr = 1e-3
		if source == 'mnistm' and target == 'svhn':
			# 1e-3 for mnistm -> svhn
			lr = 1e-3
		elif source == 'svhn' and target == 'usps':
			# 1e-5 for svhn -> usps
			lr = 1e-5
		elif source == 'usps' and target == 'mnistm':
			# 1e-4 for usps -> mnistm
			lr = 1e-4
		optim_Ext = optim.Adam(extractor_train.parameters(), lr = lr, betas = (0.5, 0.9))
		optim_Dsc = optim.Adam(discriminator.parameters(), lr = lr, betas = (0.5, 0.9))
		return extractor_pretrain, classifier_pretrain, extractor_train, discriminator, optim_Ext, optim_Dsc

def save_model(extractor, discriminator, classifier, save_path, epoch, mode):
	if mode == 'pretrain':
		torch.save(extractor.state_dict(), join(save_path, 'extractor_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
		torch.save(classifier.state_dict(), join(save_path, 'classifier_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
	elif mode == 'train':
		torch.save(extractor.state_dict(), join(save_path, 'extractor_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
		torch.save(discriminator.state_dict(), join(save_path, 'discriminator_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)

def write_csv(filenames, predictions, pred_path):
	with open(pred_path, "w+") as f:
		first_row = ['image_name', 'label']
		f.write("%s,%s\n" %(first_row[0], first_row[1]))
		for index, filename in enumerate(filenames):
			f.write("%s,%s\n" %(filenames[index], predictions[index]))

def train(args, device):
	torch.multiprocessing.freeze_support()
	from tqdm import tqdm
	from hw3_eval import evaluate
	if args.mode == 'train':
		batch_size = {'usps' : 64, 'mnistm' : 256, 'svhn' : 256}
		source_train_data, source_valid_data = load_data(args.mode, args.img_path, args.source, batch_size[args.source], args.source, args.target, True)
		target_train_data, target_valid_data = load_data(args.mode, args.img_path, args.target, batch_size[args.target], args.source, args.target, True)
		train_save_path, pretrain_save_path = make_saving_path(args)
		extractor_pretrain, classifier_pretrain, extractor_train, discriminator, optim_Ext, optim_Dsc = load_models_optims(args.load, args.load_pretrain, train_save_path, pretrain_save_path, args.mode, args.source, args.target)

		NLLLoss = nn.NLLLoss()
		NLLLoss.cuda()
		for epoch in range(args.load + 1, args.epochs):
			total_extractor_loss = total_source_loss = total_target_loss = 0
			extractor_pretrain.eval()
			extractor_train.train()
			discriminator.train()
			for index, ((source_image, source_label, source_filename), (target_image, target_label, target_filename)) in enumerate(tqdm(zip(source_train_data, target_train_data), total = min(len(source_train_data), len(target_train_data)), ncols = 70, desc = '[Training]')):
				batch_source_images, batch_source_labels, batch_source_filenames = source_image.to(device), source_label.to(device), source_filename
				batch_target_images, batch_target_labels, batch_target_filenames = target_image.to(device), target_label.to(device), target_filename
				batch_source_labels = torch.zeros(batch_source_images.shape[0], dtype = torch.long).cuda()
				batch_target_labels = torch.ones(batch_target_images.shape[0], dtype = torch.long).cuda()
				
				extractor_train.zero_grad()
				optim_Ext.zero_grad()
				target_features = extractor_train(batch_target_images)
				target_predict = discriminator(target_features)
				extractor_loss = NLLLoss(target_predict, batch_target_labels - 1)
				total_extractor_loss += extractor_loss.item()
				extractor_loss.backward()
				optim_Ext.step()

				discriminator.zero_grad()
				optim_Dsc.zero_grad()
				target_features = extractor_train(batch_target_images)
				source_features = extractor_pretrain(batch_source_images)
				source_predict, target_predict = discriminator(source_features), discriminator(target_features)
				source_loss, target_loss = NLLLoss(source_predict, batch_source_labels), NLLLoss(target_predict, batch_target_labels)
				total_source_loss += source_loss.item()
				total_target_loss += target_loss.item()
				dsc_loss = source_loss + target_loss
				dsc_loss.backward()
				optim_Dsc.step()

			avg_extractor_loss = total_extractor_loss / min(len(source_train_data), len(target_train_data))
			avg_source_loss = total_source_loss / min(len(source_train_data), len(target_train_data))
			avg_target_loss = total_target_loss / min(len(source_train_data), len(target_train_data))
			print('epoch:', epoch)
			print('train_avg_extractor_loss: {:.5f} train_avg_source_loss: {:.5f} train_avg_target_loss: {:.5f}'.format(avg_extractor_loss, avg_source_loss, avg_target_loss))
			
			with torch.no_grad():
				extractor_train.eval()
				discriminator.eval()
				total_loss = 0
				predictions, filenames = [], []
				for index, (image, label, filename) in enumerate(tqdm(target_valid_data, ncols = 70, desc = '[Valid Target]')):
					batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
					filenames.append(batch_filenames[0])
					target_features = extractor_train(batch_images)
					target_predict = classifier_pretrain(target_features)
					predictions.append(torch.argmax(target_predict, dim = 1).item())
					target_loss = NLLLoss(target_predict, batch_labels)
					total_loss += target_loss.item()
				avg_target_loss = total_loss / len(target_valid_data)
				print('valid_avg_extractor_loss: {:.5f}'.format(avg_target_loss))
				write_csv(filenames, predictions, join(args.pred_path, 'test_pred.csv'))
				evaluate(join(args.pred_path, 'test_pred.csv'), join(args.gt_path, args.target, 'train.csv'), True)
				print()
				print()
				save_model(extractor_train, discriminator, classifier_pretrain, train_save_path, epoch, args.mode)

	elif args.mode == 'valid':
		batch_size = 1
		target_valid_data = load_data('valid', args.img_path, args.target, batch_size, args.source, args.target)
		train_save_path, pretrain_save_path = make_saving_path(args)
		extractor_pretrain, classifier_pretrain, extractor_train, discriminator, optim_Ext, optim_Dsc = load_models_optims(args.load, args.load_pretrain, train_save_path, pretrain_save_path, args.mode, args.source, args.target)

		CELoss = nn.CrossEntropyLoss()
		CELoss.cuda()
		with torch.no_grad():
			extractor_train.eval()
			discriminator.eval()
			total_loss = 0
			predictions, filenames = [], []
			for index, (image, label, filename) in enumerate(tqdm(target_valid_data, ncols = 70, desc = '[Valid Target]')):
				batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
				filenames.append(batch_filenames[0])
				target_features = extractor_train(batch_images)
				target_predict = classifier_pretrain(target_features)
				predictions.append(torch.argmax(target_predict, dim = 1).item())
				target_loss = CELoss(target_predict, batch_labels)
				total_loss += target_loss.item()
			avg_target_loss = total_loss / len(target_valid_data)
			print('valid_avg_extractor_loss: {:.5f}'.format(avg_target_loss))
			write_csv(filenames, predictions, join(args.pred_path, 'test_pred.csv'))
			evaluate(join(args.pred_path, 'test_pred.csv'), join(args.gt_path, args.target, 'test.csv'))
			print()
			print()

def pretrain(args, device):
	torch.multiprocessing.freeze_support()
	from tqdm import tqdm
	batch_size = 128
	train_data = load_data('train', args.img_path, args.source, batch_size, args.source, args.target)
	valid_data = load_data('valid', args.img_path, args.target, 1,  args.source, args.target)
	train_save_path, pretrain_save_path = make_saving_path(args)
	extractor, classifier, optimizer = load_models_optims(args.load, args.load_pretrain, train_save_path, pretrain_save_path, args.mode, args.source, args.target)

	CELoss = nn.CrossEntropyLoss()
	CELoss.cuda()

	for epoch in range(args.load_pretrain + 1, args.epochs):
		total_loss = 0
		extractor.train()
		classifier.train()
		for index, (image, label, filename) in enumerate(tqdm(train_data, ncols = 70)):
			batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
			features = extractor(batch_images)
			prediction = classifier(features)
			loss = CELoss(prediction, batch_labels)
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
			extractor.eval()
			classifier.eval()
			for index, (image, label, filename) in enumerate(tqdm(valid_data, ncols = 70)):
				batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
				filenames.append(filename[0])
				features = extractor(batch_images)
				prediction = classifier(features)
				predictions.append(torch.argmax(prediction, dim = 1).item())
				loss = CELoss(prediction, batch_labels)
				total_loss += loss.item()
			avg_loss = total_loss / len(valid_data)
			save_model(extractor, None, classifier, pretrain_save_path, epoch, args.mode)
			print('valid_avg_loss: {:.5f}'.format(avg_loss))
			write_csv(filenames, predictions, join(args.pred_path, 'test_pred.csv'))
			evaluate(join(args.pred_path, 'test_pred.csv'), join(args.gt_path, args.target, 'test.csv'))
			print()
			print()

def show_tsne(source_features, source_class_labels, source_domain_labels, target_features, target_class_labels, target_domain_labels):
	source_features, source_class_labels, source_domain_labels = np.array(source_features), np.array(source_class_labels), np.array(source_domain_labels)
	target_features, target_class_labels, target_domain_labels = np.array(target_features), np.array(target_class_labels), np.array(target_domain_labels)
	# class
	features = np.concatenate((source_features, target_features), axis = 0)
	labels = np.concatenate((source_class_labels, target_class_labels), axis = 0)
	tsne_features_fit = TSNE(n_components = 2, n_jobs = -1).fit_transform(features)
	plt.scatter(tsne_features_fit[:, 0], tsne_features_fit[:, 1], c = labels, cmap = plt.cm.jet, s = 10)
	plt.show()

	# domain
	features = np.concatenate((source_features, target_features), axis = 0)
	labels = np.concatenate((source_domain_labels, target_domain_labels), axis = 0)
	# tsne_features_fit = TSNE(n_components = 2, n_jobs = -1).fit_transform(features)
	plt.scatter(tsne_features_fit[:, 0], tsne_features_fit[:, 1], c = labels, cmap = plt.cm.jet, s = 10)
	plt.show()

def run_tsne(args, device):
	from tqdm import tqdm
	batch_size = 1
	valid_source_data = load_data('valid', args.img_path, args.source, batch_size, args.source, args.target)
	valid_target_data = load_data('valid', args.img_path, args.target, batch_size, args.source, args.target)
	train_save_path, pretrain_save_path = make_saving_path(args)
	extractor_pretrain, classifier_pretrain, extractor_train, discriminator, optim_Ext, optim_Dsc = load_models_optims(args.load, args.load_pretrain, train_save_path, pretrain_save_path, args.mode, args.source, args.target)

	with torch.no_grad():
		extractor_pretrain.eval()
		source_features, source_class_labels, source_domain_labels = [], [], []
		for index, (image, label, filename) in enumerate(tqdm(valid_source_data, ncols = 70, desc = 'test_on_source')):
			batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
			features = extractor_pretrain(batch_images)
			source_features.append(features.squeeze().cpu().numpy())
			source_class_labels.append(batch_labels.cpu())
			source_domain_labels.append(0)

	with torch.no_grad():
		extractor_train.eval()
		target_features, target_class_labels, target_domain_labels = [], [], []
		for index, (image, label, filename) in enumerate(tqdm(valid_target_data, ncols = 70, desc = 'test_on_target')):
			batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
			features = extractor_train(batch_images)
			target_features.append(features.squeeze().cpu().numpy())
			target_class_labels.append(batch_labels.cpu())
			target_domain_labels.append(1)
	show_tsne(source_features, source_class_labels, source_domain_labels, target_features, target_class_labels, target_domain_labels)

def test(args, device):
	start_time = time.time()
	test_target_data = load_data('test', args.img_path, None, 1, None, args.target)
	_, classifier_pretrain, extractor_train, _, _, _ = load_models_optims(123, 123, join('./Problem4/test_models/', args.target), join('./Problem4/test_models/', args.target, 'pretrain_models/'), args.mode, None, args.target)
	with torch.no_grad():
		predictions, filenames = [], []
		extractor_train.eval()
		classifier_pretrain.eval()
		for index, (image, filename) in enumerate(test_target_data):
			batch_images, batch_filenames = image.to(device), filename
			filenames.append(batch_filenames[0])
			features = extractor_train(batch_images)
			prediction = classifier_pretrain(features)
			predictions.append(torch.argmax(prediction, dim = 1).item())
			if index % 100 == 0:
				print('progress: {:.1f}%'.format(index / len(test_target_data) * 100))
		print('progress: 100.0%')
		write_csv(filenames, predictions, args.pred_path)
	end_time = time.time()
	print('Finished in {:.2f}s'.format(end_time - start_time))
	
def cal_ac(args):
	from hw3_eval import evaluate
	evaluate(args.pred_path, args.gt_path)

if __name__ == '__main__':
	args = _parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if args.mode == 'test':
		test(args, device)
	elif args.mode == 'cal_ac':
		cal_ac(args)
	elif args.mode == 'pretrain':
		pretrain(args, device)
	elif args.mode == 'tsne':
		run_tsne(args, device)
	else:
		train(args, device)