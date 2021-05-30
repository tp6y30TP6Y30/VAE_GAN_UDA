import os
import os.path
from os.path import join
import argparse
import torch
import torch.nn as nn
from torch import optim
from dataloader import dataloader
from torch.utils.data import DataLoader
from model import Extractor, Predictor, Classifier, Gradient_Reverse_Layer
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
	parser.add_argument('--epochs', type = int, default = 0)
	parser.add_argument('--img_path', type = str, default = '../hw3_data/digits/')
	parser.add_argument('--source', type = str)
	parser.add_argument('--target', type = str)
	parser.add_argument('--per_test', action = 'store_true')
	parser.add_argument('--pred_path', type = str, default = './predict/')
	parser.add_argument('--gt_path', type = str, default = '../hw3_data/digits/')
	return parser.parse_args()

def load_data(mode, img_path, folder, batch_size, usps, split = False):
	loader = dataloader(mode, img_path, folder, usps)
	if split:
		train_split_len = int(0.7 * len(loader))
		valid_split_len = len(loader) - train_split_len
		print('train_split_len: ', train_split_len)
		print('valid_split_len: ', valid_split_len)
		train_split, valid_split = random_split(loader, [train_split_len, valid_split_len])
		train_split_data = DataLoader(train_split, batch_size = batch_size if mode == 'train' else 1, shuffle = (mode == 'train'), num_workers = 4 * (mode == 'train'), pin_memory = True)
		valid_split_data = DataLoader(valid_split, batch_size = 1, shuffle = False, num_workers = 0, pin_memory = True)
		return train_split_data, valid_split_data
	else:
		data = DataLoader(loader, batch_size = batch_size if mode == 'train' else 1, shuffle = (mode == 'train'), num_workers = 4 * (mode == 'train'), pin_memory = True)
		return data

def make_saving_path(pred_path, source, target):
	save_path = join('./models/', source + '_' + target)
	os.makedirs(save_path, exist_ok = True)
	os.makedirs(pred_path, exist_ok = True)
	return save_path

def load_models_optims(load, save_path):
	print('loading Extractor...')
	extractor = Extractor()
	print(extractor)
	total_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
	print("Total number of params = ", total_params)
	extractor.cuda().float()

	print('loading Predictor...')
	predictor = Predictor()
	print(predictor)
	total_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
	print("Total number of params = ", total_params)
	predictor.cuda().float()

	print('loading Classifier...')
	classifier = Classifier()
	print(classifier)
	total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
	print("Total number of params = ", total_params)
	classifier.cuda().float()

	if load != -1:
		extractor.load_state_dict(torch.load(join(save_path, 'extractor_' + str(load) + '.ckpt')))
		predictor.load_state_dict(torch.load(join(save_path, 'predictor_' + str(load) + '.ckpt')))
		classifier.load_state_dict(torch.load(join(save_path, 'classifier_' + str(load) + '.ckpt')))

	optimizer = optim.Adam(list(list(extractor.parameters()) + list(predictor.parameters()) + list(classifier.parameters())), lr = 1e-3, betas = (0.5, 0.9))
	return extractor, predictor, classifier, optimizer

def save_model(extractor, predictor, classifier, save_path, epoch):
	torch.save(extractor.state_dict(), join(save_path, 'extractor_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
	torch.save(predictor.state_dict(), join(save_path, 'predictor_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
	torch.save(classifier.state_dict(), join(save_path, 'classifier_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)

def write_csv(filenames, predictions, pred_path):
	with open(pred_path, "w+") as f:
		first_row = ['image_name', 'label']
		f.write("%s,%s\n" %(first_row[0], first_row[1]))
		for index, filename in enumerate(filenames):
			f.write("%s,%s\n" %(filenames[index], predictions[index]))

def get_Lambda(index, start_step, total_step):
	p = float(index + start_step) / total_step
	lambda_term = 2. / (1. + np.exp(-10 * p)) - 1
	return lambda_term

def train(args, device):
	torch.multiprocessing.freeze_support()
	from tqdm import tqdm
	from hw3_eval import evaluate
	if args.mode == 'train':
		batch_size = {'usps' : 64, 'mnistm' : 256, 'svhn' : 256}
		train_source_data = load_data(args.mode, args.img_path, args.source, batch_size[args.source], args.source == 'usps' or args.target == 'usps')
		train_target_split, valid_target_split = load_data(args.mode, args.img_path, args.target, batch_size[args.target], args.source == 'usps' or args.target == 'usps', True)

		valid_source_data = load_data('valid', args.img_path, args.source, batch_size[args.source], args.source == 'usps' or args.target == 'usps')
		valid_target_data = load_data('valid', args.img_path, args.target, batch_size[args.target], args.source == 'usps' or args.target == 'usps')
		save_path = make_saving_path(args.pred_path, args.source, args.target)
		extractor, predictor, classifier, optimizer = load_models_optims(args.load, save_path)
		NLLLoss = nn.NLLLoss()
		NLLLoss.cuda()

		best_loss = 100.0

		for epoch in range(args.load + 1, args.epochs):
			total_extractor_loss = total_predictor_loss = total_classifier_loss = 0
			extractor.train()
			predictor.train()
			classifier.train()
			start_step = epoch * min(len(train_source_data), len(train_target_split))
			total_step = args.epochs * min(len(train_source_data), len(train_target_split))
			for index, ((source_image, source_label, source_filename), (target_image, target_label, target_filename)) in enumerate(tqdm(zip(train_source_data, train_target_split), total = min(len(train_source_data), len(train_target_split)), ncols = 70, desc = 'Training')):
				batch_source_images, batch_source_labels, batch_source_filenames = source_image.to(device), source_label.to(device), source_filename
				batch_target_images, batch_target_labels, batch_target_filenames = target_image.to(device), target_label.to(device), target_filename
				source_features = extractor(batch_source_images)
				target_features = extractor(batch_target_images)
				prediction = predictor(source_features)
				predict_loss = NLLLoss(prediction, batch_source_labels)
				total_extractor_loss += predict_loss.item()
				total_predictor_loss += predict_loss.item()

				batch_source_labels = torch.zeros(batch_source_labels.shape[0], dtype = torch.long).cuda()
				batch_target_labels = torch.ones(batch_target_labels.shape[0], dtype = torch.long).cuda()
				classification_source = classifier(source_features, get_Lambda(index, start_step, total_step))
				classification_target = classifier(target_features, get_Lambda(index, start_step, total_step))
				source_loss = NLLLoss(classification_source, batch_source_labels)
				target_loss = NLLLoss(classification_target, batch_target_labels)
				total_extractor_loss += (source_loss + target_loss).item()
				total_classifier_loss += (source_loss + target_loss).item()
				total_loss = (source_loss + target_loss) + predict_loss
				optimizer.zero_grad()
				total_loss.backward()
				optimizer.step()

			avg_extractor_loss = total_extractor_loss / min(len(train_source_data), len(train_target_split))
			avg_predictor_loss = total_predictor_loss / min(len(train_source_data), len(train_target_split))
			avg_classifier_loss = total_classifier_loss / min(len(train_source_data), len(train_target_split))
			print('epoch: ', epoch)
			print('[Train] avg_extractor_loss: {:.5f} avg_predictor_loss: {:.5f} avg_classifier_loss: {:.5f}'.format(avg_extractor_loss, avg_predictor_loss, avg_classifier_loss))
			print()

			with torch.no_grad():
				predictions, filenames = [], []
				total_extractor_loss = total_predictor_loss = 0
				extractor.eval()
				predictor.eval()
				for index, (image, label, filename) in enumerate(tqdm(valid_source_data, ncols = 70, desc = 'test_on_source')):
					batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
					filenames.append(batch_filenames[0])
					features = extractor(batch_images)
					prediction = predictor(features)
					predictions.append(torch.argmax(prediction, dim = 1).item())
					predict_loss = NLLLoss(prediction, batch_labels)
					total_extractor_loss += predict_loss.item()
					total_predictor_loss += predict_loss.item()

				avg_extractor_loss = total_extractor_loss / len(valid_source_data)
				avg_predictor_loss = total_predictor_loss / len(valid_source_data)
				print('[Test_source] avg_extractor_loss: {:.5f} avg_predictor_loss: {:.5f}'.format(avg_extractor_loss, avg_predictor_loss))
				write_csv(filenames, predictions, join(args.pred_path, 'test_pred.csv'))
				evaluate(join(args.pred_path, 'test_pred.csv'), join(args.gt_path, args.source, 'test.csv'))
				print()

			with torch.no_grad():
				predictions, filenames = [], []
				total_extractor_loss = total_predictor_loss = 0
				extractor.eval()
				predictor.eval()
				for index, (image, label, filename) in enumerate(tqdm(valid_target_split, ncols = 70, desc = 'valid_on_target')):
					batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
					filenames.append(batch_filenames[0])
					features = extractor(batch_images)
					prediction = predictor(features)
					predictions.append(torch.argmax(prediction, dim = 1).item())
					predict_loss = NLLLoss(prediction, batch_labels)
					total_extractor_loss += predict_loss.item()
					total_predictor_loss += predict_loss.item()

				avg_extractor_loss = total_extractor_loss / len(valid_target_split)
				avg_predictor_loss = total_predictor_loss / len(valid_target_split)
				print('[Valid_target] avg_extractor_loss: {:.5f} avg_predictor_loss: {:.5f}'.format(avg_extractor_loss, avg_predictor_loss))
				write_csv(filenames, predictions, join(args.pred_path, 'test_pred.csv'))
				evaluate(join(args.pred_path, 'test_pred.csv'), join(args.gt_path, args.target, 'train.csv'), True)
				print()
			if args.per_test:
				with torch.no_grad():
					predictions, filenames = [], []
					total_extractor_loss = total_predictor_loss = 0
					extractor.eval()
					predictor.eval()
					for index, (image, label, filename) in enumerate(tqdm(valid_target_data, ncols = 70, desc = 'test_on_target')):
						batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
						filenames.append(batch_filenames[0])
						features = extractor(batch_images)
						prediction = predictor(features)
						predictions.append(torch.argmax(prediction, dim = 1).item())
						predict_loss = NLLLoss(prediction, batch_labels)
						total_extractor_loss += predict_loss.item()
						total_predictor_loss += predict_loss.item()

					avg_extractor_loss = total_extractor_loss / len(valid_target_data)
					avg_predictor_loss = total_predictor_loss / len(valid_target_data)
					print('[Test_target] avg_extractor_loss: {:.5f} avg_predictor_loss: {:.5f}'.format(avg_extractor_loss, avg_predictor_loss))
					write_csv(filenames, predictions, join(args.pred_path, 'test_pred.csv'))
					evaluate(join(args.pred_path, 'test_pred.csv'), join(args.gt_path, args.target, 'test.csv'))
					print()
			
			print()
			save_model(extractor, predictor, classifier, save_path, epoch)

	elif args.mode == 'valid':
		batch_size = 1
		valid_source_data = load_data('valid', args.img_path, args.source, batch_size, args.source == 'usps' or args.target == 'usps')
		valid_target_data = load_data('valid', args.img_path, args.target, batch_size, args.source == 'usps' or args.target == 'usps')
		save_path = make_saving_path(args.pred_path, args.source, args.target)
		extractor, predictor, classifier, optimizer = load_models_optims(args.load, save_path)
		NLLLoss = nn.CrossEntropyLoss()
		NLLLoss.cuda()

		with torch.no_grad():
			predictions, filenames = [], []
			total_extractor_loss = total_predictor_loss = 0
			extractor.eval()
			predictor.eval()
			for index, (image, label, filename) in enumerate(tqdm(valid_source_data, ncols = 70, desc = 'test_on_source')):
				batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
				filenames.append(batch_filenames[0])
				features = extractor(batch_images)
				prediction = predictor(features)
				predictions.append(torch.argmax(prediction, dim = 1).item())
				predict_loss = NLLLoss(prediction, batch_labels)
				total_extractor_loss += predict_loss.item()
				total_predictor_loss += predict_loss.item()

			avg_extractor_loss = total_extractor_loss / len(valid_source_data)
			avg_predictor_loss = total_predictor_loss / len(valid_source_data)
			print('[Test_source] avg_extractor_loss: {:.5f} avg_predictor_loss: {:.5f}'.format(avg_extractor_loss, avg_predictor_loss))
			write_csv(filenames, predictions, join(args.pred_path, 'test_pred.csv'))
			evaluate(join(args.pred_path, 'test_pred.csv'), join(args.gt_path, args.source, 'test.csv'))
			print()

		with torch.no_grad():
			predictions, filenames = [], []
			total_extractor_loss = total_predictor_loss = 0
			extractor.eval()
			predictor.eval()
			for index, (image, label, filename) in enumerate(tqdm(valid_target_data, ncols = 70, desc = 'test_on_target')):
				batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
				filenames.append(batch_filenames[0])
				features = extractor(batch_images)
				prediction = predictor(features)
				predictions.append(torch.argmax(prediction, dim = 1).item())
				predict_loss = NLLLoss(prediction, batch_labels)
				total_extractor_loss += predict_loss.item()
				total_predictor_loss += predict_loss.item()

			avg_extractor_loss = total_extractor_loss / len(valid_target_data)
			avg_predictor_loss = total_predictor_loss / len(valid_target_data)
			print('[Test_target] avg_extractor_loss: {:.5f} avg_predictor_loss: {:.5f}'.format(avg_extractor_loss, avg_predictor_loss))
			write_csv(filenames, predictions, join(args.pred_path, 'test_pred.csv'))
			evaluate(join(args.pred_path, 'test_pred.csv'), join(args.gt_path, args.target, 'test.csv'))
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
	valid_source_data = load_data('valid', args.img_path, args.source, batch_size, args.source == 'usps' or args.target == 'usps')
	valid_target_data = load_data('valid', args.img_path, args.target, batch_size, args.source == 'usps' or args.target == 'usps')
	save_path = make_saving_path(args.pred_path, args.source, args.target)
	extractor, predictor, classifier, optimizer = load_models_optims(args.load, save_path)

	with torch.no_grad():
		extractor.eval()
		source_features, source_class_labels, source_domain_labels = [], [], []
		for index, (image, label, filename) in enumerate(tqdm(valid_source_data, ncols = 70, desc = 'test_on_source')):
			batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
			features = extractor(batch_images)
			source_features.append(features.squeeze().cpu().numpy())
			source_class_labels.append(batch_labels.cpu())
			source_domain_labels.append(0)

	with torch.no_grad():
		extractor.eval()
		target_features, target_class_labels, target_domain_labels = [], [], []
		for index, (image, label, filename) in enumerate(tqdm(valid_target_data, ncols = 70, desc = 'test_on_target')):
			batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
			features = extractor(batch_images)
			target_features.append(features.squeeze().cpu().numpy())
			target_class_labels.append(batch_labels.cpu())
			target_domain_labels.append(1)
	show_tsne(source_features, source_class_labels, source_domain_labels, target_features, target_class_labels, target_domain_labels)

def test(args, device):
	start_time = time.time()
	test_target_data = load_data('test', args.img_path, None, 1, args.target == 'usps' or args.target == 'mnistm')
	extractor, predictor, _, _ = load_models_optims(123, join('./Problem3/test_models/', args.target))
	with torch.no_grad():
		predictions, filenames = [], []
		extractor.eval()
		predictor.eval()
		for index, (image, filename) in enumerate(test_target_data):
			batch_images, batch_filenames = image.to(device), filename
			filenames.append(batch_filenames[0])
			features = extractor(batch_images)
			prediction = predictor(features)
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
	elif args.mode == 'tsne':
		run_tsne(args, device)
	else:
		train(args, device)