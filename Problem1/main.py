import os
import os.path
from os.path import join
import argparse
import torch
import torch.nn as nn
from torch import optim
from dataloader import dataloader
from torch.utils.data import DataLoader
from model import Encoder, Decoder
import numpy as np
import matplotlib.pyplot as plt
import time
import torchvision.utils as utils
from sklearn.manifold import TSNE

def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	parser.add_argument('--load', type = int, default = -1)
	parser.add_argument('--epochs', type = int, default = 0)
	parser.add_argument('--train_img_path', type = str, default = '../hw3_data/face/train/')
	parser.add_argument('--valid_img_path', type = str, default = './valid/')
	parser.add_argument('--train_label_path', type = str, default = '../hw3_data/face/train.csv')
	parser.add_argument('--valid_label_path', type = str, default = '../hw3_data/face/test.csv')
	parser.add_argument('--need_attribute', action = 'store_true')
	parser.add_argument('--attribute', type = str, default = 'Male')
	parser.add_argument('--pred_path', type = str, default = './predict/')
	return parser.parse_args()

def load_data(mode, img_path, label_path, need_attribute, attribute):
	loader = dataloader(mode, img_path, label_path, need_attribute, attribute)
	data = DataLoader(loader, batch_size = 32 if mode == 'train' else 1, shuffle = (mode == 'train'), num_workers = 6 * (mode == 'train'), pin_memory = True)
	return data

def make_saving_path(pred_path):
	save_path = './models/'
	os.makedirs(save_path, exist_ok = True)
	os.makedirs(pred_path, exist_ok = True)
	return save_path

def load_models_optims(load, save_path):
	print('loading encoder...')
	encoder = Encoder()
	print(encoder)
	total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
	print("Total number of params = ", total_params)
	encoder.cuda().float()
	print('loading decoder...')
	decoder = Decoder()
	print(decoder)
	total_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
	print("Total number of params = ", total_params)
	decoder.cuda().float()
	if load != -1:
		encoder.load_state_dict(torch.load(join(save_path, 'encoder_' + str(load) + '.ckpt')))
		decoder.load_state_dict(torch.load(join(save_path, 'decoder_' + str(load) + '.ckpt')))
	optimizer = optim.Adam(list(list(encoder.parameters()) + list(decoder.parameters())), lr = 1e-4, betas = (0.5, 0.9))
	return encoder, decoder, optimizer

def sample_LatentSpace(batch_means, batch_logvar, mode):
	if mode == 'random' or mode == 'test':
		noises = torch.randn(32, 128).cuda()
	else:
		std = batch_logvar.mul(0.5).exp_()
		init = torch.randn(batch_means.shape).cuda()
		noises = init * std + batch_means
	return noises

def get_KL_Lambda(mse_loss):
	KL_Lambda = 1e-5
	if mse_loss < 0.015:
		KL_Lambda /= (1.0 * (mse_loss ** 2))
	return KL_Lambda

def KL_loss(mean, logvar):
	return torch.mean(-0.5 * (1 + logvar - mean ** 2 - torch.exp(logvar)))

def save_fake_images(fake_images, batch_filenames, pred_path, epoch):
	save_path = join(pred_path, ('epoch_' + str(epoch)))
	os.makedirs(save_path, exist_ok = True)
	for index, fake_image in enumerate(fake_images):
		utils.save_image(fake_image.data, join(save_path, batch_filenames[index]))

def save_model(avg_loss, best_loss, encoder, decoder, save_path, epoch):
	if avg_loss < best_loss:
		best_loss = avg_loss
		torch.save(encoder.state_dict(), join(save_path, 'encoder_best.ckpt'), _use_new_zipfile_serialization = False)
		torch.save(decoder.state_dict(), join(save_path, 'decoder_best.ckpt'), _use_new_zipfile_serialization = False)
	torch.save(encoder.state_dict(), join(save_path, 'encoder_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
	torch.save(decoder.state_dict(), join(save_path, 'decoder_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
	return best_loss

def save_loss_history(train_mse_loss_history, train_kl_loss_history):
	fig = plt.figure('train_mse_loss_history')
	plt.plot(train_mse_loss_history, label = 'mse_loss')
	plt.xlabel("iters")
	plt.ylabel("loss")
	plt.legend(loc = 'best')
	plt.savefig('./train_mse_loss_history.png')

	fig = plt.figure('train_kl_loss_history')
	plt.plot(train_kl_loss_history, label = 'kl_loss')
	plt.xlabel("iters")
	plt.ylabel("loss")
	plt.legend(loc = 'best')
	plt.savefig('./train_kl_loss_history.png')

def train(args, device):
	torch.multiprocessing.freeze_support()
	from tqdm import tqdm
	if args.mode == 'train':
		train_data = load_data(args.mode, args.train_img_path, args.train_label_path, args.need_attribute, args.attribute)
		valid_data = load_data('valid', args.valid_img_path, args.valid_label_path, args.need_attribute, args.attribute)
		save_path = make_saving_path(args.pred_path)
		encoder, decoder, optimizer = load_models_optims(args.load, save_path)

		MSELoss = nn.MSELoss()
		MSELoss.cuda()
		best_loss = 100.0
		mse_loss = torch.tensor(100.0)
		train_mse_loss_history, train_kl_loss_history = [], []

		for epoch in range(args.load + 1, args.epochs):
			encoder.train()
			decoder.train()
			total_kl_loss = total_mse_loss = 0
			for index, (image, label, filename) in enumerate(tqdm(train_data, ncols = 70)):
				batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
				mean, logvar = encoder(batch_images)
				noises = sample_LatentSpace(mean, logvar, args.mode)
				fake_images = decoder(noises)
				mse_loss = MSELoss(fake_images, batch_images)
				train_mse_loss_history.append(mse_loss.item())
				total_mse_loss += mse_loss
				kl_loss = get_KL_Lambda(mse_loss.item()) * KL_loss(mean, logvar)
				train_kl_loss_history.append(kl_loss.item())
				total_kl_loss += kl_loss

				optimizer.zero_grad()
				(kl_loss + mse_loss).backward()
				optimizer.step()
			avg_loss = (total_kl_loss + total_mse_loss) / len(train_data)
			avg_kl_loss = total_kl_loss / len(train_data)
			avg_mse_loss = total_mse_loss / len(train_data)
			print('epoch:', epoch)
			print('train_avg_loss: {:.5f} avg_kl_loss: {:.5f} avg_mse_loss: {:.5f}'.format(avg_loss, avg_kl_loss, avg_mse_loss))

			with torch.no_grad():
				encoder.eval()
				decoder.eval()
				total_kl_loss = total_mse_loss = 0
				for index, (image, label, filename) in enumerate(tqdm(valid_data, ncols = 70)):
					batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
					mean, logvar = encoder(batch_images)
					noises = sample_LatentSpace(mean, logvar, args.mode)
					fake_images = decoder(noises)
					mse_loss = MSELoss(fake_images, batch_images)
					total_mse_loss += mse_loss
					kl_loss = get_KL_Lambda(mse_loss.item()) * KL_loss(mean, logvar)
					total_kl_loss += kl_loss
					save_fake_images(fake_images, batch_filenames, args.pred_path, epoch)
				avg_loss = (total_kl_loss + total_mse_loss) / len(valid_data)
				avg_kl_loss = total_kl_loss / len(valid_data)
				avg_mse_loss = total_mse_loss / len(valid_data)
				print('valid_avg_loss: {:.5f} avg_kl_loss: {:.5f} avg_mse_loss: {:.5f}'.format(avg_loss, avg_kl_loss, avg_mse_loss))
				print()
				best_loss = save_model(avg_loss, best_loss, encoder, decoder, save_path, epoch)

		save_loss_history(train_mse_loss_history, train_kl_loss_history)

	elif args.mode == 'valid':
		valid_data = load_data(args.mode, args.valid_img_path, args.valid_label_path, args.need_attribute, args.attribute)
		save_path = make_saving_path(args.pred_path)
		encoder, decoder, optimizer = load_models_optims(args.load, save_path)

		MSELoss = nn.MSELoss()
		MSELoss.cuda()
		best_loss = mse_loss = torch.tensor(100.0)
		with torch.no_grad():
			encoder.eval()
			decoder.eval()
			total_kl_loss = total_mse_loss = 0
			for index, (image, label, filename) in enumerate(valid_data):
				batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
				mean, logvar = encoder(batch_images)
				noises = sample_LatentSpace(mean, logvar, args.mode)
				fake_images = decoder(noises)
				mse_loss = MSELoss(fake_images, batch_images)
				total_mse_loss += mse_loss
				kl_loss = get_KL_Lambda(mse_loss.item()) * KL_loss(mean, logvar)
				total_kl_loss += kl_loss
				save_fake_images(fake_images, batch_filenames, args.pred_path, args.load)
				print()
				print('filename: ', filename[0])
				print('MSELoss : ', mse_loss.item())
			avg_loss = (total_kl_loss + total_mse_loss) / len(valid_data)
			avg_kl_loss = total_kl_loss / len(valid_data)
			avg_mse_loss = total_mse_loss / len(valid_data)
			print('epoch:', args.load)
			print('valid_avg_loss: {:.5f} avg_kl_loss: {:.5f} avg_mse_loss: {:.5f}'.format(avg_loss, avg_kl_loss, avg_mse_loss))

def random_valid(args, device):
	save_path = make_saving_path(args.pred_path)
	_, decoder, _ = load_models_optims(args.load, save_path)
	with torch.no_grad():
		decoder.eval()
		torch.manual_seed(113)
		fixed_noises = sample_LatentSpace(None, None, args.mode)
		os.makedirs(args.pred_path, exist_ok = True)
		torch.save(fixed_noises, './fixed_noises.npy', _use_new_zipfile_serialization = False)
		fixed_fake_images = decoder(fixed_noises)
		utils.save_image(fixed_fake_images.data, join(args.pred_path, 'random_32.png'))

def normalize(tsne_features_fit):
	min_num_x, max_num_x = min(tsne_features_fit[:, 0]), max(tsne_features_fit[:, 0])
	min_num_y, max_num_y = min(tsne_features_fit[:, 1]), max(tsne_features_fit[:, 1])
	tsne_features_fit[:, 0] = (tsne_features_fit[:, 0] - min_num_x) / (max_num_x - min_num_x)
	tsne_features_fit[:, 1] = (tsne_features_fit[:, 1] - min_num_y) / (max_num_y - min_num_y)
	return tsne_features_fit

def save_tsne(tsne_features, labels, attribute):
	combine = sorted(list(zip(tsne_features, labels)), key = lambda element: element[-1])
	tsne_features, labels = zip(*combine)
	tsne_features = np.array(tsne_features)
	labels = np.array(labels)
	last_noattribute_index = len(labels) - np.sum(labels)
	tsne_features_fit = TSNE(n_components = 2).fit_transform(tsne_features, labels)
	tsne_features_fit = normalize(tsne_features_fit)
	n_attribute = plt.scatter(tsne_features_fit[:last_noattribute_index, 0], tsne_features_fit[:last_noattribute_index, 1], s = 10)
	p_attribute = plt.scatter(tsne_features_fit[last_noattribute_index:, 0], tsne_features_fit[last_noattribute_index:, 1], s = 10)
	plt.legend((n_attribute, p_attribute), ('Not ' + attribute, attribute), loc = 'best')
	plt.show()

def show_tsne(args, device):
	torch.multiprocessing.freeze_support()
	from tqdm import tqdm
	test_data = load_data(args.mode, '../hw3_data/face/test/', args.valid_label_path, True, args.attribute)
	save_path = make_saving_path(args.pred_path)
	encoder, _, _ = load_models_optims(args.load, save_path)
	with torch.no_grad():
		encoder.eval()
		tsne_features, labels = [], []
		for index, (image, label, filename) in enumerate(tqdm(test_data, ncols = 70)):
			batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
			mean, logvar = encoder(batch_images)
			noises = sample_LatentSpace(mean, logvar, args.mode)
			tsne_features.append(noises.squeeze().cpu().numpy())
			labels.append(batch_labels.cpu())
		save_tsne(tsne_features, labels, args.attribute)

def random_valid_test(args, device):
	_, decoder, _ = load_models_optims(99, './Problem1/')
	with torch.no_grad():
		decoder.eval()
		fixed_noises = torch.load('./Problem1/fixed_noises.npy')
		fixed_fake_images = decoder(fixed_noises)
		utils.save_image(fixed_fake_images.data, args.pred_path)
		print('Finished!')

if __name__ == '__main__':
	args = _parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if args.mode == 'test':
		random_valid_test(args, device)
	elif args.mode == 'random':
		random_valid(args, device)
	elif args.mode == 'tsne':
		show_tsne(args, device)
	else:
		train(args, device)