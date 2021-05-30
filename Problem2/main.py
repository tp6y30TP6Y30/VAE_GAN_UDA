import os
import os.path
from os.path import join
import argparse
import torch
import torch.nn as nn
from torch import optim
from dataloader import dataloader
from torch.utils.data import DataLoader
from model import Generator, Discriminator
import numpy as np
import time
import torchvision.utils as utils

def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	parser.add_argument('--load', type = int, default = -1)
	parser.add_argument('--epochs', type = int, default = 0)
	parser.add_argument('--train_img_path', type = str, default = '../hw3_data/face/train/')
	parser.add_argument('--pred_path', type = str, default = './predict/')
	return parser.parse_args()

def load_data(mode, img_path, batch_size):
	loader = dataloader(mode, img_path)
	data = DataLoader(loader, batch_size = batch_size if mode == 'train' else 1, shuffle = (mode == 'train'), num_workers = 6 * (mode == 'train'), pin_memory = True)
	return data

def make_saving_path(pred_path):
	save_path = './models/'
	os.makedirs(save_path, exist_ok = True)
	os.makedirs(pred_path, exist_ok = True)
	return save_path

def load_models_optims(load, save_path):
	print('loading Generator...')
	Gen = Generator()
	print(Gen)
	total_params = sum(p.numel() for p in Gen.parameters() if p.requires_grad)
	print("Total number of params = ", total_params)
	Gen.cuda().float()
	print('loading Discriminator...')
	Dsc = Discriminator()
	print(Dsc)
	total_params = sum(p.numel() for p in Dsc.parameters() if p.requires_grad)
	print("Total number of params = ", total_params)
	Dsc.cuda().float()
	if load != -1:
		Gen.load_state_dict(torch.load(join(save_path, 'Gen_' + str(load) + '.ckpt')))
		Dsc.load_state_dict(torch.load(join(save_path, 'Dsc_' + str(load) + '.ckpt')))
	optim_Gen = optim.Adam(Gen.parameters(), lr = 1e-4, betas = (0.5, 0.9))
	optim_Dsc = optim.Adam(Dsc.parameters(), lr = 1e-4, betas = (0.5, 0.9))
	return Gen, Dsc, optim_Gen, optim_Dsc

def save_model(Gen, Dsc, save_path, epoch):
	torch.save(Gen.state_dict(), join(save_path, 'Gen_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)
	torch.save(Dsc.state_dict(), join(save_path, 'Dsc_{}.ckpt'.format(epoch)), _use_new_zipfile_serialization = False)

def train(args, device):
	torch.multiprocessing.freeze_support()
	from tqdm import tqdm
	batch_size = 64
	train_data = load_data(args.mode, args.train_img_path, batch_size)
	save_path = make_saving_path(args.pred_path)
	Gen, Dsc, optim_Gen, optim_Dsc = load_models_optims(args.load, save_path)

	BCELoss = nn.BCELoss()
	BCELoss.cuda()
	best_loss = 100.0

	for epoch in range(args.load + 1, args.epochs):
		total_Gen_loss = total_real_loss = total_fake_loss = 0
		for index, (image, label, filename) in enumerate(tqdm(train_data, ncols = 70)):
			batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
			Gen.train()
			Dsc.train()
			Gen.zero_grad()
			optim_Gen.zero_grad()
			noises =  torch.randn(batch_images.shape[0], 128).cuda()
			fake_images = Gen(noises)
			fake_predict = Dsc(fake_images)
			Gen_loss = BCELoss(fake_predict, batch_labels.unsqueeze(-1))
			total_Gen_loss += Gen_loss.item()
			Gen_loss.backward()
			optim_Gen.step()

			Dsc.zero_grad()
			optim_Dsc.zero_grad()
			noises, fake_labels = torch.randn(batch_images.shape[0], 128).cuda(), torch.zeros(batch_images.shape[0], 1).cuda()
			fake_images = Gen(noises)
			real_predict, fake_predict = Dsc(batch_images), Dsc(fake_images)
			real_loss, fake_loss = BCELoss(real_predict, batch_labels.unsqueeze(-1)), BCELoss(fake_predict, fake_labels)
			total_real_loss += real_loss.item()
			total_fake_loss += fake_loss.item()
			Dsc_loss = real_loss + fake_loss
			Dsc_loss.backward()
			optim_Dsc.step()

		avg_Gen_loss = total_Gen_loss / len(train_data)
		avg_real_loss = total_real_loss / len(train_data)
		avg_fake_loss = total_fake_loss / len(train_data)
		print('epoch:', epoch)
		print('train_avg_Gen_loss: {:.5f} train_avg_real_loss: {:.5f} train_avg_fake_loss: {:.5f}'.format(avg_Gen_loss, avg_real_loss, avg_fake_loss))
		with torch.no_grad():
			Gen.eval()
			Dsc.eval()
			torch.manual_seed(311)
			fixed_noises = torch.randn(32, 128).cuda()
			real_labels, fake_labels = torch.ones(32, 1).cuda(), torch.zeros(32, 1).cuda()
			fixed_fake_images = Gen(fixed_noises)
			fake_predict = Dsc(fixed_fake_images)
			Gen_loss, Dsc_loss = BCELoss(fake_predict, real_labels), BCELoss(fake_predict, fake_labels)
			print('valid_Gen_loss: {:.5f} valid_fake_loss: {:.5f}'.format(Gen_loss.item(), Dsc_loss.item()))
			print()
			utils.save_image(fixed_fake_images.data, join(args.pred_path, 'epoch_{}.png'.format(epoch)))
		save_model(Gen, Dsc, save_path, epoch)

def random_valid(args, device):
	save_path = make_saving_path(args.pred_path)
	Gen, _, _, _ = load_models_optims(args.load, save_path)
	with torch.no_grad():
		Gen.eval()
		torch.manual_seed(311)
		fixed_noises = torch.randn(32, 128).cuda()
		os.makedirs(args.pred_path, exist_ok = True)
		torch.save(fixed_noises, './fixed_noises.npy', _use_new_zipfile_serialization = False)
		fixed_fake_images = Gen(fixed_noises)
		utils.save_image(fixed_fake_images.data, join(args.pred_path, 'random_32.png'))

def random_valid_test(args, device):
	Gen, _, _, _ = load_models_optims(99, './Problem2/')
	with torch.no_grad():
		Gen.eval()
		fixed_noises = torch.load('./Problem2/fixed_noises.npy').cuda()
		fixed_fake_images = Gen(fixed_noises)
		utils.save_image(fixed_fake_images.data, args.pred_path)
		print('Finished!')

if __name__ == '__main__':
	args = _parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if args.mode == 'test':
		random_valid_test(args, device)
	elif args.mode == 'random':
		random_valid(args, device)
	else:
		train(args, device)