import os
from os import listdir
from os.path import join
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch
import pandas as pd

class dataloader(Dataset):
	def __init__(self, mode, img_path, folder, source, target):
		super(dataloader, self).__init__()
		self.mode = mode
		if self.mode == 'test':
			self.img_path = img_path
			self.img_list = sorted(listdir(self.img_path))
			if target == 'mnistm':
				self.transform = transforms.Compose([transforms.ToTensor()])
			elif target == 'svhn':
				self.transform = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()])
			elif target == 'usps':
				self.transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()])
		else:
			self.img_path = join(img_path, folder, 'train/' if self.mode == 'train' else 'test/')
			self.label_path = join(img_path, folder, 'train.csv' if self.mode == 'train' else 'test.csv')
			self.img_list = sorted(listdir(self.img_path))
			self.csv_reader = pd.read_csv(self.label_path).sort_values('image_name')
			if source == 'usps' and (target == 'mnistm' or target == 'usps'):
				self.transform = transforms.Compose([transforms.ToTensor()])
			elif source == 'mnistm' and (target == 'svhn' or target == 'mnistm'):
				self.transform = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor()])
			elif source == 'svhn' and (target == 'usps' or target == 'svhn'):
				self.transform = transforms.Compose([transforms.Grayscale(1), transforms.ToTensor()])

	def __len__(self):
		return len(self.img_list)

	def __getitem__(self, index):
		image = Image.open(join(self.img_path, self.img_list[index]))
		image = self.transform(image)
		if image.shape[0] == 1:
			image = image.repeat(3, 1, 1).squeeze()
		if self.mode == 'test':
			return image, self.img_list[index]
		else:
			label = int(self.csv_reader[self.csv_reader['image_name'] == self.img_list[index]]['label'])
			return image, label, self.img_list[index]

if __name__ == '__main__':
	test_dataloader = dataloader('train', '../hw3_data/digits/', 'usps')
	test_data = DataLoader(test_dataloader, batch_size = 1, shuffle = False)
	from tqdm import tqdm
	for index, (image, label, filename) in enumerate(tqdm(test_data, ncols = 70)):
		print(image.shape)
		print(label)
		print(filename)
		break