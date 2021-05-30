import os
from os import listdir
from os.path import join
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd

# image_name,Bangs,Big_Lips,Black_Hair,Blond_Hair,Brown_Hair,Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Smiling,Straight_Hair,Wavy_Hair,Wearing_Lipstick

transform = transforms.Compose([transforms.ToTensor(),
							   ])

augmentation = transforms.Compose([transforms.RandomHorizontalFlip(p = 0.2),
								   transforms.ToTensor(),
								  ])

class dataloader(Dataset):
	def __init__(self, mode, img_path, label_path, need_attribute, attribute):
		super(dataloader, self).__init__()
		self.mode = mode
		self.img_path = img_path
		self.img_list = sorted(listdir(self.img_path))
		self.label_path = label_path
		self.need_attribute = need_attribute
		self.attribute = attribute

	def __len__(self):
		return len(self.img_list)

	def __getitem__(self, index):
		image = Image.open(join(self.img_path, self.img_list[index]))
		image = augmentation(image) if self.mode == 'train' else transform(image)
		label = 0
		if self.need_attribute:
			csv_reader = pd.read_csv(self.label_path)
			label = int(csv_reader[csv_reader['image_name'] == self.img_list[index]][self.attribute])
		return image, label, self.img_list[index]

if __name__ == '__main__':
	test = dataloader('train', '../hw3_data/face/train/', '../hw3_data/face/train.csv', 'Male')
	test_data = DataLoader(test, batch_size = 1, shuffle = True)
	print(len(test_data))
	for index, (image, label) in enumerate(test_data):
		print(index, image.shape, label)
		break