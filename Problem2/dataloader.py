import os
from os import listdir
from os.path import join
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch

transform = transforms.Compose([transforms.ToTensor(),
							   ])

augmentation = transforms.Compose([transforms.RandomHorizontalFlip(p = 0.2),
								   transforms.ToTensor(),
								  ])

class dataloader(Dataset):
	def __init__(self, mode, img_path):
		super(dataloader, self).__init__()
		self.mode = mode
		self.img_path = img_path
		self.img_list = sorted(listdir(self.img_path))

	def __len__(self):
		return len(self.img_list)

	def __getitem__(self, index):
		image = Image.open(join(self.img_path, self.img_list[index]))
		image = augmentation(image) if self.mode == 'train' else transform(image)
		label = torch.tensor(1.0)
		return image, label, self.img_list[index]

if __name__ == '__main__':
	test = dataloader('train', '../hw3_data/face/train/', '../hw3_data/face/train.csv', 'Male')
	test_data = DataLoader(test, batch_size = 1, shuffle = True)
	print(len(test_data))
	for index, (image, label) in enumerate(test_data):
		print(index, image.shape, label)
		break