import torch
import torch.nn as nn

class Conv_Block(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super(Conv_Block, self).__init__()
		self.block = nn.Sequential(
						nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
						nn.BatchNorm2d(out_channels),
						nn.ReLU(True),
					 )
	def forward(self, input):
		return self.block(input)


class Linear_Block(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Linear_Block, self).__init__()
		self.block = nn.Sequential(
						nn.Linear(in_channels, out_channels),
						nn.ReLU(True),
					 )
	def forward(self, input):
		return self.block(input)

class Extractor(nn.Module):
	def __init__(self):
		super(Extractor, self).__init__()
		self.extract = nn.Sequential(
							Conv_Block(3, 64, 3, 2, 1),
							Conv_Block(64, 128, 3, 2, 1),
							Conv_Block(128, 256, 3, 2, 1),
							Conv_Block(256, 512, 3, 2, 1),
							nn.MaxPool2d(2)
					   )
	def forward(self, input):
		batch, channel, width, height = input.shape
		features = self.extract(input).view(batch, -1)
		return features

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		self.preprocess = Linear_Block(512, 1024)
		self.classify = nn.Sequential(
							Linear_Block(1024, 512),
							Linear_Block(512, 256),
							Linear_Block(256, 128),
							Linear_Block(128, 64),
							nn.Linear(64, 10),
							nn.LogSoftmax(dim = 1)
					    )

	def forward(self, input):
		features = self.preprocess(input)
		prediction = self.classify(features)
		return prediction

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.preprocess = Linear_Block(512, 1024)
		self.classify = nn.Sequential(
							Linear_Block(1024, 512),
							Linear_Block(512, 256),
							Linear_Block(256, 128),
							Linear_Block(128, 64),
							nn.Linear(64, 2),
							nn.LogSoftmax(dim = 1)
					    )

	def forward(self, input):
		features = self.preprocess(input)
		prediction = self.classify(features)
		return prediction