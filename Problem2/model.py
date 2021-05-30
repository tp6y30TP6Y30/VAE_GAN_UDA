import torch
import torch.nn as nn

class Conv_Block(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super(Conv_Block, self).__init__()
		self.block = nn.Sequential(
						nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
						nn.BatchNorm2d(out_channels),
						nn.LeakyReLU(0.2, inplace = True)
					 )
	def forward(self, input):
		output = self.block(input)
		return output

class Tran_Block(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super(Tran_Block, self).__init__()
		self.block = nn.Sequential(
						nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
						nn.BatchNorm2d(out_channels),
						nn.ReLU(True)
					 )
	def forward(self, input):
		output = self.block(input)
		return output

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.encode = nn.Sequential(
							Conv_Block(3, 64, 4, 2, 1),
							Conv_Block(64, 128, 4, 2, 1),
							Conv_Block(128, 256, 4, 2, 1),
							Conv_Block(256, 512, 4, 2, 1),
							nn.Conv2d(512, 1, 4, 1, 0),
							nn.Sigmoid()
					   )
	def forward(self, input):
		batch, channel, width, height = input.shape
		prediction = self.encode(input).view(batch, 1)
		return prediction

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.decode = nn.Sequential(
							Tran_Block(128, 512, 4, 1, 0),
							Tran_Block(512, 256, 4, 2, 1),
							Tran_Block(256, 128, 4, 2, 1),
							Tran_Block(128, 64, 4, 2, 1),
							nn.ConvTranspose2d(64, 3, 4, 2, 1),
							nn.Tanh()
					  )
	def forward(self, input):
		feature = input.view(input.shape[0], -1, 1, 1)
		image = self.decode(feature)
		return image