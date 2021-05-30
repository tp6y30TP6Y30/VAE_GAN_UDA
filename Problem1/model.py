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

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.encode = nn.Sequential(
							Conv_Block(3, 64, 4, 2, 1),
							Conv_Block(64, 128, 4, 2, 1),
							Conv_Block(128, 256, 4, 2, 1),
							Conv_Block(256, 512, 4, 2, 1),
							Conv_Block(512, 1024, 4, 2, 1),
							Conv_Block(1024, 2048, 4, 2, 1)
					   )
		self.meanGen = nn.Linear(2048, 128)
		self.logvarGen = nn.Linear(2048, 128)
	def forward(self, input):
		batch, channel, width, height = input.shape
		feature = self.encode(input)
		feature = feature.view(batch, -1)
		mean, logvar = self.meanGen(feature), self.logvarGen(feature)
		return mean, logvar

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.preprocess = nn.Sequential(
							nn.Linear(128, 2048),
							nn.BatchNorm1d(2048),
							nn.ReLU(True)
						  )
		self.decode = nn.Sequential(
							Tran_Block(2048, 1024, 4, 2, 1),
							Tran_Block(1024, 512, 4, 2, 1),
							Tran_Block(512, 256, 4, 2, 1),
							Tran_Block(256, 128, 4, 2, 1),
							Tran_Block(128, 64, 4, 2, 1),
							nn.ConvTranspose2d(64, 3, 4, 2, 1),
							nn.Tanh()
					  )
	def forward(self, input):
		feature = self.preprocess(input)
		feature = feature.view(feature.shape[0], -1, 1, 1)
		image = self.decode(feature)
		return image