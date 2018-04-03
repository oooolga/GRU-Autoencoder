__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import argparse, os
import copy, scipy
import scipy.misc

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def load_data(batch_size, test_batch_size, alpha=1):

	train_data = dset.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
	train_data.train_data = train_data.train_data[:50000]
	train_data.train_labels = train_data.train_labels[:50000]
	valid_data = dset.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
	valid_data.train_data = valid_data.train_data[50000:]
	valid_data.train_labels = valid_data.train_labels[50000:]

	train_data_len = len(train_data)
	train_data.train_data = train_data.train_data[:int(alpha*train_data_len)]
	train_data.train_labels = train_data.train_labels[:int(alpha*train_data_len)]

	train_loader = torch.utils.data.DataLoader(
		train_data, batch_size=batch_size, shuffle=True, drop_last=True)
	valid_loader = torch.utils.data.DataLoader(
		valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
	test_loader = torch.utils.data.DataLoader(
		dset.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
		batch_size=test_batch_size, shuffle=True, drop_last=True)

	return train_loader, valid_loader, test_loader

def factorization(n):
	from math import sqrt
	for i in range(int(sqrt(float(n))), 0, -1):
		if n % i == 0:
			if i == 1: print('Who would enter a prime number of filters')
			return int(n / i), i

def visualize_kernel(kernel_tensor, im_name='conv1_kernel.jpg', pad=1, im_scale=1.0,
					 model_name='', rescale=True, result_path='.'):

	# map tensor wight in [0,255]
	if rescale:
		max_w = torch.max(kernel_tensor)
		min_w = torch.min(kernel_tensor)
		scale = torch.abs(max_w-min_w)
		kernel_tensor = (kernel_tensor - min_w) / scale * 255.0
		kernel_tensor = torch.ceil(kernel_tensor)

	# pad kernel
	p2d = (pad, pad, pad, pad)
	padded_kernel_tensor = F.pad(kernel_tensor, p2d, 'constant', 0)

	# get the shape of output
	grid_Y, grid_X = factorization(kernel_tensor.size(0))
	Y, X = padded_kernel_tensor.size(2), padded_kernel_tensor.size(3)

	# reshape
	# (grid_Y*grid_X) x y_dim x x_dim x num_chann
	padded_kernel_tensor = padded_kernel_tensor.permute(0, 2, 3, 1)
	padded_kernel_tensor = padded_kernel_tensor.cpu().view(grid_X, grid_Y*Y, X, -1)
	padded_kernel_tensor = padded_kernel_tensor.permute(0, 2, 1, 3)
	#padded_kernel_tensor = padded_kernel_tensor.view(1, grid_X*X, grid_Y*Y, -1)

	# kernel in numpy
	kernel_im = np.uint8((padded_kernel_tensor.data).numpy()).reshape(grid_X*X,
																	   grid_Y*Y, -1)
	kernel_im = scipy.misc.imresize(kernel_im, im_scale, 'nearest')
	print('|\tSaving {}...'.format(os.path.join(result_path, model_name+'_'+im_name)))
	scipy.misc.imsave(os.path.join(result_path, model_name+'_'+im_name), kernel_im)