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
import copy

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
		train_data, batch_size=batch_size, shuffle=True)
	valid_loader = torch.utils.data.DataLoader(
		valid_data, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(
		dset.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
		batch_size=test_batch_size, shuffle=True)

	return train_loader, valid_loader, test_loader
