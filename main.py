__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

import model
from model.utils import *
from model.LSTMModel import LSTMAutoEncoder
import argparse, pdb
import torch

use_cuda = torch.cuda.is_available()
torch.backends.cudnn.benchmark=True

def parse():
	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float,
						help='Learning rate')
	parser.add_argument('-m', '--momentum', default=0.5, type=float, help="Momentum")
	parser.add_argument('-s', '--seed', default=123, type=int, help='Random seed')
	parser.add_argument('--batch_size', default=128, type=int,
						help='Mini-batch size for training')
	parser.add_argument('--test_batch_size', default=1000, type=int,
						help='Mini-batch size for testing')
	parser.add_argument('--plot_iter', default=200, type=int, help='Plot iteration')
	parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
	parser.add_argument('-r', '--result_path', default='./result', type=str,
						help='Result path')
	parser.add_argument('-a', '--alpha', default=0.6, type=float,
						help='Alpha')

	args = parser.parse_args()
	return args

def output_model_setting(args):
	print('Learning rate: {}'.format(args.learning_rate))
	print('Mini-batch size: {}\n'.format(args.batch_size))

def train(model, optimizer, train_loader):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):

		data = data.view(-1, 784, 1)

		if use_cuda:
			data, target = data.cuda(), target.cuda()

		data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)

		optimizer.zero_grad()
		pred, decoded = model(data)
		loss = model.loss(pred, target, data, decoded)
		loss.backward()
		optimizer.step()


def eval(model, data_loader):

	model.eval()

	total_loss, correct = 0, 0
	total_data, total_batch = 0, 0

	for batch_idx, (data, target) in enumerate(data_loader):

		data = data.view(-1, 784, 1)

		if use_cuda:
			data, target = data.cuda(), target.cuda()

		data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)

		pred, decoded = model(data)

		total_loss += model.loss(pred, target, data, decoded)

		_, predicted = torch.max(pred.data, 1)

		correct += (predicted == target.data).sum()

		total_data += len(data)
		total_batch += 1

		print("loss={}".format(total_loss.cpu().data[0]))

	avg_loss = total_loss / float(total_batch)
	accuracy = correct / float(total_data)
	return avg_loss, accuracy



if __name__ == '__main__':

	args = parse()
	output_model_setting(args)

	torch.manual_seed(args.seed)
	
	if use_cuda:
		torch.cuda.manual_seed_all(args.seed)

	train_loader, valid_loader, test_loader = load_data(batch_size=args.batch_size,
														test_batch_size=args.test_batch_size,
														alpha=args.alpha)

	print('Number of training data: {}\n'.format(len(train_loader.dataset)))

	model = LSTMAutoEncoder(input_size=1, hidden_size=16, batch_size=args.batch_size, num_layers=3,
							num_class=10, sequence_len=784, gamma=0.4)
	
	if use_cuda:
		model.cuda()

	optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

	avg_loss, accuracy = eval(model, train_loader)
	print('Epoch {}:\tloss={}\taccuracy={}'.format(0, avg_loss, accuracy))

	for epoch_i in range(args.epochs):
		train(model, optimizer, train_loader)
		avg_loss, accuracy = eval(model, train_loader)
		print('Epoch {}:\tloss={}\taccuracy={}'.format(epoch_i+1, avg_loss, accuracy))

