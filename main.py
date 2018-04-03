__author__	= 	"Olga (Ge Ya) Xu"
__email__ 	=	"olga.xu823@gmail.com"

import model
from model.utils import *
from model.LSTMModel import LSTMAutoEncoder
import argparse, pdb, os

#torch.backends.cudnn.enabled = False

use_cuda = torch.cuda.is_available()

#torch.backends.cudnn.benchmark=True

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
	parser.add_argument('--epochs', default=20, type=int, help='Number of epochs')
	parser.add_argument('-a', '--alpha', default=0.6, type=float,
						help='Alpha')
	parser.add_argument('-g', '--gamma', default=0.3, type=float,
						help='Gamma')
	parser.add_argument('--number_plot', default=36, type=int,
						help='Number of examples to plot')
	parser.add_argument('-r', '--result_path', default='./result', type=str,
						help='Result path')
	parser.add_argument('--model_name', default='LSTM_AE', type=str,
						help='Model name')

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

		data, target = Variable(data, requires_grad=False), \
					   Variable(target, requires_grad=False)

		optimizer.zero_grad()
		pred, decoded = model(data)
		loss, _, _ = model.loss(pred, target, data, decoded)
		loss.backward()
		optimizer.step()


def eval(model, data_loader, check_mem=False, plot_flat=True, epoch_i=0, mode='train'):

	model.eval()

	if check_mem:
		pid = os.getpid()
		prev_mem = 0

	total_loss, total_e_loss, total_c_loss, correct = 0, 0, 0, 0
	total_data, total_batch = 0, 0
	

	for batch_idx, (data, target) in enumerate(data_loader):

		data = data.view(-1, 784, 1)

		if use_cuda:
			data, target = data.cuda(), target.cuda()

		data, target = Variable(data, requires_grad=False, volatile=True), \
					   Variable(target, requires_grad=False, volatile=True)

		pred, decoded = model(data)

		all_loss, encoder_loss, class_loss = model.loss(pred, target, data, decoded)
		total_loss += all_loss
		total_e_loss += encoder_loss
		total_c_loss += class_loss

		_, predicted = torch.max(pred.data, 1)

		correct += (predicted == target.data).sum()

		total_data += len(data)
		total_batch += 1

		if check_mem:
			cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
			add_mem = cur_mem - prev_mem
			prev_mem = cur_mem
			print("added mem: %sM"%(add_mem))

		if plot_flat and batch_idx == 0:
			visualize = decoded[:num_plot].squeeze().unsqueeze(1).repeat(1,3,1)
			visualize = visualize.view(num_plot,-1,28,28)
			visualize_kernel(visualize, im_name='epoch{}_{}.jpg'.format(epoch_i, mode),
							 model_name=model_name, rescale=True, result_path=result_path)

		del pred, decoded, data, target

	avg_loss = total_loss / float(total_batch)
	accuracy = correct / float(total_data)
	avg_e_loss = total_e_loss / float(total_batch)
	avg_c_loss = total_c_loss / float(total_batch)
	return avg_loss.cpu().data[0], accuracy, avg_e_loss.cpu().data[0], avg_c_loss.cpu().data[0]



if __name__ == '__main__':

	args = parse()
	output_model_setting(args)
	if not os.path.exists(args.result_path):
   		os.makedirs(args.result_path)

	global result_path, model_name, num_plot
	result_path, model_name, num_plot = args.result_path, args.model_name, args.number_plot
	
	torch.manual_seed(args.seed)
	
	if use_cuda:
		torch.cuda.manual_seed_all(args.seed)

	train_loader, valid_loader, test_loader = load_data(batch_size=args.batch_size,
														test_batch_size=args.test_batch_size,
														alpha=args.alpha)

	print('Number of training data: {}\n'.format(len(train_loader.dataset)))

	model = LSTMAutoEncoder(input_size=1, hidden_size=16, num_layers=3,
							num_class=10, sequence_len=784, gamma=0.4)
	
	if use_cuda:
		model.cuda()

	optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
	train_loss, valid_loss, train_acc, valid_acc = [],[],[],[]

	print('Epoch {}:'.format(0))
	avg_loss, accuracy, e_loss, c_loss = eval(model, train_loader)
	train_loss.append(avg_loss); train_acc.append(accuracy)
	print('|\t[TRAIN]: loss={:.3f}\t\taccuracy={:.3f}'.format(avg_loss, accuracy))
	print('|\t[TRAIN]: encoder loss={:.3f}\tclassification loss={:.3f}'.format(e_loss, c_loss))
	avg_loss, accuracy, e_loss, c_loss = eval(model, valid_loader, mode='valid')
	valid_loss.append(avg_loss); valid_acc.append(accuracy)
	print('|\t[VALID]: loss={:.3f}\t\taccuracy={:.3f}'.format(avg_loss, accuracy))
	print('|\t[VALID]: encoder loss={:.3f}\tclassification loss={:.3f}'.format(e_loss, c_loss))

	for epoch_i in range(args.epochs):
		train(model, optimizer, train_loader)
		
		print('Epoch {}:'.format(epoch_i+1))
		avg_loss, accuracy, e_loss, c_loss = eval(model, train_loader, epoch_i=epoch_i+1)
		train_loss.append(avg_loss); train_acc.append(accuracy)
		print('|\t[TRAIN]: loss={:.3f}\t\taccuracy={:.3f}'.format(avg_loss, accuracy))
		print('|\t[TRAIN]: encoder loss={:.3f}\tclassification loss={:.3f}'.format(e_loss, c_loss))
		avg_loss, accuracy, e_loss, c_loss = eval(model, valid_loader, epoch_i=epoch_i+1, mode='valid')
		valid_loss.append(avg_loss); valid_acc.append(accuracy)
		print('|\t[VALID]: loss={:.3f}\t\taccuracy={:.3f}'.format(avg_loss, accuracy))
		print('|\t[VALID]: encoder loss={:.3f}\tclassification loss={:.3f}'.format(e_loss, c_loss))

	plt.plot(list(range(0,epoch_i+1,1)), train_loss, 'ro-', label='train loss')
	plt.plot(list(range(0,epoch_i+1,1)), valid_loss, 'bs-', label='valid loss')
	plt.title('average loss at each epoch')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend(loc=1)
	plt.savefig(os.path.join(result_path, model_name+'_loss.png'))
	plt.clf()

	plt.plot(list(range(0,args.epochs+1,1)), train_acc, 'ro-', label='train accuracy')
	plt.plot(list(range(0,args.epochs+1,1)), valid_acc, 'bs-', label='valid accuracy')
	plt.title('average accuracy at each epoch')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.legend(loc=4)
	plt.savefig(os.path.join(result_path, model_name+'_accuracy.png'))
	plt.clf()


