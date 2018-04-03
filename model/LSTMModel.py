import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
import torch.nn.functional as F

class LSTMAutoEncoder(nn.Module):
	def __init__(self, input_size, hidden_size, batch_size, num_layers, num_class, sequence_len,
				 gamma=0.5):
		super(LSTMAutoEncoder, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_class = num_class
		self.num_layers = num_layers
		self.sequence_len = sequence_len
		self.batch_size = batch_size

		self.encoder = nn.GRU(self.input_size, self.hidden_size, self.num_layers,
							  batch_first=True)
		self.decoder = nn.GRU(self.hidden_size, self.input_size,
							  1, batch_first=True)
		self.out_layer = nn.Linear(self.hidden_size*sequence_len, self.num_class)
		self.encoded_h0 = nn.Parameter(torch.zeros(self.num_layers, batch_size, self.hidden_size))
		self.decoded_h0 = nn.Parameter(torch.zeros(1, batch_size, self.input_size))

		self.idx = [i for i in range(x.size(1)-1, -1, -1)]
		self.idx = Variable(torch.LongTensor(self.idx))
		
		if torch.cuda.is_available():
			self.encoded_h0 = self.encoded_h0.cuda()
			self.decoded_h0 = self.decoded_h0.cuda()
			self.idx = self.idx.cuda()

		self.gamma = gamma

		self.MSELoss = nn.MSELoss()
		self.NLLLoss = nn.NLLLoss()


	def forward(self, x):
		
		encoded_h, encoded_hn = self.encoder(x, self.encoded_h0)
		out = F.log_softmax(self.out_layer(encoded_h.contiguous().view(self.batch_size, -1)))

		encoded_h = encoded_h.index_select(1, self.idx)
		decoded_h, decoded_hn = self.decoder(encoded_h, self.decoded_h0)
		return out, decoded_h


	def loss(self, out, target, inp, decoded):
		return (1-self.gamma)*self.MSELoss(decoded.contiguous().view(self.batch_size, -1),
										   inp.contiguous().view(self.batch_size, -1)) +\
			self.gamma*self.NLLLoss(out, target)

