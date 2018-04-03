import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable
import torch.nn.functional as F

class LSTMAutoEncoder(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_class, sequence_len,
				 gamma=0.5):
		super(LSTMAutoEncoder, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_class = num_class
		self.num_layers = num_layers

		self.encoder = nn.GRU(self.input_size, self.hidden_size, self.num_layers,
							  batch_first=True)
		self.decoder = nn.GRU(self.hidden_size, self.input_size,
							  1, batch_first=True)
		self.out_layer = nn.Linear(self.hidden_size*sequence_len, self.num_class)

		self.encoded_h_init = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))
		self.decoded_h_init = nn.Parameter(torch.zeros(1, 1, self.input_size))

		self.idx = [i for i in range(sequence_len-1, -1, -1)]
		self.idx = Variable(torch.LongTensor(self.idx), volatile=True)
		
		if torch.cuda.is_available():
			self.idx = self.idx.cuda()

		self.gamma = gamma

	def set_init(self, batch_size):
		return self.encoded_h_init.clone().repeat(1, batch_size, 1), \
				self.decoded_h_init.clone().repeat(1, batch_size, 1)


	def forward(self, x):
		batch_size = x.size(0)
		
		encoded_h0, decoded_h0 = self.set_init(batch_size)


		encoded_h, _ = self.encoder(x, encoded_h0)
		out = F.log_softmax(self.out_layer(encoded_h.contiguous().view(batch_size, -1)), dim=1)

		encoded_h = encoded_h.index_select(1, self.idx)
		decoded_h, _= self.decoder(encoded_h, decoded_h0)
		del encoded_h
		return out, decoded_h


	def loss(self, out, target, inp, decoded):
		batch_size = inp.size(0)
		autoencoder_loss = F.mse_loss(decoded.contiguous().view(batch_size, -1),
									  inp.contiguous().view(batch_size, -1)) 
		classification_loss = F.nll_loss(out, target)

		return (1-self.gamma)*autoencoder_loss+self.gamma*classification_loss, \
				autoencoder_loss, classification_loss
			

