import torch.nn as nn
from Encoder import Encoder
from torch.autograd import Variable
import torch

class Decoder(nn.Module):
    ''' Simple GRU Decoder, converts words to embeddings if not using word embeddings
        and then runs them through a 3 layer GRU cell, outputs a probability of the vocabulary,
        adds the conv feature map as the hidden state
    '''
    def __init__(self,vocab_size,embed_size=300,hidden_size=300,num_layers=3,batch_size=1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,vocab_size)

        self.hidden = self.initHidden()
        self.init_weights()

    def init_weights(self): 
        '''Initialize the weights'''
        for m in self.modules():
            if isinstance(m,nn.Linear):
                m.weight.data.uniform_(-0.1, 0.1)
                m.bias.data.fill_(0)
            elif isinstance(m,nn.Embedding):
                m.weight.data.uniform_(-0.1, 0.1)

    def forward(self,x):
        x = self.embedding(x)
        print(x.size())
        x, self.hidden = self.gru(x,self.hidden)
        print(x.size())
        x = self.fc(x)
        print(x.size())
        return x
    
    def initHidden(self):
        return Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size))
