import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import pack_padded_sequence


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
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
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

    def forward(self,features,captions,lengths):
        x = self.embedding(captions)
        x = torch.cat((features.unsqueeze(1), x), 1)
        packed = pack_padded_sequence(x,lengths,batch_first=True)
        x, _ = self.lstm(packed)
        x = self.fc(x[0])
        return x
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.fc(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(int(predicted.data.cpu().numpy()[0]))
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        # sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return torch.Tensor(sampled_ids)

    def initHidden(self):
        return Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size))
