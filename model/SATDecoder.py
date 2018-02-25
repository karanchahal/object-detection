import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class SATDecoder(nn.Module):
    ''' Simple GRU Decoder, converts words to embeddings if not using word embeddings
        and then runs them through a 3 layer GRU cell, outputs a probability of the vocabulary,
        adds the conv feature map as the hidden state
    '''
    def __init__(self,vocab_size,feature_size=512,embed_size=512,hidden_size=512,num_layers=3,batch_size=1):
        super(SATDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size,vocab_size)
        self.lstm_input = nn.Linear(embed_size*2,embed_size)
        self.f_att = nn.Linear(feature_size,embed_size)
        
        self.init_weights()

    def init_weights(self): 
        '''Initialize the weights'''
        for m in self.modules():
            if isinstance(m,nn.Linear):
                m.weight.data.uniform_(-0.1, 0.1)
                m.bias.data.fill_(0)
            elif isinstance(m,nn.Embedding):
                m.weight.data.uniform_(-0.1, 0.1)

    def soft_attention(self,features,attention):
        ''' To do '''
        attention = attention.unsqueeze(2)
        attention = torch.sum(features*attention,dim=1)
        attention = self.f_att(attention)
        # print(attention.size())
        return attention

    def forward(self,features,captions,lengths):
        x = self.embedding(captions)
        self.hidden = self.initHidden(features)
        caption_length = x.size(1)

        for i in range(caption_length):
            
            last_hidden_state = self.hidden[-1].unsqueeze(2)
            attention = torch.bmm(features,last_hidden_state).squeeze(2)
            attention = F.sigmoid(attention)
            z_vectors = self.soft_attention(features,attention)
            cap_input = x[:,i,:]
            input_and_context = torch.cat((z_vectors,cap_input),dim=1)
            inp = self.lstm_input(input_and_context)
            x,self.hidden = self.gru(x,self.hidden)

        # print(x.size())
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

    def initHidden(self,features):
        hidden = Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size))
        features = torch.sum(features,dim=1)
        hidden[0] = features
        return hidden