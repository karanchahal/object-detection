from torchvision.models.resnet import resnet34
import torch.nn as nn
from Encoder import Encoder
from torch.autograd import Variable
import torch

class Decoder(nn.Module):
    
    def __init__(self,vocab_size,embed_size=300,hidden_size=300,num_layers=3,batch_size=1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
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
        x, self.hidden = self.gru(x,self.hidden)
        x = self.fc(x)
        return x
    
    def initHidden(self):
        return Variable(torch.zeros(self.num_layers,self.batch_size,self.hidden_size))


a = torch.autograd.Variable(torch.zeros((1,3,224,224)))
conv = resnet34(pretrained=True)
enc = Encoder(conv)
b = enc(a)
print(b.size())
batch_size = 2
dec = Decoder(vocab_size=10000,batch_size=batch_size)
c = torch.zeros((batch_size,1)).long()
c = Variable(c)
c = dec(c)
print(c.size())
