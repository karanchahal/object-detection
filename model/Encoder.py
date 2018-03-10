from torchvision.models.resnet import resnet34
import torch.nn as nn
import torch
from torch.autograd import Variable

class Encoder(nn.Module):
    
    def __init__(self,conv,hidden_size=256):
        super(Encoder, self).__init__()
        self.conv = self.prune(conv)
        inplanes = conv.fc.in_features
        # self.fc = nn.Sequential(
        #     nn.Linear(inplanes,hidden_size),
        #     nn.BatchNorm1d(hidden_size,momentum=0.01)
        # )

        self.linear = nn.Linear(inplanes, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.01)
        self.init_weights()

    def init_weights(self): 
        '''Initialize the weights'''
        # for m in self.fc:
        #     if isinstance(m,nn.Linear):
        #         m.weight.data.normal_(0.0, 0.02)
        #         m.bias.data.fill_(0)
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def prune(self,conv):
        '''Prune last fully connected layer and fix conv layers'''
        modules = list(conv.children())[:-1]
        
        # for m in modules:
        #     m.require_grad = False

        return nn.Sequential(*modules)

    def forward(self,images):
        features = self.conv(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
