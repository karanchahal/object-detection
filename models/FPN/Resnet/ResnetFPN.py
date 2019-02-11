import torch.nn as nn
import torch
from core import resnet34

class ResnetFPN(nn.Module):
    def __init__(self):
        super(ResnetFPN, self).__init__()
        self.resnet = resnet34(pretrained=True)
        self.resnet.init_fpn()
        
    def forward(self, x):
        return self.resnet(x)

def test():
    A = torch.randn(1,3,800,800)
    model = ResnetFPN()
    a,b,c,d = model(A)
    print(a.size())
    print(b.size())
    print(c.size())
    print(d.size())