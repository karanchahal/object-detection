import torch
from torch.autograd import Variable
from torchvision.models.resnet import resnet34
from Decoder import Decoder
from Encoder import Encoder
import coloredlogs, logging
# logging settings

# Create a logger object.
logger = logging.getLogger(__name__)
batch_size = 2

a = torch.autograd.Variable(torch.zeros((batch_size,3,224,224)))
conv = resnet34(pretrained=True)
enc = Encoder(conv)
b = enc(a)


dec = Decoder(vocab_size=10000,batch_size=batch_size)
c = torch.zeros((batch_size,1)).long()
c = Variable(c)
c = dec(c)

