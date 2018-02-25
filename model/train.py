import torch
from torch.autograd import Variable
from torchvision.models.resnet import resnet34
from SATDecoder import SATDecoder
from SATEncoder import SATEncoder
import coloredlogs, logging
# logging settings

# Create a logger object.
logger = logging.getLogger(__name__)
batch_size = 1

a = torch.autograd.Variable(torch.zeros((batch_size,3,224,224)))
conv = resnet34(pretrained=True)
enc = SATEncoder(conv)
b = enc(a)


dec = SATDecoder(vocab_size=10000,batch_size=batch_size)
caption_len = 10
lens = [caption_len for i in range(batch_size)]
c = torch.zeros((batch_size,caption_len)).long()
c = Variable(c)
c = dec(b,c,lens)

