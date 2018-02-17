import torch
from data.dataset import get_image_ids,coco,coco_caps,get_ann_ids
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.DataLoader import CocoDataset, RandomCrop, collate_fn
from text.vocab import WordModel
import coloredlogs, logging
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models.resnet import resnet34
from model.Encoder import Encoder
from model.Decoder import Decoder
import torch.nn as nn
# logging settings

# Create a logger object.
logger = logging.getLogger(__name__)

use_cuda = torch.cuda.is_available()

batch_size = 4
dataset_size = 16
annIds = get_ann_ids()
imgIds = get_image_ids()
logger.warning("Loading Dataset")

'''Dataset Loading'''

word_model = WordModel()
word_model.load(filename="model_logs/word_model.pkl")

image_transform = transforms.Compose([
        RandomCrop(224),
        transforms.ToTensor()
    ])
dataset = CocoDataset(annIds=annIds,
                    coco=coco,
                    coco_caps=coco_caps,
                    word_model=word_model,
                    transform=image_transform
                )
dataloader = torch.utils.data.DataLoader(
                                        dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=4,
                                        collate_fn=collate_fn
                                    )

logger.warning("Starting training loop")
'''Training Loop'''

print('Number of examples', word_model.vocab.examples)
print('Number of words', len(word_model.vocab.id2word))

# Loading the model

logger.warning("Loading the model")
conv = resnet34(pretrained=True)


encoder = Encoder(conv)
decoder = Decoder(vocab_size=word_model.vocab.length(),batch_size=batch_size)

print('CUDA is ', use_cuda)

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

criterion = nn.CrossEntropyLoss()
params = decoder.parameters()
optimizer = torch.optim.Adam(params, lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    # Train the model
    for i,data in enumerate(dataloader):
        # logger.info("Training batch no. " + str(i) + " of size 4")
        images,captions,lengths = data
        images,captions = Variable(images.cuda(),volatile=True),Variable(captions.cuda())
        
        targets = pack_padded_sequence(captions,lengths,batch_first=True)[0]

        decoder.zero_grad()
        encoder.zero_grad()

        features = encoder(images)
        outputs = decoder(features,captions,lengths)
        loss = criterion(outputs,targets)

        if i%5000 == 0:
            logger.info("Epoch is " + str(epoch) +  "Loss is " + str(loss.data[0]) + " of batch number " + str(i))
        loss.backward()
        optimizer.step()
        
    torch.save(encoder.state_dict(), 'encoder.tar')
    torch.save(decoder.state_dict(),'decoder.tar')


