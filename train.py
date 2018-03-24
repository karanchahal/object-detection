import torch
from data.show_and_tell.dataset import get_image_ids,coco,coco_caps,get_ann_ids,get_test_train_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.show_and_tell.DataLoader import CocoDataset, RandomCrop, Rescale, collate_fn
from text.vocab import WordModel
import coloredlogs, logging
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torchvision.models.resnet import resnet34
from model.show_and_tell.Encoder import Encoder
from model.show_and_tell.Decoder import Decoder
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import torch.nn as nn
import numpy as np
# logging settings



# Create a logger object.
logger = logging.getLogger(__name__)

use_cuda = torch.cuda.is_available()

batch_size = 4
dataset_size = 16

annIds = get_ann_ids()
imgIds = get_image_ids()
train_ids, val_ids = get_test_train_split(annIds,percentage=0.05)
train_ids = train_ids[:1000]
logger.warning("Loading Dataset")
word_model = WordModel()
word_model.load(filename="model_logs/word_model.pkl")

image_transform = transforms.Compose([
        Rescale(250),
        RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])

train_dataset = CocoDataset(annIds=train_ids,
                    coco=coco,
                    coco_caps=coco_caps,
                    word_model=word_model,
                    transform=image_transform
                )
train_dataloader = torch.utils.data.DataLoader(
                                        train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=4,
                                        collate_fn=collate_fn
                                    )

val_dataset = CocoDataset(annIds=val_ids,
                    coco=coco,
                    coco_caps=coco_caps,
                    word_model=word_model,
                    transform=image_transform
                )
val_dataloader = torch.utils.data.DataLoader(
                                        val_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=4,
                                        collate_fn=collate_fn
                                    )




logger.warning("Starting training loop")
'''Training Loop'''

print('Number of examples', word_model.vocab.examples)
print('Number of words', len(word_model.vocab.id2word))

logger.warning("Loading the model")

encoder = Encoder()
decoder = Decoder(vocab_size=word_model.vocab.length())

logger.info('CUDA is ' + str(use_cuda))
if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)
num_epochs = 2

# encoder.load_state_dict(torch.load('./encoder.tar'))
# decoder.load_state_dict(torch.load('./decoder.tar'))

logger.warning('Loading weights')

logger.warning('Training Started')
num_examples = len(train_dataloader)
main_loss = 0

for epoch in range(num_epochs):
    # Train the model
    running_loss = 0.0
    total_step = len(train_dataloader)
    for i,data in enumerate(train_dataloader):
        # logger.info("Training batch no. " + str(i) + " of size 4")
        images,captions,lengths = data
       
        if use_cuda:
            images,captions = Variable(images.cuda(),volatile=True),Variable(captions.cuda())
        else:
            images,captions = Variable(images),Variable(captions)
        
        targets = pack_padded_sequence(captions,lengths,batch_first=True)[0]

        decoder.zero_grad()
        encoder.zero_grad()

        features = encoder(images)
        outputs = decoder(features,captions,lengths)
        loss = criterion(outputs,targets)
        running_loss += loss.data[0]
        if i%100 == 0:
            logger.info('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 
        
        loss.backward()
        optimizer.step()
    
    # evaluate(encoder,decoder,val_dataloader)
    torch.save(encoder.state_dict(), 'encoder.tar')
    torch.save(decoder.state_dict(),'decoder.tar')


