import torch
from data.dataset import get_image_ids,coco,coco_caps,get_ann_ids,get_test_train_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.DataLoader import CocoDataset, RandomCrop, Rescale, collate_fn
from text.vocab import WordModel
import coloredlogs, logging
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torchvision.models.resnet import resnet34
from model.Encoder import Encoder
from model.Decoder import Decoder
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import torch.nn as nn
# logging settings


def bleu(reference,candidate):
    cc = SmoothingFunction()
    score = sentence_bleu(reference, candidate)
    return score

# Create a logger object.
logger = logging.getLogger(__name__)

use_cuda = torch.cuda.is_available()

batch_size = 4
dataset_size = 16

annIds = get_ann_ids()
imgIds = get_image_ids()
train_ids, val_ids = get_test_train_split(annIds,percentage=0.99)

logger.warning("Loading Dataset")
word_model = WordModel()
word_model.load(filename="model_logs/word_model.pkl")

image_transform = transforms.Compose([
        Rescale(250),
        RandomCrop(224),
        transforms.ToTensor(),
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




def evaluate(encoder,decoder,val_dataloader):
    
    logger.warning('Model is being evaluated with ' + str(len(val_dataloader)*batch_size) + " examples")
    
    running_loss = 0.0
    running_score = 0.0
    num_examples = len(val_dataloader)
    print(num_examples)

    for i,data in enumerate(val_dataloader):
        images,captions,lengths = data

        if use_cuda:
            images,captions = Variable(images.cuda(),volatile=True),Variable(captions.cuda())
        else:
            images,captions = Variable(images,volatile=True),Variable(captions)
        
        targets,pad_lengths = pack_padded_sequence(captions,lengths,batch_first=True)

        
        decoder.zero_grad()
        encoder.zero_grad()

        features = encoder(images)
        outputs = decoder(features,captions,lengths)
        loss = criterion(outputs,targets)
        targets = pad_packed_sequence((targets,pad_lengths),batch_first=True)[0]
        outputs = pad_packed_sequence((outputs,pad_lengths),batch_first=True)[0]

        _, outputs = outputs.data.topk(1)
        outputs = torch.squeeze(outputs)
        if use_cuda:
            targets_in_sentence = word_model.to_sentence(targets.data.cpu().numpy())
            outputs_in_sentence = word_model.to_sentence(outputs.cpu().numpy())
        else:
            targets_in_sentence = word_model.to_sentence(targets.data.numpy())
            outputs_in_sentence = word_model.to_sentence(outputs.numpy())
        

        score = bleu(targets_in_sentence, outputs_in_sentence)
        
        running_score += float(score/num_examples)
        running_loss += float(loss.data[0]/num_examples)
    
    logger.warning("The bleu score is " + str(running_score) + " and loss is " + str(running_loss))



logger.warning("Starting training loop")
'''Training Loop'''

print('Number of examples', word_model.vocab.examples)
print('Number of words', len(word_model.vocab.id2word))

logger.warning("Loading the model")
conv = resnet34(pretrained=True)
encoder = Encoder(conv)
decoder = Decoder(vocab_size=word_model.vocab.length(),batch_size=batch_size)

logger.info('CUDA is ' + str(use_cuda))
if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.fc.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)
num_epochs = 1


logger.warning('Loading weights')

logger.warning('Training Started')
for epoch in range(num_epochs):
    # Train the model
    running_loss = 0.0
    for i,data in enumerate(train_dataloader):
        # logger.info("Training batch no. " + str(i) + " of size 4")
        images,captions,lengths = data
       
        if use_cuda:
            images,captions = Variable(images.cuda()),Variable(captions.cuda())
        else:
            images,captions = Variable(images),Variable(captions)
        
        targets = pack_padded_sequence(captions,lengths,batch_first=True)[0]

        decoder.zero_grad()
        encoder.zero_grad()

        features = encoder(images)
        outputs = decoder(features,captions,lengths)
        loss = criterion(outputs,targets)

        running_loss += float(loss.data[0]/(i+1))

        if i%5000 == 0:
            logger.info("Epoch is " + str(epoch) +  " Loss is " + str(running_loss) + " of batch number " + str(i))
        
        loss.backward()
        optimizer.step()
    
    # evaluate(encoder,decoder,val_dataloader)
    torch.save(encoder.state_dict(), 'encoder.tar')
    torch.save(decoder.state_dict(),'decoder.tar')

torch.save(encoder.state_dict(), 'encoder.tar')
torch.save(decoder.state_dict(),'decoder.tar')
