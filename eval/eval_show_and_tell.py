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
from scipy import misc
from skimage import io, transform
from skimage.viewer import ImageViewer
# logging settings

''' Need to paste it in root and run , also data loader needs to be configured in show and tell settings'''


def bleu(reference,candidate):
    cc = SmoothingFunction()
    score = sentence_bleu(reference, candidate)
    return score

# Create a logger object.
logger = logging.getLogger(__name__)

use_cuda = torch.cuda.is_available()

batch_size = 1
dataset_size = 16


annIds = get_ann_ids()
imgIds = get_image_ids()
train_ids, val_ids = get_test_train_split(annIds,percentage=0.1)

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




def evaluate(encoder,decoder,val_dataloader):
    
    logger.warning('Model is being evaluated with ' + str(len(val_dataloader)*batch_size) + " examples")
    
    running_loss = 0.0
    running_score = 0.0
    num_examples = len(train_dataloader)
    logger.info('The number of examples are : ' + str(num_examples))

    for i,data in enumerate(train_dataloader):
        images,captions,lengths = data

        if use_cuda:
            images,captions = Variable(images.cuda(),volatile=True),Variable(captions.cuda())
        else:
            images,captions = Variable(images,volatile=True),Variable(captions)
        
        targets,pad_lengths = pack_padded_sequence(captions,lengths,batch_first=True)

        decoder.zero_grad()
        encoder.zero_grad()
        

        # features = encoder(images)
        # ids = decoder(features)

        features = encoder(images)
        outputs = decoder(features,captions,lengths)

        targets = pad_packed_sequence((targets,pad_lengths),batch_first=True)[0]
        outputs = pad_packed_sequence((outputs,pad_lengths),batch_first=True)[0]

        _, outputs = outputs.data.topk(1)
        outputs = torch.squeeze(outputs,2)
        
        if use_cuda:
            targets_in_sentence = word_model.to_sentence(targets.data.cpu().numpy())
            outputs_in_sentence = word_model.to_sentence(outputs.cpu().numpy())
        else:
            targets_in_sentence = word_model.to_sentence(targets.data.numpy())
            outputs_in_sentence = word_model.to_sentence(outputs.numpy())
        
        img = images.data.cpu().numpy()
        print(targets_in_sentence[0])
        print(outputs_in_sentence[0])

        break

    
       
    
    # logger.warning("The bleu score is " + str(running_score) + " and loss is " + str(running_loss))

def sample(encoder,decoder,filepaths):
    
    for path in filepaths:
        image = io.imread(path)
        image = image_transform(image)
        print(image.size())
        image = Variable(image.cuda().unsqueeze(0))
        print(image.size())
        features = encoder(image)
        print(features.size())
        ids = decoder.sample(features)
        
        print(word_model.to_sentence_from_ids(ids.cpu().data))

        

print('Number of examples', word_model.vocab.examples)
print('Number of words', len(word_model.vocab.id2word))

logger.warning("Loading the model")
encoder = Encoder()
decoder = Decoder(vocab_size=word_model.vocab.length())

logger.info('CUDA is ' + str(use_cuda))

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
logger.warning("Loading weights")
encoder.load_state_dict(torch.load('./encoder.tar'))
decoder.load_state_dict(torch.load('./decoder.tar'))
encoder.eval()
evaluate(encoder,decoder,train_dataloader)
# sample(encoder,decoder,filepaths=['test2.jpg','test3.jpg'])
