import torch
from torch.autograd import Variable
from data.dataset import get_image_ids,coco,coco_caps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.DataLoader import CocoDataset, RandomCrop
from text.vocab import WordModel
import coloredlogs, logging
from model.Encoder import Encoder
from torchvision.models.resnet import resnet34
# logging settings

# Create a logger object.
logger = logging.getLogger(__name__)

dataset_size = 16

imgIds = get_image_ids()
logger.warning("Loading Dataset")

'''Dataset Loading'''

image_transform = transforms.Compose([
        RandomCrop(224),
        transforms.ToTensor()
    ])
dataset = CocoDataset(imgIds=imgIds,
                    coco=coco,
                    coco_caps=coco_caps,
                    transform=image_transform
                )
dataloader = torch.utils.data.DataLoader(
                                        dataset,
                                        batch_size=4,
                                        shuffle=True,
                                        num_workers=4
                                    )

logger.warning("Starting training loop")
'''Training Loop'''

word_model = WordModel()
word_model.load(filename='model_logs/word_model.pkl')
conv = resnet34(pretrained=True)
encoder = Encoder(conv)

# batch_size = 4
# decoder = Decoder(vocab_size=word_model.vocab.length(),batch_size=batch_size)
# c = torch.zeros((batch_size,1)).long()
# c = Variable(c)
# c = dec(c)


for i,data in enumerate(dataloader):
    logger.info("Training batch no. " + str(i) + " of size 4")
    images, captions = Variable(data['image']), data['captions']

    captions = word_model.parse(captions)
    c = captions[0][0]
    out = encoder(images)
    # decoder(out)
    for ci in c:
        print(word_model.vocab.word(ci))
    
    print(out.size())
    break
    
