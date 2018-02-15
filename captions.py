import torch
from data.dataset import get_image_ids,coco,coco_caps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.DataLoader import CocoDataset, RandomCrop
from text.vocab import WordModel
import coloredlogs, logging

# logging settings

# Create a logger object.
logger = logging.getLogger(__name__)

dataset_size = 16

imgIds = get_image_ids()
logger.warning("Loading Dataset")

'''Dataset Loading'''

word_model = WordModel()
word_model.load(filename="model_logs/word_model.pkl")

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

print('Number of examples', word_model.vocab.examples)
print('Number of words', len(word_model.vocab.id2word))



for i,data in enumerate(dataloader):
    logger.info("Training batch no. " + str(i) + " of size 4")
    images, captions = data['image'], data['captions']
    for j,caption_batch in enumerate(word_model.captionloader(captions)):
        print(j)
        print(len(caption_batch))
        print(caption_batch[0].size())
    break
