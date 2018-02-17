import torch
from data.dataset import get_image_ids,coco,coco_caps, get_ann_ids
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.DataLoader import CocoDataset, RandomCrop
from text.vocab import WordModel
import coloredlogs, logging

# logging settings

# Create a logger object.
logger = logging.getLogger(__name__)

dataset_size = 16

annIds = get_ann_ids()
logger.warning("Loading Dataset")

'''Dataset Loading'''

image_transform = transforms.Compose([
        RandomCrop(224),
        transforms.ToTensor()
    ])
dataset = CocoDataset(annIds=annIds,
                    coco=coco,
                    coco_caps=coco_caps,
                    transform=image_transform,
                    create_vocab=True
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
word_model.load_embeddings()

for i,captions in enumerate(dataloader):
    logger.info("Training batch no. " + str(i) + " of size 4")
    word_model.build_vocab(captions)

print('Total examples are  ',word_model.vocab.examples)

word_model.save(filename='model_logs/word_model.pkl')