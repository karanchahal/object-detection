import torch
from data.dataset import get_image_ids,coco,coco_caps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.DataLoader import CocoDataset, RandomCrop
import coloredlogs, logging
# logging settings

# Create a logger object.
logger = logging.getLogger(__name__)

imgIds = get_image_ids(1)
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

for i,data in enumerate(dataloader):
    logger.info("Training batch no. " + str(i) + " of size 4")
    images, captions = data['image'], data['captions']
    if(i == 3):
        break

