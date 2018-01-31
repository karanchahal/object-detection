from __future__ import print_function, division
import torch
from pycocotools.coco import COCO
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import coloredlogs, logging
from DataLoader import CocoDataset, RandomCrop
# logging settings

# Create a logger object.
logger = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG')
coloredlogs.install(fmt='%(asctime)s,%(msecs)03d %(levelname)s %(message)s')
# Dataset Location variables

dataDir='../../coco/' # path to dataset folder
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
annCapFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)

# Loading annotations
logger.warning("Loading annotations images")
coco=COCO(annFile)
logger.warning("Loading annotations captions")
coco_caps=COCO(annCapFile)

def get_image_ids(index):
    logger.warning("Generating image ids")
    # get all category ids
    catIds = coco.getCatIds()
    imgIds = []
    # appending image ids of each category
    for id in catIds:
       imgIds += coco.getImgIds(catIds=id)
    return imgIds
    


image_transform = transforms.Compose([
        RandomCrop(224),
        transforms.ToTensor()
    ])
    
imgIds = get_image_ids(1)
logger.warning("Instantiating Dataset Object")
dataset = CocoDataset(imgIds=imgIds,coco=coco,coco_caps=coco_caps,transform=image_transform)
dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=4, shuffle=True,
                                             num_workers=4)
for i,data in enumerate(dataloader):
    logger.info("Training batch no. " + str(i) + " of size 4")
    images, captions = data['image'], data['captions']
    if(i == 3):
        break

