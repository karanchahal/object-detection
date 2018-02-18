from __future__ import print_function, division
import torch
from pycocotools.coco import COCO
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import coloredlogs, logging
# logging settings

# Create a logger object.
logger = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG')
coloredlogs.install(fmt='%(asctime)s,%(msecs)03d %(levelname)s %(message)s')
# Dataset Location variables

dataDir='.' # path to dataset folder
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
annCapFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)

# Loading annotations
logger.warning("Loading annotations images")
coco=COCO(annFile)
logger.warning("Loading annotations captions")
coco_caps=COCO(annCapFile)


def get_ann_ids(sample=None):
    logger.warning("Generating annotation ids")
    # get all annotation ids
    annIds = list(coco_caps.anns.keys())
   
    if sample == None:
        # Returning the entire dataset
        return annIds
    else:
        # Returning a subset of the dataset
        return ann_ids[:sample]

def get_image_ids(sample=None):
    logger.warning("Generating image ids")
    # get all category ids
    catIds = coco.getCatIds()
    imgIds = []
    # appending image ids of each category
    for id in catIds:
       imgIds += coco.getImgIds(catIds=id)

    if sample == None:
        # Returning the entire dataset
        return imgIds
    else:
        # Returning a subset of the dataset
        return imgIds[:sample]
    

def get_test_train_split(ids):
    ''' Splits train and validation set in ratio 4:1'''
    train_len = int(len(ids)*0.8)
    print(train_len)
    train_ids = ids[:train_len]
    val_ids = ids[train_len:]
    return train_ids, val_ids
