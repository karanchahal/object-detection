from __future__ import print_function, division
import torch
from pycocotools.coco import COCO
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


dataDir='../../coco/' # path to dataset folder
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]

def get_image_and_caption(index):



    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
        
    # get all images containing given categories, select one at random

    catIds = coco.getCatIds(catNms=nms)
    imgIds = []
    print(len(catIds))
    for id in catIds:
       imgIds += coco.getImgIds(catIds=id)
    

    
get_image_and_caption(1)
