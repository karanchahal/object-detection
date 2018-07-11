import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import coloredlogs, logging
import numpy as np
import pycocotools as pycoco
from torch.nn.utils.rnn import pack_padded_sequence
from skimage.transform import resize
from skimage.viewer import ImageViewer
from data.show_and_tell.dataset import get_image_ids,coco,coco_caps,get_ann_ids,get_test_train_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
logger = logging.getLogger(__name__)


PROJECT_DIR = './'
use_cuda = torch.cuda.is_available()

class CocoDatasetObjectDetection(Dataset):
    """Object detection dataset."""

    def __init__(self, imgIds, coco, transform=None,bounding_box_mode=False):
        """
        Args:
            annIds (string): annotation ids of the captions.
            coco (object): Coco image data helper object.
            coco_caps (object): Coco caption data helper object.
            word_model (object): Vocab object to get ids of words
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.coco = coco
        self.imageIds = imgIds
        self.transform = transform
        self.bounding_box_mode = bounding_box_mode

    def write_to_log(self,sentence,filename):
        logger.error(str(sentence) + ' writing to log')
        self.word_err_file = open(filename, "a+")
        self.word_err_file.write(sentence + '\n')
        

    def __len__(self):
        return len(self.imageIds)

    def __getitem__(self, idx):
        # logger.warning("Generating sample of annotation id: " + str(self.annIds[idx]))

        img_id = self.imageIds[idx]
        image_node = self.coco.loadImgs(img_id)[0]

        return image_node


      
def collate_fn(data):
    ''' input: image nodes
        output: images tensor of shape (3,224,224), captions of padded length,lengths
    '''
#     data.sort(key=lambda x: len(x[1]), reverse=True)
    
    return data
  
  
def getBoundingBoxCoords(id):
    ''' This function returns the bounding box coordinates of an image id'''
    annIds = coco.getAnnIds(imgIds=id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coordsList = []
    classList = []
    for a in anns:
        coordsList.append(a['bbox'])
        classList.append(a['category_id'])
    
    return coordsList,classList
        
  
def visualiseDataSet():
    
    batch_size = 1
    imgIds = get_image_ids()
    train_ids, _ = get_test_train_split(imgIds,percentage=0.05)
    train_ids = train_ids[:20]
  
    train_dataset = CocoDatasetObjectDetection(
                    imgIds=train_ids,
                    coco=coco
                )
    train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=1,
                    collate_fn=collate_fn
                )

    for i,img_nodes in enumerate(train_dataloader):
        for img_node in img_nodes:
            image = io.imread(img_node['coco_url'])
            plotBiggestBoundingBox(image,img_node['id'])
    

visualiseDataSet()
