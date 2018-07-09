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
from data.rpn.anchors import generateAnchors
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
    train_ids, val_ids = get_test_train_split(imgIds,percentage=0.05)
    train_ids = train_ids[:10]
    
    # image_transform = transforms.Compose([
    #     Rescale(250),
    #     RandomCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), 
    #                          (0.229, 0.224, 0.225))
    # ])
    
  
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
        
        # get bounding box coords
        for img_node in img_nodes:
            # try:
                
            I = io.imread(img_node['coco_url'])
            w = img_node['width']
            h = img_node['height']
            print(w)
            print(h)
            anchors = generateAnchors(w,h,30)

            # Create figure and axes
            fig,ax = plt.subplots(1)
            plt.axis('off')

            # load image
            plt.imshow(I)
            print(img_node)
            coordsList,classList = getBoundingBoxCoords(img_node['id'])

            # visualise example
            colors = ['r','g','b']
            i = 0
            for i,bbox in enumerate(coordsList):
                rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=2,edgecolor=colors[i%(len(colors)-1)],facecolor='none')
                # Add the patch to the Axes
                
                hey = coco.dataset
                key = int(classList[i])-1
                print("Class for object of colour " + colors[i%(len(colors)-1)]+ " is ",coco.dataset['categories'][key]['name'])
                ax.add_patch(rect)

                i+=1

            plt.show()

                # for anchor in anchors[:]:
                #     x_center = anchor[0] - (anchor[2]/2)
                #     y_center = anchor[1] - (anchor[3]/2)
                #     rect = patches.Rectangle((anchor[0],anchor[1]),10,10,linewidth=1,edgecolor='r',facecolor='none')
                #     # Add the patch to the Axes
                #     ax.add_patch(rect)
                    # x_center = anchor[0] - (anchor[2]/2)
                    # y_center = anchor[1] - (anchor[3]/2)
                
                    # rect = patches.Rectangle((x_center,y_center),anchor[2],anchor[3],linewidth=1,edgecolor='r',facecolor='none')
                    # # Add the patch to the Axes
                    # ax.add_patch(rect)
                
            # except:
            #     logger.warning("Invalid bbox coordinates")
    

visualiseDataSet()



#todo
# create function to test the image dimensions
# create function to generate anchors for image