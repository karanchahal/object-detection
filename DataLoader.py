import os
import cv2
import urllib
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
from PIL import Image
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
    
    return data




def retrieveImage(path):
  '''
  
  Gets an image using OpenCV given the file path.
  Arguments
    1. URL/PATH : (string) - The path of the file, can also be a URL
  Returns 
    1. image - (numpy array) - The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
  
  '''
  
  assert(isinstance(path,str)), "The URL should be of type string, you are passing a type %s in the retrieveImage function"%(type(path))

  flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
  if not os.path.exists(path) and not str(path).startswith("http"):
      raise OSError('No such file or directory: {}'.format(path))
  elif os.path.isdir(path) and not str(path).startswith("http"):
      raise OSError('Is a directory: {}'.format(path))
  else:
      try:
          if str(path).startswith("http"):
              req = urllib.request.urlopen(str(path))
              image = np.asarray(bytearray(req.read()), dtype="uint8")
              im = cv2.imdecode(image, flags).astype(np.float32)/255
          else:
              im = cv2.imread(str(path), flags).astype(np.float32)/255
          if im is None: raise OSError(f'File not recognized by opencv: {path}')
          return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
      except Exception as e:
          raise OSError('Error handling image at: {}'.format(path)) from e
          

def resizeImage(image,size):
  '''
  TODO
  '''
  
  h,w,_ = image.shape
  ratio = float(h/w)
  
  if min(h,w) >= size:
    interpolation = cv2.INTER_AREA
  else:
    interpolation = cv2.INTER_LINEAR 
    
  if w == min(h,w):
    target_height = int(ratio*size)
    return cv2.resize(image,(size,target_height), interpolation=interpolation)
  else: 
    target_width = int(size/ratio)
    return cv2.resize(image,(target_width,size),interpolation=interpolation)

def displayImage(image):
    plt.axis('off')
    plt.imshow(image)
    plt.show()
  
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
          image = retrieveImage(img_node['coco_url'])
          old_h,old_w,_ = image.shape
          image = resizeImage(image,224)
          new_h,new_w,_ = image.shape
          width_offset = float(new_w/old_w)
          height_offset = float(new_h/old_h)
          plotAllBoundingBoxes(image,img_node['id'], width_offset,height_offset)


visualiseDataSet()
