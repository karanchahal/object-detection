import os
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import coloredlogs, logging
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
logger = logging.getLogger(__name__)

PROJECT_DIR = '/home/karan/attention-caption/'
DATASET_DIR = '/home/karan/coco/images/'
use_cuda = torch.cuda.is_available()

class CocoDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, annIds, coco, coco_caps, word_model=None, transform=None,create_vocab=False):
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
        self.coco_caps = coco_caps
        self.annIds = annIds
        self.word_model = word_model
        self.transform = transform
        self.create_vocab = create_vocab

    def write_to_log(self,sentence,filename):
        logger.error(str(sentence) + ' writing to log')
        self.word_err_file = open(filename, "a+")
        self.word_err_file.write(sentence + '\n')
        

    def __len__(self):
        return len(self.annIds)

    def __getitem__(self, idx):
        logger.warning("Generating sample of annotation id: " + str(self.annIds[idx]))
        
        ann_id = self.annIds[idx]
        caption = self.coco_caps.anns[ann_id]['caption']
        img_id = self.coco_caps.anns[ann_id]['image_id']
        path = self.coco_caps.loadImgs(img_id)[0]['coco_url']
       
        # create vocab
        if self.create_vocab:
            return caption

        # Create caption as list of ids
        caption = self.word_model.parse(caption)

        # Create image
        image = torch.Tensor(np.zeros((3,224,224)))

        try:
            image = io.imread(path)
        except Exception as e:
            print(e)
            self.write_to_log(path,filename=PROJECT_DIR + 'errors/image_error.log')
            return image,caption
            
        try:
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(e)
            # Temporary to get all vocab characteristics
            logger.error('Image of size ' + str(image.shape) + ' failed to rescale')
            image = torch.Tensor(np.zeros((3,224,224)))
            self.write_to_log(path,filename=PROJECT_DIR + 'errors/image_error.log')
        

        return image,caption

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        
        if len(image.shape) < 3:
            logger.error('Converting to RGB from grayscale')
            image = self.add_channels(image)
            print(image.shape)
        
        if image.shape[0] < self.output_size[0] or image.shape[1] < self.output_size[1]:
            logger.error('Rescaling image to be bigger')
            image = self.rescale(image)
            print(image.shape)

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image

    def add_channels(self,image):
        logger.warning('Adding channels to grayscale image')
        image = np.stack((image,)*3)
        return np.transpose(image,(1,2,0))

    def rescale(self,image):
        logger.warning('Rescaling for small images')
        logger.warning(str(sentence) + ' writing to log')
        return image.reshape((250,250,3))

def collate_fn(data):
    ''' input: tuple of images and captions
        output: images tensor of shape (3,224,224), captions of padded length,lengths
    '''
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    
    # Merge images
    images = torch.stack(images,0)
    
    #Merge captions
    
    lengths = [len(cap) for cap in captions]

    targets = torch.zeros(len(captions), max(lengths)).long() # adding padding token to everything
    
    for i,cap in enumerate(captions):
        end = lengths[i]
        targets[i,:end] = cap[:end]
    
    return images,targets,lengths
