import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import coloredlogs, logging
import numpy as np
logger = logging.getLogger(__name__)

PROJECT_DIR = '/home/karan/attention-caption/'

class CocoDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, annIds, coco, coco_caps, vocab, transform=None):
        """
        Args:
            annIds (string): annotation ids of the captions.
            coco (object): Coco image data helper object.
            coco_caps (object): Coco caption data helper object.
            vocab (object): Vocab object to get ids of words
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.coco = coco
        self.coco_caps = coco_caps
        self.annIds = imgIds
        self.vocab = vocab
        self.transform = transform

    def write_to_log(self,sentence,filename):
        logger.error(str(sentence) + ' writing to log')
        self.word_err_file = open(filename, "a+")
        self.word_err_file.write(sentence + '\n')
        

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        logger.warning("Generating sample of image id: " + str(self.imgIds[idx]))
        
        ann_id = self.annIds[idx]
        caption = self.coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']

        capIds = self.coco_caps.getAnnIds(imgIds=img['id']);
        captions = self.coco_caps.loadAnns(capIds)
        captions = [ c['caption'] for c in captions ]
        path = coco.loadImgs(img_id)[0]['coco_url']
        
        # Create caption as list of ids

        # Create image
        image = torch.Tensor(np.zeros((3,224,224)))

        try:
            image = io.imread(os.path.join(DATASET_PATH, path)).convert('RGB'))
        except Exception as e:
            print(e)
            self.write_to_log(img['coco_url'],filename=PROJECT_DIR + 'errors/image_error.log')
            return image,caption
            
        try:
            if self.transform:
                sample['image'] = self.transform(sample['image'])
        except Exception as e:
            print(e)
            # Temporary to get all vocab characteristics
            logger.error('Image of size ' + str(sample['image'].shape) + ' failed to rescale')
            sample['image'] = torch.Tensor(np.zeros((3,224,224)))
            self.write_to_log(img['coco_url'],filename=PROJECT_DIR + 'errors/image_error.log')

        return sample['image'],sample['captions']

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
            image = self.add_channels(image)
            print(image.shape)
        
        if image.shape[0] < self.output_size[0] or image.shape[1] < self.output_size[1]:
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
    images, captions = zip(*data)
    print(len(captions))
    # Merge images
    images = torch.stack(images,0)
    print(images.size())
    #Merge captions
    lengths = [len(c) for c in cap for cap in captions]
    captions = torch.zeros(len(captions))
    return images
