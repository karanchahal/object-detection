import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import coloredlogs, logging
import numpy as np
logger = logging.getLogger(__name__)

class CocoDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgIds, coco, coco_caps, transform=None):
        """
        Args:
            imgIds (string): image ids of the images.
            coco (object): Coco image data helper object.
            coco_caps (object): Coco caption data helper object.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.coco = coco
        self.coco_caps = coco_caps
        self.imgIds = imgIds
        self.transform = transform

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        logger.warning("Generating sample of image id: " + str(self.imgIds[idx]))
        img = self.coco.loadImgs(self.imgIds[idx])[0]
        image = io.imread(img['coco_url'])

        capIds = self.coco_caps.getAnnIds(imgIds=img['id']);
        captions = self.coco_caps.loadAnns(capIds)
        captions = [ c['caption'] for c in captions ]
        
        sample = {'image': image, 'captions': captions}

        if self.transform:
                sample['image'] = self.transform(image)

        return sample

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

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image