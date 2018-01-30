import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import coloredlogs, logging

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
            sample = self.transform(sample)

        return sample