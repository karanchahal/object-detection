import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
class CocoDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, imgIds, coco, transform=None):
        """
        Args:
            imgIds (string): image ids of the images.
            coco (object): Coco data helper object.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.coco = coco
        self.imgIds = imgIds
        self.transform = transform

    def __len__(self):
        return len(self.imgIds)

    def __getitem__(self, idx):
        img = self.coco.loadImgs(self.imgIds[idx])[0]
        print(img)
        image = io.imread(img['coco_url'])
        print(image.shape)
        if self.transform:
            sample = self.transform(sample)

        return image