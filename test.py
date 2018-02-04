import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.DataLoader import CocoDataset, RandomCrop
import coloredlogs, logging
import numpy as np

# image_transform = transforms.Compose([
#         RandomCrop(224),
#         transforms.ToTensor()
#     ])

# with open('errors/image_error.log','rb') as myfile:
#     content = myfile.readlines()


# for line in content:
#     print(line)
#     line = line.decode('UTF-8')
#     line = str(line[:-1])
#     image = io.imread(line)

#     if len(image.shape) < 3:
#         image = np.stack((image,)*3)
#         image = image.transpose((1,2,0))
#     else:
#         print(image.shape)
#         image.resize((250,250,3))
#     image = image_transform(image)
#     print(image.size())

import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data.DataLoader import CocoDataset, RandomCrop
from text.vocab import WordModel
import coloredlogs, logging


vocab = pickle.load(open('models/word_model.pkl','rb'))
print(vocab.id)
print(vocab.examples)