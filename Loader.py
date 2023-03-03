from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def default_loader(path):
    return Image.open(path).convert('RGB')


class Dataset(data.Dataset):
    def __init__(self, imagePath, fineSize):
        super(Dataset, self).__init__()
        self.imagePath = imagePath
        self.image_list = [x for x in listdir(imagePath) if is_image_file(x)]
        self.fineSize = fineSize
        # self.normalize = transforms.Normalize(mean=[103.939,116.779,123.68],std=[1, 1, 1])
        # normalize = transforms.Normalize(mean=[123.68,103.939,116.779],std=[1, 1, 1])
        self.prep = transforms.Compose([
            transforms.Resize(fineSize),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
        ])

    def __getitem__(self, index):
        imgPath = os.path.join(self.imagePath, self.image_list[index])
        img = default_loader(imgPath)

        # resize
        if (self.fineSize != 0):
            w, h = img.size
            if (w > h):
                if (w != self.fineSize):
                    neww = self.fineSize
                    newh = int(h * neww / w)
                    img = img.resize((neww, newh))
            else:
                if (h != self.fineSize):
                    newh = self.fineSize
                    neww = int(w * newh / h)
                    img = img.resize((neww, newh))

        # Preprocess Images
        img = transforms.ToTensor()(img)
        return img.squeeze(0), self.image_list[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)
