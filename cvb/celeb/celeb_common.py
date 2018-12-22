import os
import numpy as np
from glob import glob
from PIL import Image

import torch
from torchvision import transforms
import torchvision.datasets as dset

import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        print("Found {} images in subfolders of: {}".format(len(imgs), root))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_loader(root, batch_size, scale_size, num_workers=2, shuffle=True):
    # dataset_name = os.path.basename(root)
    # image_root = os.path.join(root, 'splits', split)
    image_root = root

    dataset = ImageFolder(root=image_root, transform=transforms.Compose([
        transforms.CenterCrop(160),
        transforms.Scale(scale_size),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers))
    data_loader.shape = [int(num) for num in dataset[0][0].size()]

    return data_loader


class CelebDecoder(nn.Module):
    def __init__(self, nz, nc, ngf, act_out = None):
        super(CelebDecoder, self).__init__()
        self.act_out = act_out
        self.input_dim = nz

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
        )

    def forward(self, input):
        output = self.main(input.view(input.shape[0], input.shape[1], 1, 1))
        if self.act_out is not None:
            output =  self.act_out(output)
        return output
