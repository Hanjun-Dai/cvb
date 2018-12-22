from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import numpy as np

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
from itertools import chain
from matplotlib import pyplot as plt

from cvb.common.cmd_args import cmd_args
from cvb.common.pytorch_util import weights_init

from cvb.common.vae_common import binary_cross_entropy

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 4)
        weights_init(self)

    def forward(self, z):
        h1 = F.elu( self.fc1(z) )
        h2 = F.elu( self.fc2(h1) )
        h3 = F.elu( self.fc3(h2) )

        return F.sigmoid( self.fc4(h3) )

def create_scatter(x_test_list, encoder, savepath=None):
    plt.figure(figsize=(5,5), facecolor='w')

    for i in range(4):
        z_out = encoder(x_test_list[i]).data.cpu().numpy()
        print(i, z_out[0])
        plt.scatter(z_out[:, 0], z_out[:, 1],  edgecolor='none', alpha=0.5)

    plt.xlim(-3, 3); plt.ylim(-3.5, 3.5)

    plt.axis('equal')
    plt.axis('off')
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close('all')


def loss_function(recon_x, x, mu, logvar):
    BCE = binary_cross_entropy(recon_x, x, size_average=False)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (BCE + KLD) / x.shape[0]
