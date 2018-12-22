from __future__ import print_function
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

from cvb.common.cmd_args import cmd_args
from cvb.common.pytorch_util import weights_init

from cvb.common.my_modules import MyModule, MyLinear

def binary_cross_entropy(pred, target, size_average=True):
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)
    loss = -torch.sum( target * torch.log( pred + 1e-20 ) + (1.0 - target) * torch.log( 1.0 - pred + 1e-20 ) )
    if size_average:
        return loss / pred.size()[0]
    else:
        return loss


def loss_function(recon_x, x, mu, logvar):
    BCE = binary_cross_entropy(recon_x, x, size_average=False)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (BCE + KLD) / x.shape[0]


class AbstractGaussianEncoder(object):

    def encode(self, x):
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class FCGaussianEncoder(nn.Module, AbstractGaussianEncoder):
    def __init__(self, input_dim, latent_dim):
        super(FCGaussianEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim        

        self.fc1 = nn.Linear(input_dim, 400)

        self.fc2 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(400, latent_dim)

        weights_init(self)

    def encode(self, x):
        h2 = F.relu( self.fc1(x.view(-1, self.input_dim)) )
        return self.fc2(h2), self.fc3(h2)

    def forward(self, x):
        return AbstractGaussianEncoder.forward(self, x)

        
class FCDecoder(nn.Module):
    def __init__(self, latent_dim, input_dim, act_out = F.sigmoid):
        super(FCDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.act_out = act_out
    
        self.fc1 = nn.Linear(latent_dim, 400)
        self.fc2 = nn.Linear(400, input_dim)
        weights_init(self)

    def forward(self, z):
        h1 = F.relu(self.fc1(z))

        return self.act_out(self.fc2(h1))


class FC3LayerDecoder(nn.Module):
    def __init__(self, latent_dim, input_dim, act_out = F.sigmoid):
        super(FC3LayerDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.act_out = act_out
    
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, input_dim)
        weights_init(self)

    def forward(self, z):
        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc2(h1))
        
        return self.act_out(self.fc3(h2))
