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
from itertools import chain

from cvb.common.cmd_args import cmd_args
from cvb.common.pytorch_util import weights_init
from cvb.common.bregman_divergence import EuclideanDist
from cvb.common.mirror_descent import optimize_variable
from cvb.common.vae_common import binary_cross_entropy

from cvb.toy_img.toy_common import loss_function, Decoder, create_scatter

# used for init z
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(4 + 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)

        self.fc4 = nn.Linear(64, 2)

        weights_init(self)

    def forward(self, x, eps):
        x = torch.cat((x, eps), dim=1)

        h1 = F.elu( self.fc1(x) ) 
        h2 = F.elu( self.fc2(h1) )
        h3 = F.elu( self.fc3(h2) )

        return self.fc4(h3)

class Nu(nn.Module):
    def __init__(self):
        super(Nu, self).__init__()

        self.fc1 = nn.Linear(4 + 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

        weights_init(self)

    def forward(self, x, z):
        x = torch.cat((x, z), dim=1)

        h1 = F.relu( self.fc1(x) ) 
        h2 = F.relu( self.fc2(h1) )
        out = self.fc3(h2)

        return out

def log_score(data, z):
    s = torch.mean( nu(data, z) )
    prior_z = torch.Tensor(z.shape[0], 2).normal_(0, 1)

    opt_nu.zero_grad()
    loss_nu = -torch.mean( nu(data, z.detach()) ) + torch.mean( torch.exp( nu(data, prior_z) ) )
    loss_nu.backward()
    opt_nu.step()

    return s, loss_nu.item()


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    encoder = Encoder()
    decoder = Decoder()
    nu = Nu()

    if cmd_args.ctx == 'gpu':
        encoder.cuda()
        decoder.cuda()
        nu.cuda()

    optimizer = optim.Adam(chain( encoder.parameters(), decoder.parameters() ) , lr=cmd_args.learning_rate)
    opt_nu = optim.RMSprop( nu.parameters(), lr = cmd_args.learning_rate)

    x_test_list = []
    for i in range(4):
        x_test_labels = torch.LongTensor( [i] * cmd_args.batch_size ).view(-1)
        x_test = torch.zeros(x_test_labels.shape[0], 4)
        x_test.scatter_(1, x_test_labels.view(-1, 1), 1)
        x_test_list.append(x_test)

    pbar = tqdm(range(100000))

    for iter in pbar: 
        encoder.train()
        idx = torch.LongTensor(cmd_args.batch_size).random_(0, 4)
        data = torch.zeros(cmd_args.batch_size, 4)
        data.scatter_(1, idx.view(-1, 1), 1)

        if cmd_args.ctx == 'gpu':
            data = data.cuda()

        fz = lambda z: binary_cross_entropy(decoder(z), data) + log_score(data, z)[0]
        xi = torch.Tensor( data.shape[0], 2 ).normal_()
        z_init = encoder(data, xi)

        best_z = optimize_variable(z_init, fz, EuclideanDist, nsteps = cmd_args.unroll_steps)
        optimizer.zero_grad()
        loss = fz(best_z)
        loss.backward()
        optimizer.step()
        
        pbar.set_description('minibatch loss: %.4f' % (loss.item()) )

        if iter % 100 == 0:
            encoder_func = lambda x: encoder(x, torch.Tensor( x.shape[0], 2 ).normal_())
            create_scatter(x_test_list, encoder_func, savepath=os.path.join(cmd_args.save_dir, '%08d.pdf' % iter))
