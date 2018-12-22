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
from matplotlib import pyplot as plt

from cvb.common.cmd_args import cmd_args
from cvb.common.pytorch_util import weights_init
from cvb.toy_img.toy_common import loss_function, Decoder, create_scatter


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)

        self.fc4 = nn.Linear(64, 2)
        self.fc5 = nn.Linear(64, 2)

        weights_init(self)

    def encode(self, x):
        h1 = F.elu( self.fc1(x) ) 
        h2 = F.elu( self.fc2(h1) )
        h3 = F.elu( self.fc3(h2) )

        return self.fc4(h3), self.fc5(h3)

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

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    encoder = Encoder()
    decoder = Decoder()
        
    if cmd_args.ctx == 'gpu':
        encoder.cuda()
        decoder.cuda()

    optimizer = optim.Adam(chain( encoder.parameters(), decoder.parameters() ) , lr=cmd_args.learning_rate)
    
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

        optimizer.zero_grad()
        z, mu, logvar = encoder(data)
        recon_batch = decoder(z)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()

        pbar.set_description('minibatch loss: %.4f' % loss.item())

        if iter % 100 == 0:
            encoder_func = lambda x: encoder(x)[0]
            create_scatter(x_test_list, encoder_func, savepath=os.path.join(cmd_args.save_dir, '%08d.png' % iter))