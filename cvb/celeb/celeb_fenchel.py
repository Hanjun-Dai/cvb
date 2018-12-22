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
from torchvision import transforms, utils

from cvb.common.cmd_args import cmd_args
from cvb.common.pytorch_util import weights_init
from cvb.common.bregman_divergence import EuclideanDist
from cvb.common.mirror_descent import optimize_variable, optimize_func
from cvb.celeb.celeb_common import get_loader, CelebDecoder
from cvb.common.vae_common import binary_cross_entropy
import cvb.common.neural_optim as neural_optim

celeb_loader = get_loader(cmd_args.dropbox + '/data/celebA', 100, 64)
test_loader = get_loader(cmd_args.dropbox + '/data/celebA', 100, 64)

class Encoder(nn.Module):
    def __init__(self, nc, ndf, latent_dim, out_dim = None):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        if out_dim is None:
            self.out_dim = latent_dim
        else:
            self.out_dim = out_dim
        
        self.x2h = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 5, 2, 2, bias=False),
            nn.ReLU(inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, 5, 2, 2, bias=False),
            nn.ReLU(inplace=True),
        )

        self.z2h = nn.Sequential(
        #    nn.Conv2d(self.latent_dim, 4 * ndf, 1),
      #      nn.ReLU(inplace=True),
    #        nn.Conv2d(4 * ndf, 4 * ndf, 1),
  #          nn.ReLU(inplace=True),
            nn.Linear(cmd_args.latent_dim, ndf * 4 * 4 * 4)
        )
        self.fc2 = nn.Linear(ndf * 4 * 4 * 4 * 2, 400)
        self.fc3 = nn.Linear(400, self.out_dim)

        weights_init(self)

    def forward(self, x, xi):
        if len(xi.shape) == 2:
            xi = xi.view(xi.shape[0], xi.shape[1], 1, 1)
        hx = self.x2h(x).view(x.shape[0], -1)
        hz = self.z2h(xi.view(xi.shape[0], -1)).view(xi.shape[0], -1)
        x = torch.cat((hx, hz), dim=1)

        h1 = F.relu(self.fc2(x))
        out = self.fc3(h1)

        return out

def log_score(data, z):
    s = torch.mean( nu(data, z) )
    prior_z = torch.Tensor(z.shape[0], cmd_args.latent_dim).normal_(0, 1)
    if cmd_args.ctx == 'gpu':
        prior_z = prior_z.cuda()

    opt_nu.zero_grad()
    loss_nu = -torch.mean( nu(data, z.detach()) ) + torch.mean( torch.exp( nu(data, prior_z) ) )
    loss_nu.backward()
    opt_nu.step()

    return s, loss_nu.item()

def train(epoch):
    train_loss = 0
    pbar = tqdm(celeb_loader)
    num_mini_batches = 0
    for (data, _) in pbar:
        if cmd_args.ctx == 'gpu':
            data = data.cuda()

        optimizer.zero_grad()
        fz = lambda z: binary_cross_entropy(decoder(z).view(data.shape[0], -1), data.view(data.shape[0], -1)) + log_score(data, z)[0]
        xi = torch.Tensor( data.shape[0], cmd_args.latent_dim ).normal_()
        if cmd_args.ctx == 'gpu':
            xi = xi.cuda()
        
        z_init = encoder(data, xi)
        best_z = optimize_variable(z_init, fz, EuclideanDist, nsteps = cmd_args.unroll_steps)
                
        optimizer.zero_grad()
        loss = fz(best_z)
        loss.backward()

        prior_z = torch.Tensor(data.shape[0], cmd_args.latent_dim).normal_(0, 1)
        if cmd_args.ctx == 'gpu':
            prior_z = prior_z.cuda()
        cur_loss = loss.item() + torch.mean( torch.exp( nu(data, prior_z).detach() ) ).item()

        train_loss += cur_loss
        optimizer.step()

        pbar.set_description('minibatch loss: %.4f' % cur_loss)
        num_mini_batches += 1
    msg = 'train epoch %d, average loss %.4f' % (epoch, train_loss / num_mini_batches)
    print(msg)


def test(epoch):
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        bak_nu_dict = nu.state_dict()
        bak_opt_dict = opt_nu.state_dict()
        
        if cmd_args.ctx == 'gpu':
            data = data.cuda()
        data = Variable(data)
        with torch.no_grad():
            prior_z = torch.Tensor(data.shape[0], cmd_args.latent_dim).normal_(0, 1)
            if cmd_args.ctx == 'gpu':
                prior_z = prior_z.cuda()
            cur_loss = torch.mean( torch.exp( nu(data, prior_z) ) ).item()

        fz = lambda z: binary_cross_entropy(decoder(z).view(data.shape[0], -1), data.view(data.shape[0], -1)) + log_score(data, z)[0]
        xi = torch.Tensor( data.shape[0], cmd_args.latent_dim ).normal_()
        if cmd_args.ctx == 'gpu':
            xi = xi.cuda()
        z_init = encoder(data, xi)

        if cmd_args.unroll_test:
            best_z = optimize_variable(z_init, fz, EuclideanDist, nsteps = cmd_args.unroll_steps)
        else:
            best_z = z_init

        loss = fz(best_z)
        nu.load_state_dict(bak_nu_dict)
        opt_nu.load_state_dict(bak_opt_dict)
        recon_batch = decoder(best_z)

        test_loss += (loss.item() + cur_loss) * data.shape[0]            
            
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                recon_batch.view(cmd_args.batch_size, 3, 64, 64)[:n]])
            save_image(comparison.data.cpu(),
                    '%s/vae_reconstruction_' % cmd_args.save_dir + str(epoch) + '.png', nrow=n)
        break

def do_vis(epoch):
    for i, (data, _) in tqdm(enumerate(test_loader)):
        t = data[:64].view(64, 3, 64, 64)
        save_image(t.cpu(), '%s/dataset-%d-' % (cmd_args.save_dir, i) + str(epoch) + '.png', nrow=8)
        if cmd_args.ctx == 'gpu':
            data = data.cuda()
        bak_nu_dict = nu.state_dict()
        bak_opt_dict = opt_nu.state_dict()

        fz = lambda z: binary_cross_entropy(decoder(z).view(data.shape[0], -1), data.view(data.shape[0], -1)) + log_score(data, z)[0]
        xi = torch.Tensor( data.shape[0], cmd_args.latent_dim ).normal_()
        if cmd_args.ctx == 'gpu':
            xi = xi.cuda()
        z_init = encoder(data, xi)

        if cmd_args.unroll_test:
            best_z = optimize_variable(z_init, fz, EuclideanDist, nsteps = cmd_args.unroll_steps)
        else:
            best_z = z_init

        recon_batch = decoder(best_z)

        n = min(data.size(0), 8)
        comparison = torch.cat([data[:n],
                                recon_batch.view(cmd_args.batch_size, 3, 64, 64)[:n]])
        save_image(comparison.data.cpu(),
                    '%s/recon-%d-' % (cmd_args.save_dir, i) + str(epoch) + '.png', nrow=n)
        z = torch.Tensor(64, cmd_args.latent_dim).normal_(0, 1)
        if cmd_args.ctx == 'gpu':
            z = z.cuda()
        sample = decoder(z).view(64, 3, 64, 64)
        save_image(sample.data.cpu(),
                    '%s/prior-%d-' % (cmd_args.save_dir, i) + str(epoch) + '.png', nrow=8)

        z = z_init
        sample = decoder(z)[:64].view(64, 3, 64, 64)
        save_image(sample.data.cpu(),
                    '%s/posterior-%d-' % (cmd_args.save_dir, i) + str(epoch) + '.png', nrow=8)
        nu.load_state_dict(bak_nu_dict)
        opt_nu.load_state_dict(bak_opt_dict)
        if i + 1 >= cmd_args.vis_num:
            break

if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    encoder = Encoder(3, 64, latent_dim=cmd_args.latent_dim)
    decoder = CelebDecoder(cmd_args.latent_dim, 3, 64, act_out=F.sigmoid)
    nu = Encoder(3, 64, latent_dim=cmd_args.latent_dim, out_dim=1)

    if cmd_args.ctx == 'gpu':
        encoder.cuda()
        decoder.cuda()
        nu.cuda()
    
    optimizer = optim.Adam(chain( encoder.parameters(), decoder.parameters() ) , lr=cmd_args.learning_rate)
    opt_nu = optim.RMSprop( nu.parameters(), lr = cmd_args.mda_lr)
 
    if cmd_args.init_model_dump is not None:
        encoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.encoder'))
        decoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.decoder'))
        nu.load_state_dict(torch.load(cmd_args.init_model_dump + '.nu'))
        optimizer.load_state_dict(torch.load(cmd_args.init_model_dump + '.optimizer'))
        opt_nu.load_state_dict(torch.load(cmd_args.init_model_dump + '.opt_nu'))

    if cmd_args.vis_num > 0:
        epoch = cmd_args.init_model_dump.split('-')[-1]
        do_vis(epoch)
        sys.exit()
 
    for epoch in range(cmd_args.num_epochs):        
        train_loss = train(epoch)
        # scheduler.step(train_loss)
#        test(epoch)
        if epoch % cmd_args.epoch_save != 0:
            continue
        torch.save(decoder.state_dict(), cmd_args.save_dir + '/epoch-%d.decoder' % epoch)
        torch.save(encoder.state_dict(), cmd_args.save_dir + '/epoch-%d.encoder' % epoch)
        torch.save(nu.state_dict(), cmd_args.save_dir + '/epoch-%d.nu' % epoch)
        torch.save(optimizer.state_dict(), cmd_args.save_dir + '/epoch-%d.optimizer' % epoch)
        torch.save(opt_nu.state_dict(), cmd_args.save_dir + '/epoch-%d.opt_nu' % epoch)

