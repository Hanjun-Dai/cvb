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
from cvb.mnist.mnist_common import train_loader, test_loader, convert_data, test_importance_sampling
from cvb.mnist.mnist_common import AvbEncoder, AvbDecoder
from cvb.common.bregman_divergence import EuclideanDist
from cvb.common.mirror_descent import optimize_variable
from cvb.common.vae_common import FCDecoder, binary_cross_entropy, FCGaussianEncoder


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class Nu(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Nu, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim        

        nc = 1
        ndf = 16
        self.x_main = nn.Sequential(
            nn.Conv2d(nc, ndf, 5, 2, 2, bias=True),
            nn.Softplus(),
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=True),
            nn.Softplus(),
            nn.Conv2d(ndf * 2, ndf * 2, 5, 2, 2, bias=True),
            nn.Softplus(),
            Flatten(),
            nn.Linear(ndf * 2 * 4 * 4, 300),
            nn.Softplus(),
        )

        self.z_main = nn.Sequential(
            nn.Linear(latent_dim, 300),
            nn.Softplus(),
        )

        self.joint = nn.Sequential(
            nn.Linear(300 * 2, 300),
            nn.Softplus(),
            nn.Linear(300, 1),
        )
       
        weights_init(self)

    def forward(self, x, eps):
        hx = self.x_main(x)
        hz = self.z_main(eps)
        h_xz = torch.cat((hx, hz), dim=1)

        score = self.joint(h_xz)
        return score


def log_score(data, z, update = True):
    s = torch.mean( nu(data, z) )

    prior_z = torch.Tensor(z.shape[0], cmd_args.latent_dim).normal_(0, 1)
    if cmd_args.ctx == 'gpu':
        prior_z = prior_z.cuda()

    opt_nu.zero_grad()
    loss_nu = -torch.mean( nu(data, z.detach()) ) + torch.mean( torch.exp( nu(data, prior_z) ) )
    if update:
        loss_nu.backward()
        opt_nu.step()

    return s, loss_nu.item()


def kl_loss(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return KLD / mu.shape[0]


def train(epoch):
    encoder.train()
    train_loss = 0
    pbar = tqdm(train_loader)
    num_mini_batches = 0
    loss_list = []
    for (data, _) in pbar:
        data = convert_data(data)
     
        optimizer.zero_grad()
        # bak_nu_dict = nu.state_dict()
        fz = lambda z: binary_cross_entropy(decoder(z), data) + log_score(data, z, update=True)[0]

        z_init, mu, logvar = encoder(data)
        best_z = optimize_variable(z_init, fz, EuclideanDist, nsteps = cmd_args.unroll_steps, eps=0)
        
        kl = kl_loss(mu, logvar)
        obj = fz(best_z)
        if num_mini_batches % 1 == 0:
            loss = kl  + obj
            loss.backward()
            optimizer.step()

        recon_loss = binary_cross_entropy(decoder(best_z), data)
        vae_loss = kl.item() + recon_loss.item()
        train_loss += loss.item()
            
        pbar.set_description('vae loss: %.4f, recon: %.4f, fenchel_obj: %.4f' % (vae_loss, recon_loss.item(), obj.item()))
        loss_list.append(loss.item())
        # nu.load_state_dict(bak_nu_dict)
#        for _ in range(1):
#            log_score(data, best_z)
        num_mini_batches += 1
    msg = 'train epoch %d, average loss %.4f' % (epoch, np.mean(loss_list))
    print(msg)


def test(epoch):
    test_loss = 0
    encoder.eval()
    for i, (data, _) in tqdm(enumerate(test_loader)):
        bak_nu_dict = nu.state_dict()
        bak_opt_dict = opt_nu.state_dict()

        data = convert_data(data)

        fz = lambda z: binary_cross_entropy(decoder(z), data) + log_score(data, z)[0]
        z_init, mu, logvar = encoder(data)

        if cmd_args.unroll_test:
            best_z = optimize_variable(z_init, fz, EuclideanDist, nsteps = cmd_args.unroll_steps)
        else:
            best_z = z_init

        loss = fz(best_z) + kl_loss(mu, logvar)

        nu.load_state_dict(bak_nu_dict)
        opt_nu.load_state_dict(bak_opt_dict)
        recon_batch = decoder(best_z)
        test_loss += loss.item() * data.shape[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n].view(-1, 1, cmd_args.img_size, cmd_args.img_size),
                                  recon_batch.view(cmd_args.batch_size, 1, cmd_args.img_size, cmd_args.img_size)[:n]])
            save_image(comparison.data.cpu(),
                     '%s/gauss_fenchel_reconstruction_' % cmd_args.save_dir + str(epoch) + '.png', nrow=n)
    test_loss /= len(test_loader.dataset)
    msg = 'test epoch %d, average loss %.4f' % (epoch, test_loss)
    print(msg)
    return test_loss


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    encoder = AvbEncoder(latent_dim=cmd_args.latent_dim) #FCDecoder(cmd_args.latent_dim, 784)
    decoder = AvbDecoder(isize=cmd_args.img_size, latent_dim=cmd_args.latent_dim) #FCGaussianEncoder(784, cmd_args.latent_dim)
    nu = Nu(cmd_args.img_size * cmd_args.img_size, cmd_args.latent_dim)

    if cmd_args.ctx == 'gpu':
        encoder.cuda()
        decoder.cuda()
        nu.cuda()

    if cmd_args.init_model_dump is not None:
        encoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.encoder', map_location='cpu' if cmd_args.ctx == 'cpu' else None))
        decoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.decoder', map_location='cpu' if cmd_args.ctx == 'cpu' else None))
        nu.load_state_dict(torch.load(cmd_args.init_model_dump + '.nu', map_location='cpu' if cmd_args.ctx == 'cpu' else None))

    if cmd_args.vis_num > 0:
        epoch = cmd_args.init_model_dump.split('-')[-1]
        do_vis(epoch)
    if cmd_args.test_is:
        test_importance_sampling(test_loader, encoder, decoder)
    if cmd_args.test_is or cmd_args.vis_num:
        sys.exit()
 
    optimizer = optim.RMSprop(chain( encoder.parameters(), decoder.parameters() ) , lr=cmd_args.learning_rate)
    opt_nu = optim.RMSprop( nu.parameters(), lr = cmd_args.learning_rate)

    for epoch in range(cmd_args.num_epochs):
    #    test(epoch)
        # scheduler.step(test_loss)
        train(epoch)
        torch.save(decoder.state_dict(), cmd_args.save_dir + '/epoch-%d.decoder' % epoch)
        torch.save(encoder.state_dict(), cmd_args.save_dir + '/epoch-%d.encoder' % epoch)
        torch.save(nu.state_dict(), cmd_args.save_dir + '/epoch-%d.nu' % epoch)
    # test(epoch)
