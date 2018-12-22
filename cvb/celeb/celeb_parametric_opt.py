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
from cvb.common.my_modules import MyLinear, MyModule, MyConvTranspose2d, MyBatchNorm2d, MyConv2d, MySequential, MyReLU
from cvb.common.mirror_descent import optimize_func
import cvb.common.neural_optim as neural_optim

from cvb.celeb.celeb_common import get_loader, CelebDecoder
from cvb.common.vae_common import loss_function

celeb_loader = get_loader(cmd_args.dropbox + '/data/celebA', 100, 64)
test_loader = get_loader(cmd_args.dropbox + '/data/celebA', 100, 64)

class MyCelebEncoder(MyModule):
    def __init__(self, nc, ndf, latent_dim):
        super(MyCelebEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.main = MySequential(
            # input is (nc) x 64 x 64
            MyConv2d(nc, ndf, 5, 2, 2, bias=False),
            MyReLU(inplace=True),
            # state size. (ndf) x 32 x 32
            MyConv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
            MyBatchNorm2d(ndf * 2),
            MyReLU(inplace=True),
            # state size. (ndf*2) x 16 x 16
            MyConv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
            MyBatchNorm2d(ndf * 4),
            MyReLU(inplace=True),
            MyConv2d(ndf * 4, ndf * 4, 5, 2, 2, bias=False),
            MyBatchNorm2d(ndf * 4),
            MyReLU(inplace=True),
        )

        self.fc2 = MyLinear(ndf * 4 * 4 * 4, latent_dim)
        self.fc3 = MyLinear(ndf * 4 * 4 * 4, latent_dim)
        weights_init(self)

    def encode(self, x):
        h1 = self.main(x).view(x.shape[0], -1)
        return self.fc2(h1), self.fc3(h1)

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


def train(epoch):
    encoder.train()
    train_loss = 0
    pbar = tqdm(celeb_loader)
    num_mini_batches = 0
    for (data, _) in pbar:
        data = Variable(data)
        if cmd_args.ctx == 'gpu':
            data = data.cuda()

        optimizer.zero_grad()
        inner_opt.zero_grad()

        fz = lambda z, mu, logvar: loss_function(decoder(z).view(data.shape[0], -1), data.view(data.shape[0], -1), mu, logvar)
        best_z, mu, logvar = optimize_func(data, encoder, fz, inner_opt, nsteps = cmd_args.unroll_steps)
        
        loss = fz(best_z, mu, logvar)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        pbar.set_description('minibatch loss: %.4f' % loss.item())
        num_mini_batches += 1
    print('Epoch %d, average loss %.4f' % (epoch, train_loss / num_mini_batches))
    return train_loss / num_mini_batches


def test(epoch):
    encoder.eval()
    inner_opt.set_freeze_flag(True)
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if cmd_args.ctx == 'gpu':
            data = data.cuda()
        data = Variable(data)

        fz = lambda z, mu, logvar: loss_function(decoder(z).view(data.shape[0], -1), data.view(data.shape[0], -1), mu, logvar)
        if cmd_args.unroll_test:
            inner_opt.zero_grad()
            bak_dict = encoder.diff_var_dict()
            best_z, mu, logvar = optimize_func(data, encoder, fz, inner_opt, nsteps = cmd_args.unroll_steps)
            encoder.load_diff_var_dict(bak_dict)
        else:
            best_z, mu, logvar = encoder(data)

        loss = fz(best_z, mu, logvar)
        recon_batch = decoder(best_z)
        test_loss += loss.item() * data.shape[0]                    
            
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                recon_batch.view(cmd_args.batch_size, 3, 64, 64)[:n]])
            save_image(comparison.data.cpu(),
                    '%s/vae_reconstruction_' % cmd_args.save_dir + str(epoch) + '.png', nrow=n)
        break
    inner_opt.set_freeze_flag(False)

def get_init_posterior(data):
    fz = lambda z, mu, logvar: loss_function(decoder(z).view(data.shape[0], -1), data.view(data.shape[0], -1), mu, logvar)
    if cmd_args.unroll_test:
        inner_opt.zero_grad()
        bak_dict = encoder.diff_var_dict()
        best_z, mu, logvar = optimize_func(data, encoder, fz, inner_opt, nsteps = cmd_args.unroll_steps)
        encoder.load_diff_var_dict(bak_dict)
    else:
        best_z, mu, logvar = encoder(data)
    return best_z, mu, logvar

def do_vis(epoch):
    encoder.eval()
    inner_opt.set_freeze_flag(True)
    for i, (data, _) in tqdm(enumerate(test_loader)):
        t = data[:64].view(64, 3, 64, 64)
        save_image(t.cpu(), '%s/dataset-%d-' % (cmd_args.save_dir, i) + str(epoch) + '.png', nrow=8)
        if cmd_args.ctx == 'gpu':
            data = data.cuda()

        best_z, mu, logvar = get_init_posterior(data)
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
        encoder.train()

        z = encoder.reparameterize(mu, logvar)
        sample = decoder(z)[:64].view(64, 3, 64, 64)
        save_image(sample.data.cpu(),
                    '%s/posterior-%d-' % (cmd_args.save_dir, i) + str(epoch) + '.png', nrow=8)
 
        encoder.eval()
        if i + 1 >= cmd_args.vis_num:
            break


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    encoder = MyCelebEncoder(3, 64, latent_dim=cmd_args.latent_dim)
    decoder = CelebDecoder(cmd_args.latent_dim, 3, 64, act_out=F.sigmoid)
    if cmd_args.ctx == 'gpu':
        encoder.cuda()
        decoder.cuda()

    if cmd_args.init_model_dump is not None:
        encoder.load_diff_var_dict(torch.load(cmd_args.init_model_dump + '.encoder'))
        decoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.decoder'))

    optimizer = optim.Adam(decoder.parameters(), lr=cmd_args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-5)

    opt_class = getattr(neural_optim, 'Neural' + cmd_args.inner_opt)
    inner_opt = opt_class(encoder, lr = cmd_args.mda_lr)
    
    if cmd_args.vis_num > 0:
        epoch = cmd_args.init_model_dump.split('-')[-1]
        do_vis(epoch)
        sys.exit()
 
    for epoch in range(cmd_args.num_epochs):        
        train_loss = train(epoch)
        scheduler.step(train_loss)
        test(epoch)
        torch.save(decoder.state_dict(), cmd_args.save_dir + '/epoch-%d.decoder' % epoch)
        torch.save(encoder.diff_var_dict(), cmd_args.save_dir + '/epoch-%d.encoder' % epoch)

