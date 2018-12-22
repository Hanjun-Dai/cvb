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
from cvb.common.my_modules import MyLinear, MyModule, MyConvTranspose2d, MyBatchNorm2d, MyConv2d, MySequential, MyReLU, MySoftplus
from cvb.mnist.mnist_common import train_loader, test_loader, convert_data, AvbDecoder
from cvb.mnist.mnist_common import test_importance_sampling
from cvb.common.mirror_descent import optimize_func
from cvb.common.vae_common import loss_function, AbstractGaussianEncoder, FCDecoder
import cvb.common.neural_optim as neural_optim


class MyMnistFCEncoder(MyModule, AbstractGaussianEncoder):
    def __init__(self, input_dim, latent_dim):
        super(MyMnistFCEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim        

        self.fc1 = MyLinear(input_dim, 400)
        self.fc2 = MyLinear(400, latent_dim)
        self.fc3 = MyLinear(400, latent_dim)
        weights_init(self)

    def encode(self, x):
        h1 = F.relu( self.fc1(x.view(-1, self.input_dim)) )        
        return self.fc2(h1), self.fc3(h1)

    def forward(self, x):
        return AbstractGaussianEncoder.forward(self, x)


class MyMnistCnnEncoder(MyModule, AbstractGaussianEncoder):
    def __init__(self, latent_dim):
        super(MyMnistCnnEncoder, self).__init__()
        self.latent_dim = latent_dim
        nc = 1
        ndf = 16

        self.main = MySequential(
            MyConv2d(nc, ndf, 5, 2, 2, bias=True),
            #nn.BatchNorm2d(ndf),
            MySoftplus(),
            MyConv2d(ndf, ndf * 2, 5, 2, 2, bias=True),
            #nn.BatchNorm2d(ndf * 2),
            MySoftplus(),
            MyConv2d(ndf * 2, ndf * 2, 5, 2, 2, bias=True),
            #nn.BatchNorm2d(ndf * 2),
            MySoftplus(),
        )
        self.fc1 = MySequential(
            MyLinear(ndf * 2 * 4 * 4, 300),
            MySoftplus(),
        )
        #self.fc1_bnorm = nn.BatchNorm1d(300)
        self.fc2 = MyLinear(300, latent_dim)
        self.fc3 = MyLinear(300, latent_dim)
        weights_init(self)

    def encode(self, x):
        h1 = self.main(x).view(x.shape[0], -1)
        h1 = self.fc1(h1)
        return self.fc2(h1), self.fc3(h1)

    def forward(self, x):
        return AbstractGaussianEncoder.forward(self, x)


def train(epoch):
    train_loss = 0
    encoder.train()
    pbar = tqdm(train_loader)
    num_mini_batches = 0
    for (data, _) in pbar:
        data = convert_data(data)
     
        optimizer.zero_grad()
        inner_opt.zero_grad()
        fz = lambda z, mu, logvar: loss_function(decoder(z), data, mu, logvar)
        best_z, mu, logvar = optimize_func(data, encoder, fz, inner_opt, nsteps = cmd_args.unroll_steps)
                
        loss = fz(best_z, mu, logvar)
        loss.backward()
        
        train_loss += loss.item()
        optimizer.step()

        pbar.set_description('minibatch loss: %.4f' % loss.item())
        num_mini_batches += 1
    msg = 'train epoch %d, average loss %.4f' % (epoch, train_loss / num_mini_batches)
    print(msg)


def test(epoch):
    encoder.eval()
    inner_opt.set_freeze_flag(True)
    test_loss = 0
    for i, (data, _) in tqdm(enumerate(test_loader)):
        data = convert_data(data)
        fz = lambda z, mu, logvar: loss_function(decoder(z), data, mu, logvar)
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
                                  recon_batch.view(-1, 1, cmd_args.img_size, cmd_args.img_size)[:n]])
            save_image(comparison.data.cpu(),
                     '%s/parametric_op_reconstruction_' % cmd_args.save_dir + str(epoch) + '.png', nrow=n)
    inner_opt.set_freeze_flag(False)
    test_loss /= len(test_loader.dataset)
    msg = 'test epoch %d, average loss %.4f' % (epoch, test_loss)
    print(msg)
    return test_loss


def get_init_posterior(data):
    inner_opt.set_freeze_flag(True)
    fz = lambda z, mu, logvar: loss_function(decoder(z), data, mu, logvar)
    if cmd_args.unroll_test:
        inner_opt.zero_grad()
        bak_dict = encoder.diff_var_dict()
        best_z, mu, logvar = optimize_func(data, encoder, fz, inner_opt, nsteps = cmd_args.unroll_steps)
        encoder.load_diff_var_dict(bak_dict)
    else:
        best_z, mu, logvar = encoder(data)
    return best_z, mu, logvar


def do_vis(epoch):
    inner_opt.set_freeze_flag(True)
    for i, (data, _) in tqdm(enumerate(test_loader)):
        save_image(data[:64].data.cpu(),
                    '%s/dataset-%d-' % (cmd_args.save_dir, i) + str(epoch) + '.png', nrow=8)
        encoder.eval()
        data = convert_data(data)
        best_z, mu, logvar = get_init_posterior(data)
        recon_batch = decoder(best_z)

        n = min(data.size(0), 8)
        comparison = torch.cat([data[:n],
                                recon_batch.view(cmd_args.batch_size, 1, cmd_args.img_size, cmd_args.img_size)[:n]])
        save_image(comparison.data.cpu(),
                    '%s/recon-%d-' % (cmd_args.save_dir, i) + str(epoch) + '.png', nrow=n)

        z = torch.Tensor(64, cmd_args.latent_dim).normal_(0, 1)
        if cmd_args.ctx == 'gpu':
            z = z.cuda()
        sample = decoder(z).view(64, 1, cmd_args.img_size, cmd_args.img_size)
        save_image(sample.data.cpu(),
                    '%s/prior-%d-' % (cmd_args.save_dir, i) + str(epoch) + '.png', nrow=8)
 
        encoder.train()
        z, _, _ = encoder(data[0:64])
        if cmd_args.ctx == 'gpu':
            z = z.cuda()
        sample = decoder(z).view(64, 1, cmd_args.img_size, cmd_args.img_size)
        save_image(sample.data.cpu(),
                    '%s/posterior-%d-' % (cmd_args.save_dir, i) + str(epoch) + '.png', nrow=8)
        
        if i + 1 >= cmd_args.vis_num:
            break


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    if cmd_args.arch == 'mlp':
        decoder = FCDecoder(cmd_args.latent_dim, cmd_args.img_size * cmd_args.img_size)
        encoder = MyMnistFCEncoder(cmd_args.img_size * cmd_args.img_size, cmd_args.latent_dim)
    else:
        decoder = AvbDecoder(isize=cmd_args.img_size, latent_dim=cmd_args.latent_dim)
        encoder = MyMnistCnnEncoder(latent_dim=cmd_args.latent_dim)

    if cmd_args.init_model_dump is not None:
        encoder.load_diff_var_dict(torch.load(cmd_args.init_model_dump + '.encoder', map_location='cpu' if cmd_args.ctx == 'cpu' else None))
        decoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.decoder', map_location='cpu' if cmd_args.ctx == 'cpu' else None))

    if cmd_args.ctx == 'gpu':
        encoder.cuda()
        decoder.cuda()
    
    optimizer = optim.Adam(decoder.parameters(), lr=cmd_args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-5)

    opt_class = getattr(neural_optim, 'Neural' + cmd_args.inner_opt)
    inner_opt = opt_class(encoder, lr = cmd_args.mda_lr)

    if cmd_args.vis_num > 0:
        epoch = cmd_args.init_model_dump.split('-')[-1]
        do_vis(epoch)
    if cmd_args.test_is:
        test_importance_sampling(test_loader, get_init_posterior, decoder)
    if cmd_args.test_is or cmd_args.vis_num:
        sys.exit()
 
    for epoch in range(cmd_args.num_epochs):
        test_loss = test(epoch)
        scheduler.step(test_loss)
        train(epoch)
        torch.save(decoder.state_dict(), cmd_args.save_dir + '/epoch-%d.decoder' % epoch)
        torch.save(encoder.diff_var_dict(), cmd_args.save_dir + '/epoch-%d.encoder' % epoch)
    test(epoch)
