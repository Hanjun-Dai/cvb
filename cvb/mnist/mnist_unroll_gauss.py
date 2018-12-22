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
from cvb.mnist.mnist_common import train_loader, test_loader, convert_data
from cvb.mnist.mnist_common import AvbEncoder, AvbDecoder, test_importance_sampling
from cvb.common.mirror_descent import optimize_gaussian
from cvb.common.vae_common import loss_function, FCDecoder, FCGaussianEncoder, binary_cross_entropy
import cvb.common.neural_optim as neural_optim


def train(epoch):
    train_loss = 0
    encoder.train()
    pbar = tqdm(train_loader)
    num_mini_batches = 0
    for (data, _) in pbar:
        data = convert_data(data)

        optimizer.zero_grad()
        fz = lambda z, mu, logvar: loss_function(decoder(z), data, mu, logvar)
        best_z, mu, logvar = optimize_gaussian(data, encoder, fz, inner_opt_class, nsteps = cmd_args.unroll_steps, training=True)
                
        loss = fz(best_z, mu, logvar)
        loss.backward()
        
        train_loss += loss.item()
        optimizer.step()
        recon_loss = binary_cross_entropy(decoder(mu), data)
        pbar.set_description('minibatch loss: %.4f, recon: %.4f' % (loss.item(), recon_loss.item()))
        num_mini_batches += 1
    msg = 'train epoch %d, average loss %.4f' % (epoch, train_loss / num_mini_batches)
    print(msg)


def get_init_posterior(data):
    fz = lambda z, mu, logvar: loss_function(decoder(z), data, mu, logvar)
    if cmd_args.unroll_test:
        best_z, mu, logvar = optimize_gaussian(data, encoder, fz, inner_opt_class, nsteps = cmd_args.unroll_steps, training = True)
    else:
        best_z, mu, logvar = encoder(data)
    return best_z, mu, logvar


def test(epoch):
    encoder.eval()
    test_loss = 0
    for i, (data, _) in tqdm(enumerate(test_loader)):
        data = convert_data(data)
        fz = lambda z, mu, logvar: loss_function(decoder(z), data, mu, logvar)
        if cmd_args.unroll_test:
            best_z, mu, logvar = optimize_gaussian(data, encoder, fz, inner_opt_class, nsteps = cmd_args.unroll_steps, training = True)
        else:
            best_z, mu, logvar = encoder(data)

        loss = fz(best_z, mu, logvar)
        recon_batch = decoder(best_z)
        test_loss += loss.item() * data.shape[0]
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(cmd_args.batch_size, 1, cmd_args.img_size, cmd_args.img_size)[:n]])
            save_image(comparison.data.cpu(),
                     '%s/unroll_gauss_reconstruction_' % cmd_args.save_dir + str(epoch) + '.png', nrow=n)
    test_loss /= len(test_loader.dataset)
    msg = 'test epoch %d, average loss %.4f' % (epoch, test_loss)
    print(msg)
    return test_loss


def get_init_posterior(data):
    fz = lambda z, mu, logvar: loss_function(decoder(z), data, mu, logvar)
    if cmd_args.unroll_test:
        best_z, mu, logvar = optimize_gaussian(data, encoder, fz, inner_opt_class, nsteps = cmd_args.unroll_steps, training = True)
    else:
        best_z, mu, logvar = encoder(data)
    return best_z, mu, logvar


def do_vis(epoch):
    encoder.eval()
    for i, (data, _) in tqdm(enumerate(test_loader)):
        save_image(data[:64].data.cpu(),
                    '%s/dataset-%d-' % (cmd_args.save_dir, i) + str(epoch) + '.png', nrow=8)
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
        
        if i + 1 >= cmd_args.vis_num:
            break


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    if cmd_args.arch == 'mlp':
        decoder = FCDecoder(cmd_args.latent_dim, cmd_args.img_size * cmd_args.img_size)
        encoder = FCGaussianEncoder(cmd_args.img_size * cmd_args.img_size, cmd_args.latent_dim)
    else:
        decoder = AvbDecoder(isize=cmd_args.img_size, latent_dim=cmd_args.latent_dim)
        encoder = AvbEncoder(latent_dim=cmd_args.latent_dim)

    if cmd_args.init_model_dump is not None:
        encoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.encoder'))
        decoder.load_state_dict(torch.load(cmd_args.init_model_dump + '.decoder'))

    if cmd_args.ctx == 'gpu':
        encoder.cuda()
        decoder.cuda()
    optimizer = optim.Adam(chain( encoder.parameters(), decoder.parameters() ) , lr=cmd_args.learning_rate)
    inner_opt_class = getattr(neural_optim, 'Neural' + cmd_args.inner_opt)

    if cmd_args.vis_num > 0:
        epoch = cmd_args.init_model_dump.split('-')[-1]
        do_vis(epoch)
    if cmd_args.test_is:
        test_importance_sampling(test_loader, get_init_posterior, decoder)
    if cmd_args.test_is or cmd_args.vis_num:
        sys.exit()
 
    for epoch in range(cmd_args.num_epochs):
        test_loss = test(epoch)
        train(epoch)
        torch.save(decoder.state_dict(), cmd_args.save_dir + '/epoch-%d.decoder' % epoch)
        torch.save(encoder.state_dict(), cmd_args.save_dir + '/epoch-%d.encoder' % epoch)
    test(epoch)
