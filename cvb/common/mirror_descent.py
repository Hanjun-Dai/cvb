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
from cvb.common.my_modules import MyGaussian
from cvb.common.neural_optim import NeuralSGD, NeuralRMSprop


def optimize_variable(z0, fz, breg_div, nsteps, eps=0.0):
    prev_z = z0
    f_zk = fz(prev_z)    
    lr = cmd_args.mda_lr
    for i in range(nsteps):
        dz = torch.autograd.grad(outputs=f_zk, inputs=prev_z,
                              grad_outputs = torch.ones(f_zk.size()).cuda() if cmd_args.ctx == 'gpu' else torch.ones(f_zk.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        cand = breg_div.grad_phi(prev_z) - lr * dz
        new_z = breg_div.inv_grad_phi(cand)
        if eps > 0:
            noise = new_z.data.new(new_z.size()).normal_() * eps
            new_z = new_z + noise
        new_fz = fz(new_z)
        if new_fz.item() > f_zk.item():
            lr *= cmd_args.mda_decay_factor
            eps *= cmd_args.mda_decay_factor
        f_zk = new_fz
        prev_z = new_z
        
    return prev_z


def optimize_gaussian(input, init_encoder, fz, inner_opt_class, nsteps, training):
    _, mu, logvar = init_encoder(input)
    dist = MyGaussian(mu, logvar)
    if training:
        dist.train()
    else:
        dist.eval()
    optim = inner_opt_class(dist, lr = cmd_args.mda_lr)

    z = dist.sample()

    f_zk = fz( z, dist.mu, dist.logvar )

    lr = optim.lr
    for i in range(nsteps):
        dz = torch.autograd.grad(outputs=f_zk, inputs=dist.diff_variables(),
                              grad_outputs = torch.ones(f_zk.size()).cuda() if cmd_args.ctx == 'gpu' else torch.ones(f_zk.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)
        optim.step(dz, lr = lr)

        z = dist.sample()
        new_fz = fz( z, dist.mu, dist.logvar )
        if new_fz.item() > f_zk.item():
            lr *= cmd_args.mda_decay_factor
            if lr < cmd_args.min_mda_lr:
                lr = cmd_args.min_mda_lr

        f_zk = new_fz
    
    return z, dist.mu, dist.logvar


def optimize_func(input, z_from_input, fz, optim, nsteps):    
    z = z_from_input(input)
    if type(z) is tuple:
        f_zk = fz( *z )
    else:
        f_zk = fz(z)
    lr = optim.lr
    for i in range(nsteps):
        dz = torch.autograd.grad(outputs=f_zk, inputs=z_from_input.diff_variables(),
                              grad_outputs = torch.ones(f_zk.size()).cuda() if cmd_args.ctx == 'gpu' else torch.ones(f_zk.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)
        
        optim.step(dz, lr = lr)
        
        z = z_from_input(input)
        if type(z) is tuple:
            new_fz = fz( *z )
        else:
            new_fz = fz( z )
        if new_fz.item() > f_zk.item():
            lr *= cmd_args.mda_decay_factor
            if lr < cmd_args.min_mda_lr:
                lr = cmd_args.min_mda_lr

        f_zk = new_fz
    
    return z
