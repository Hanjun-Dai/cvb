from __future__ import print_function

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
from tqdm import tqdm

from cvb.common.cmd_args import cmd_args

def max_f(x_input, energy_func, generator, optimizerF, coupled=False):
    optimizerF.zero_grad()

    f_x = energy_func(x_input)
    loss_true = -F.torch.mean(f_x)
    
    sampled_x = generator(num_samples = x_input.size()[0])
    if not coupled:
        sampled_x = sampled_x.detach()
        
    f_sampled_x = energy_func(sampled_x)
    loss_fake = F.torch.mean(f_sampled_x)
    loss = loss_true + loss_fake

    loss.backward()
    optimizerF.step()
    return loss
