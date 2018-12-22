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
from torch.optim.optimizer import required

from tqdm import tqdm
from collections import defaultdict

class NeuralOptim(object):
    def __init__(self, func, lr=required, momentum=0, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.lr = lr
        self.momentum = momentum        
        self.weight_decay = weight_decay                
        self.func = func
        self.state = defaultdict(dict)
        self.fronzen = False

    def zero_grad(self):
        self.func.detach_diff_vars()
    
    def set_freeze_flag(self, flag):
        self.fronzen = flag

class NeuralSGD(NeuralOptim):
    def __init__(self, func, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):    
        super(NeuralSGD, self).__init__(func, lr, momentum, weight_decay)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")                                

        self.dampening = dampening
        self.nesterov = nesterov        

    def step(self, grads, lr = None):
        if lr is None:
            lr = self.lr
        update_dict = {}        
        for named_param, grad in zip(self.func.named_diff_variables(), grads):
            name, param = named_param            
            assert param.shape == grad.shape
            update_dict[name] = param - lr * grad
        self.func.replace_diff_vars(update_dict)

class NeuralRMSprop(NeuralOptim):
    def __init__(self, func, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super(NeuralRMSprop, self).__init__(func, lr, momentum, weight_decay)

        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))        
        
        self.alpha = alpha
        self.eps = eps        
        self.centered = centered
    
    def step(self, grads, lr = None):
        if lr is None:
            lr = self.lr
        update_dict = {}

        for named_param, grad in zip(self.func.named_diff_variables(), grads):
            name, param = named_param            
            assert param.shape == grad.shape

            state = self.state[name]
            if len(state) == 0: # state initialization
                state['step'] = 0
                state['square_avg'] = torch.zeros_like(param.data)
            
            square_avg = state['square_avg']
            
            if self.fronzen:
                square_avg = square_avg.clone()
            else:
                state['step'] += 1
            # actual update
            square_avg.mul_(self.alpha).addcmul_(1 - self.alpha, grad.data, grad.data)
            avg = square_avg.sqrt().add_(self.eps)
            v_avg = Variable(avg)

            update_dict[name] = param - lr * grad / v_avg

        self.func.replace_diff_vars(update_dict)