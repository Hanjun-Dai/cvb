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

class EuclideanDist(object):

    @staticmethod
    def dist(x, y):
        return torch.sum( (x - y) ** 2, dim=1, keepdim=True)

    @staticmethod
    def phi(x):
        return x ** 2.0

    @staticmethod
    def grad_phi(x):
        return 2.0 * x

    @staticmethod
    def inv_grad_phi(x):
        return x / 2.0
    
