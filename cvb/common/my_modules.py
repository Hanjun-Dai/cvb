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
import math
from collections import OrderedDict
import itertools
from torch.nn.modules.utils import _single, _pair, _triple

from cvb.common.pytorch_util import glorot_uniform


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self._diff_vars = OrderedDict()

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        diff_vars = self.__dict__.get('_diff_vars')
        if isinstance(value, Variable) and value.requires_grad:
            if diff_vars is None:
                raise AttributeError("cannot assign diffvar before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules, self._parameters)
            self.register_diff_var(name, value)
        else:
            super(MyModule, self).__setattr__(name, value)
        
    def __getattr__(self, name):
        if '_diff_vars' in self.__dict__:
            _diff_vars = self.__dict__['_diff_vars']
            if name in _diff_vars:
                return _diff_vars[name]
        return super(MyModule, self).__getattr__(name)

    def register_diff_var(self, name, var):
        if '_diff_vars' not in self.__dict__:
            raise AttributeError(
                "cannot assign diffvar before Module.__init__() call")

        if hasattr(self, name) and name not in self._diff_vars:
            raise KeyError("attribute '{}' already exists".format(name))

        if var is None:
            self._diff_vars[name] = None
        elif not isinstance(var, Variable) or var.requires_grad == False:
            raise TypeError("cannot assign '{}' object to variable '{}' "
                            "(torch.autograd.Variable with grad or None required)"
                            .format(torch.typename(var), name))
        else:
            self._diff_vars[name] = var

    def diff_variables(self):
        for name, value in self.named_diff_variables():
            yield value

    def named_diff_variables(self, prefix=''):
        for name, p in self._diff_vars.items():
            if p is not None:
                yield prefix + ('.' if prefix else '') + name, p
        for mname, module in self.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in module.named_diff_variables(submodule_prefix):
                yield name, p

    def replace_diff_vars(self, list_vars, prefix=''):
        cnt = 0
        for name, p in self._diff_vars.items():
            if p is not None:
                cnt += 1
                fullname = prefix + ('.' if prefix else '') + name
                new_p = list_vars[fullname]
                assert p.shape == new_p.shape
                setattr(self, name, new_p)
        
        for mname, module in self.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            cnt += module.replace_diff_vars(list_vars, submodule_prefix)
        if prefix == '':           
            assert cnt == len(list_vars)
        return cnt

    def detach_diff_vars(self):
        for name, p in self._diff_vars.items():
            if p is not None:
                new_p = Variable(p.data, requires_grad=True)
                setattr(self, name, new_p)
        for _, module in self.named_children():            
            module.detach_diff_vars()

    def _apply(self, fn):
        for name, param in self._diff_vars.items():
            if param is not None:
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        return super(MyModule, self)._apply(fn)

    def diff_var_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]] = dict(version=self._version)
        for name, param in self._diff_vars.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf
        for name, module in self._modules.items():
            if module is not None:
                module.diff_var_dict(destination, prefix + name + '.', keep_vars=keep_vars)
        return destination

    def _load_from_diff_var_dict(self, diff_var_dict, prefix, strict, missing_keys, unexpected_keys, error_msgs):
        local_name_params = itertools.chain(self._diff_vars.items(), self._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in diff_var_dict:
                input_param = diff_var_dict[key]
                if isinstance(input_param, Parameter):
                    # backwards compatibility for serialized parameters
                    input_param = input_param.data
                try:
                    param.copy_(input_param)
                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key, input_param in diff_var_dict.items():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    def load_diff_var_dict(self, diff_var_dict, strict=True):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(diff_var_dict, '_metadata', None)
        diff_var_dict = diff_var_dict.copy()
        if metadata is not None:
            diff_var_dict._metadata = metadata

        def load(module, prefix=''):
            module._load_from_diff_var_dict(
                diff_var_dict, prefix, strict, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)
        
class MyLinear(MyModule):
    def __init__(self, in_features, out_features, bias=True):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Variable(torch.Tensor(out_features, in_features), requires_grad=True)
        if bias:
            self.bias = Variable(torch.Tensor(out_features), requires_grad=True)
            self.bias.data.zero_()
        else:
            self.register_diff_var('bias', None)
        glorot_uniform(self.weight.data)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class MyGaussian(MyModule):
    def __init__(self, mu, logvar):
        super(MyGaussian, self).__init__()
        self.mu = mu
        self.logvar = logvar

    def sample(self):
        if not self.training:
            return self.mu

        mu = self.mu
        logvar = self.logvar
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

class _MyConvTransNd(MyModule):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_MyConvTransNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Variable(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size), requires_grad=True)
        else:
            self.weight = Variable(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size), required=True)
        if bias:
            self.bias = Variable(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_diff_var('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class _MyConvTransposeMixin(object):

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        func = self._backend.ConvNd(
            self.stride, self.padding, self.dilation, self.transposed,
            output_padding, self.groups)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)

    def _output_padding(self, input, output_size):
        if output_size is None:
            return self.output_padding

        output_size = list(output_size)
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[-2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})"
                .format(k, k + 2, len(output_size)))

        def dim_size(d):
            return ((input.size(d + 2) - 1) * self.stride[d] -
                    2 * self.padding[d] + self.kernel_size[d])

        min_sizes = [dim_size(d) for d in range(k)]
        max_sizes = [min_sizes[d] + self.stride[d] - 1 for d in range(k)]
        for size, min_size, max_size in zip(output_size, min_sizes, max_sizes):
            if size < min_size or size > max_size:
                raise ValueError((
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})").format(
                        output_size, min_sizes, max_sizes, input.size()[2:]))

        return tuple([output_size[d] - min_sizes[d] for d in range(k)])

class MyConvTranspose2d(_MyConvTransposeMixin, _MyConvTransNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(MyConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

class _MyBatchNorm(MyModule):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_MyBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Variable(torch.Tensor(num_features), requires_grad=True)
            self.bias = Variable(torch.Tensor(num_features), requires_grad=True)
        else:
            self.register_diff_var('weight', None)
            self.register_diff_var('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_diff_var('running_mean', None)
            self.register_diff_var('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

class MyBatchNorm2d(_MyBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class _MyConvNd(MyModule):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_MyConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Variable(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size), requires_grad=True)
        else:
            self.weight = Variable(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size), requires_grad=True)
        if bias:
            self.bias = Variable(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_diff_var('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class MyConv2d(_MyConvNd):                             
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MyConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)    

class MySequential(MyModule):

    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return MySequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input                        


class MyThreshold(MyModule):
    def __init__(self, threshold, value, inplace=False):
        super(MyThreshold, self).__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace
        # TODO: check in THNN (if inplace == True, then assert value <= threshold)

    def forward(self, input):
        return F.threshold(input, self.threshold, self.value, self.inplace)

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'threshold={}, value={}{}'.format(
            self.threshold, self.value, inplace_str
        )

class MyReLU(MyThreshold):
    def __init__(self, inplace=False):
        super(MyReLU, self).__init__(0, 0, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str        


class MySoftplus(MyModule):
    __constants__ = ['beta', 'threshold']

    def __init__(self, beta=1, threshold=20):
        super(MySoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        return F.softplus(input, self.beta, self.threshold)

    def extra_repr(self):
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)
