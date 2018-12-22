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
from torchvision import datasets, transforms
import torch.utils.data
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from PIL import Image

from cvb.common.cmd_args import cmd_args
from cvb.common.pytorch_util import weights_init

img_size = cmd_args.img_size


class AvbEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(AvbEncoder, self).__init__()
        self.latent_dim = latent_dim
        nc = 1
        ndf = 16
        # self.transform = transform.CenterCrop((64, 64))

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 5, 2, 2, bias=True),
            #nn.BatchNorm2d(ndf),
            nn.Softplus(),
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=True),
            #nn.BatchNorm2d(ndf * 2),
            nn.Softplus(),
            nn.Conv2d(ndf * 2, ndf * 2, 5, 2, 2, bias=True),
            #nn.BatchNorm2d(ndf * 2),
            nn.Softplus(),
        )
        self.fc1 = nn.Linear(ndf * 2 * 4 * 4, 300)
        #self.fc1_bnorm = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300, latent_dim)
        self.fc3 = nn.Linear(300, latent_dim)
        weights_init(self)

    def encode(self, x):
        h1 = self.main(x).view(x.shape[0], -1)
        h1 = self.fc1(h1)
        h1 = F.softplus(h1)
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


class AvbDecoder(nn.Module):
    def __init__(self, isize, latent_dim, act_out = F.sigmoid):
        super(AvbDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.act_out = act_out

        s = isize
        self.s2, self.s4, self.s8 = int(np.ceil(s/2.0)), int(np.ceil(s/4.0)), int(np.ceil(s/8.0))
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 300),
            #nn.BatchNorm1d(300),
            nn.Softplus(),
            nn.Linear(300, self.s8 * self.s8 * 32),
            #nn.BatchNorm1d(self.s8 * self.s8 * 32),
            nn.Softplus()
        )

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 5, 2, 2, output_padding=1),
            #nn.BatchNorm2d(32),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 16, 5, 2, 2, output_padding=1),
            #nn.BatchNorm2d(16),
            nn.Softplus(),
            nn.ConvTranspose2d(16, 1, 5, 2, 2, output_padding=1)
        )

    def forward(self, input):
        h1 = self.fc1(input)
        x = h1.view(-1, 32, self.s8, self.s8)

        output = self.cnn(x)
        if self.act_out is not None:
            output =  self.act_out(output)
        return output


class BinaryMNIST(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    @property
    def targets(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))

            noise = torch.rand(self.train_data.size())
            float_data = self.train_data.float() / 255.0
            float_data = torch.sign(float_data - noise) > 0
            self.train_data = float_data * 255

        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

            noise = torch.rand(self.test_data.size())
            float_data = self.test_data.float() / 255.0
            float_data = torch.sign(float_data - noise) > 0
            self.test_data = float_data * 255

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

if cmd_args.binary:
    data_class = BinaryMNIST
else:
    data_class = datasets.MNIST

train_loader = torch.utils.data.DataLoader(
    data_class(cmd_args.dropbox + '/data/mnist', train=True, download=False,
                transform=transforms.Compose([
                    transforms.Pad((img_size - 28) // 2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(cmd_args.data_mean, cmd_args.data_mean, cmd_args.data_mean), std=(cmd_args.data_std, cmd_args.data_std, cmd_args.data_std))
                ])),
    batch_size=cmd_args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    data_class(cmd_args.dropbox + '/data/mnist', train=False, download=False,
                transform=transforms.Compose([
                    transforms.Pad((img_size - 28) // 2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(cmd_args.data_mean, cmd_args.data_mean, cmd_args.data_mean), std=(cmd_args.data_std, cmd_args.data_std, cmd_args.data_std))
                ])),
    batch_size=cmd_args.test_batch_size, shuffle=True)

def convert_data(data):
    data = Variable(data)
    if cmd_args.ctx == 'gpu':
        data = data.cuda()
    return data

def mnist_recon_loss(pred, target):
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    loss = -( target * torch.log( pred + 1e-20 ) + (1.0 - target) * torch.log( 1.0 - pred + 1e-20 ) )

    return torch.sum(loss, dim=1, keepdim=True)

def norm_nll(x, mu, log_var):
    x = x.view(x.size()[0], -1)
    variance = torch.exp(log_var)
    f = torch.sum( ((x - mu) ** 2) / 2.0 / variance, dim=1, keepdim=True )

    f += x.shape[1] / 2.0 * np.log( 2 * np.pi ) 
    f += 0.5 * torch.sum( log_var, 1, keepdim=True )
    return f

def norm_sample(mu, log_var):
    std = log_var.mul(0.5).exp_()
    eps = std.data.new(std.size()).normal_()
    return eps.mul(std).add_(mu)

from scipy.misc import logsumexp

def test_importance_sampling(test_loader, encoder, decoder):
    pbar = tqdm(test_loader)
    tot = 0.0
    num = 0
    torch.set_grad_enabled(False)
    for (data, _) in pbar:
        data = convert_data(data)
        with torch.enable_grad():
            _, mu, log_var = encoder(data)
        s = 0.0 
        n_steps = 1000
        mu0 = mu * 0.0
        log_var0 = log_var * 0.0
        w_list = []
        if cmd_args.img_size == 32:
            data = data[:, 0, 2:-2, 2:-2]
        for i in tqdm(range(n_steps)):
            z = norm_sample(mu, log_var)
            recon_x = decoder(z)
            if cmd_args.img_size == 32:
                recon_x = recon_x[:, 0, 2:-2, 2:-2]
            log_px_z = -mnist_recon_loss(recon_x, data)
            log_pz = -norm_nll(z, mu0, log_var0)
            log_qz = -norm_nll(z, mu, log_var)
            wi = log_px_z + log_pz - log_qz - np.log(n_steps)
            w_list.append(wi)
        w = torch.cat(w_list, dim=1).data.cpu().numpy()
        s = logsumexp(w, axis=1)
        num += data.shape[0]
        tot += np.sum(s)
        print('ll_mean: %.4f ll_std: %.4f' % (np.mean(s), np.std(s)))
        print('cur_mean %.4f' % (tot / num))
