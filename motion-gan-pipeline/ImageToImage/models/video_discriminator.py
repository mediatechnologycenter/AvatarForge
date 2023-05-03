# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_
import torch.nn.init as init


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, 
                                    out_channels=out_channels, 
                                    kernel_size=kernel_size,
                                   stride=stride, 
                                   padding=padding, 
                                   dilation=dilation, 
                                   groups=groups, 
                                   bias=bias))

def snconv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias))

def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))


class Discriminator_3D(nn.Module):
    def __init__(self, d_conv_dim=96, T=8):
        super(Discriminator_3D, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x T x 96 x 96
            snconv3d(3, d_conv_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x T/2 x 48 x 48
            snconv3d(d_conv_dim, d_conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(d_conv_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x T/4 x 24 x 24
            snconv3d(d_conv_dim * 2, d_conv_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(d_conv_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x T/8 x 12 x 12

            #TODO: fix this because depends on batchsize for us
            # snconv3d(d_conv_dim * 4, d_conv_dim * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm3d(d_conv_dim * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x T/16  x 6 x 6
            )
        self.linear = snlinear(d_conv_dim * 4, 1) # was 8
        self.final_conv = snconv2d(d_conv_dim * 4, 1, 4, 1, 1, bias=False)
        # self.embed = sn_embedding(num_classes, d_conv_dim*8)
        # # Weight init
        # self.apply(init_weights)
        # xavier_uniform_(self.embed.weight)

    def forward(self, input):
        input = input.transpose(2,1)
        output = self.main(input)
        output = output.squeeze(2)
        output_conv = self.final_conv(output)
        # output = torch.sum(output, dim=[3,4]).view(-1, output.size(1))
        # output_linear = self.linear(output)
        # y = class_id.long()
        # embed = self.embed(y)
        # prod = (output * embed).sum(1)
        return output_conv 
