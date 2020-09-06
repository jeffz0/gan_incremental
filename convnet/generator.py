import copy
import random
import time
import pdb
import os, sys

import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn, optim
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn



class Generator(nn.Module):
    def __init__(self, width = 12, deconv = False):
        super(Generator, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        nfms = width
        if deconv:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, nfms*8, 4, stride=2, padding=1),  # [batch, width*8, 2, 2]
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(nfms*8, 256, 4, stride=2, padding=1),  # [batch, width*4, 4, 4]
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),  # [batch, width*2, 8, 8]
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),  # [batch, width*1, 16, 16]
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),# [batch, width*1, 14, 14]
            )
        else:
            self.decoder = nn.Sequential(
                nn.Upsample(size=(2,2), mode='nearest'),
                nn.Conv2d(512, nfms*8, kernel_size=2, stride=1, padding=1), 
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 
                nn.Upsample(size=(4,4), mode='nearest'),
                nn.Conv2d(nfms*8, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ngf*8) x 4 x 4
                nn.Upsample(size=(8,8), mode='nearest'),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ngf*4) x 8 x 8
                nn.Upsample(size=(14,14), mode='nearest'),
                nn.Conv2d( 256, 256, kernel_size=3, stride=1, padding=1),
            )
        self.output_act = nn.Tanh()

    def forward(self, x):
        decoded = self.decoder(x)
        decoded = self.output_act(decoded)
        return decoded

class Generator_emb(nn.Module):
    # Generator with embedding of class as input
    def __init__(self, width = 12, deconv = False):
        super(Generator_emb, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.embedding = nn.Embedding(10, 100) #, max_norm=1.)

        nfms = width
        if deconv:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512+100, nfms*8, 4, stride=2, padding=1),  # [batch, width*8, 2, 2]
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(nfms*8, 256, 4, stride=2, padding=1),  # [batch, width*4, 4, 4]
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),  # [batch, width*2, 8, 8]
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),  # [batch, width*1, 16, 16]
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),# [batch, width*1, 14, 14]
            )
        else:
            self.decoder = nn.Sequential(
                nn.Upsample(size=(2,2), mode='nearest'),
                nn.Conv2d(512+100, nfms*8, kernel_size=2, stride=1, padding=1), 
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 
                nn.Upsample(size=(4,4), mode='nearest'),
                nn.Conv2d(nfms*8, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ngf*8) x 4 x 4
                nn.Upsample(size=(8,8), mode='nearest'),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ngf*4) x 8 x 8
                nn.Upsample(size=(14,14), mode='nearest'),
                nn.Conv2d( 256, 256, kernel_size=3, stride=1, padding=1),
            )
        self.output_act = nn.Tanh()

    def forward(self, x, labels):
#         import pdb
#         pdb.set_trace()
        gen_input = torch.cat((self.embedding(labels).unsqueeze(2).unsqueeze(3), x), 1)
        decoded = self.decoder(gen_input)
        decoded = self.output_act(decoded)
        return decoded

    
class Generator_Image(nn.Module):
    def __init__(self, width = 12, deconv = False):
        super(Generator_Image, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        nfms = width
        if deconv:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(512, nfms*8, 4, stride=2, padding=1),  # [batch, width*8, 2, 2]
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(nfms*8, nfms*4, 4, stride=2, padding=1),  # [batch, width*4, 4, 4]
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(nfms*4, nfms*2, 4, stride=2, padding=1),  # [batch, width*2, 8, 8]
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(nfms*2, nfms, 4, stride=2, padding=1),  # [batch, width*1, 16, 16]
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(nfms, 3, 4, stride=2, padding=1),  # [batch, width*1, 16, 16]
            )
        else:
            self.decoder = nn.Sequential(
                nn.Upsample(size=(2,2), mode='nearest'),
                nn.Conv2d(512, nfms*8, kernel_size=2, stride=1, padding=1), 
                nn.LeakyReLU(0.2, inplace=True),
                # state size. 
                nn.Upsample(size=(4,4), mode='nearest'),
                nn.Conv2d(nfms*8, nfms*4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ngf*8) x 4 x 4
                nn.Upsample(size=(8,8), mode='nearest'),
                nn.Conv2d(nfms*4, nfms*2, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ngf*4) x 8 x 8
                nn.Upsample(size=(16,16), mode='nearest'),
                nn.Conv2d( nfms*2, nfms*1, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(size=(32,32), mode='nearest'),
                nn.Conv2d(nfms, 3, kernel_size=3, stride=1, padding=1),
            )
        self.output_act = nn.Tanh()

    def forward(self, x):
        decoded = self.decoder(x)
        decoded = self.output_act(decoded)
        return decoded



def generator_image(deconv):
    return Generator_Image(width=32, deconv=deconv)

def generator(deconv):
    return Generator(width=32, deconv=deconv)

def generator_emb(deconv):
    return Generator_emb(width=32, deconv=deconv)