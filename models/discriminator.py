import numpy as np
import torch.nn as nn

from models.init_weight import init_net
from models.model_blocks import ResBlock


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=6):
        super(Discriminator, self).__init__()
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1)]
        for i in range(n_layers):
            if i >= 3:
                sequence += [ResBlock(512, 512, down_sample=True, norm=False)]
            else:
                mult = 2**i
                sequence += [ResBlock(ndf * mult, ndf * mult * 2, down_sample=True, norm=False)]
        sequence += [
            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.sequence = init_net(nn.Sequential(*sequence))

    def forward(self, input):
        return self.sequence(input)
