import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

import numpy as np

import models.image_encoder

class LSM_depth(nn.Module):

    def __init__(self, device):

        super(LSM_depth, self).__init__()

        self.device = device

        self.img_shape = [224,224] #[137,137]

        self.image_enc = models.image_encoder.ImUnet().to(self.device)
        self.image_enc.deconv_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=(1,1)).to(device)
        self.conv_1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1).to(device)


    def forward(self, imgs):

        batch_size = imgs.shape[0]
        nviews = imgs.shape[1]
        depth_estimate = self.image_enc(imgs.view(-1,3, self.img_shape[0], self.img_shape[1] ))
        depth_estimate = self.conv_1(depth_estimate)
        depth_estimate = depth_estimate.view(batch_size, nviews, 1, self.img_shape[0], self.img_shape[1])

        return depth_estimate
