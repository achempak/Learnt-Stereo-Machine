import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

import numpy as np

import kaolin
from kaolin.datasets import shapenet

import models.grid_fusion
import models.grid_reasoning
import models.image_encoder
import camera.unprojection
from config import SHAPENET_IM, SHAPENET_VOX
from shapenet_pytorch import ShapeNetDataset
from lsm import LSM


### DATALOADING
nvox = 32
vox_dir = SHAPENET_VOX[nvox]
im_dir = SHAPENET_IM
split_file = './splits.json'
batch_size = 10
nviews = 4

#dataset = ShapeNetDataset(im_dir, vox_dir, nviews, nvox, split_file, train=True, categories=['vessel', 'sofa'])

dataset = ShapeNetDataset(im_dir, vox_dir, nviews, nvox, split_file, train=True)
#img, vol, K, R = dataset[0]

batch_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
#### END DATALOADING

lsm = LSM()

for i, batch in enumerate(batch_loader):

    imgs, vox, K, R = batch
    imgs = imgs.type(torch.FloatTensor)
    vox = vox.type(torch.FloatTensor)
    K = K.type(torch.FloatTensor)
    R = R.type(torch.FloatTensor)

    K = K.view(-1, 3, 3)
    R = R.view(-1, 3, 4)

    vox_prd = lsm(imgs, K, R)
    print(vox_pred.shape)


# Test the binary cross entropy loss on random voxel occupancy grid
#loss_func = torch.nn.BCELoss()
#rand_vox = torch.empty(batch_size,1,32,32,32).random_(2).to(device)
#loss = loss_func(final_grid, rand_vox)
#print("test loss = ",loss)
