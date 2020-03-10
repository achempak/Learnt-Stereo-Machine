import torch
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


# hyperparameters for Grid Fusion model
height = width = depth = 32
in_channels = 32
hidden_dim = [32, 12] # last hidden dim is output dim
kernel_size = (3, 3, 3) # kernel size for two stacked hidden layer
num_layers = 2 # number of stacked hidden layer

# create 3D feature grids from unprojection step
batch_size = 10

# Cuda/gpu setup
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if use_gpu:
    dtype = torch.cuda.FloatTensor # computation in GPU
else:
    dtype = torch.FloatTensor


### DATALOADING
nvox = 32
vox_dir = SHAPENET_VOX[nvox]
im_dir = SHAPENET_IM
split_file = './splits.json'
nviews = 4

#dataset = ShapeNetDataset(im_dir, vox_dir, nviews, nvox, split_file, train=True, categories=['vessel', 'sofa'])

dataset = ShapeNetDataset(im_dir, vox_dir, nviews, nvox, split_file, train=True)
img, vol, K, R = dataset[0]

batch_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
#### END DATALOADING




# 2d unet
image_enc = models.image_encoder.ImUnet()

# unprojection
grid_params = vol.shape[1:] #[128,128,128]
img_shape = img.shape[-2:] #[137,137]

# camera.unprojection.unproj_grid(grid_params, img_shape, feats, K, R)

# Grid Fusion model
grid_fusion = models.grid_fusion.ConvGRU(input_size=(depth, height, width),
                input_dim=in_channels,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                num_layers=num_layers,
                dtype=dtype,
                bias = True,
                return_all_layers = False).to(device)

#feature_grids = torch.rand(batch_size, sequence_length, in_channels, depth, height, width).type(dtype)  # (b,t,c,d,h,w)
#print("batch of feature_grids shape = {} ({})".format(feature_grids.shape,'batch size, # of views, # in channels, depth, height, width'))


# Run grid fusion model
#layer_output_list, last_state_list = grid_fusion(feature_grids)
#fused_feature_grid = last_state_list[0]
#print("fused_feature_grid shape = ",fused_feature_grid.shape)

# Run grid reasoning model
grid_reasoning = models.grid_reasoning.Modified3DUNet(in_channels=hidden_dim[-1], n_classes=1, base_n_filter = 2).to(device)
#batch_size, in_channels, depth, height, width
#final_grid = grid_reasoning(fused_feature_grid)
#print("final_grid shape = ",final_grid.shape)


for i, batch in enumerate(batch_loader):

    imgs, vox, K, R = batch
    imgs = imgs.type(torch.FloatTensor)
    vox = vox.type(torch.FloatTensor)
    K = K.type(torch.FloatTensor)
    R = R.type(torch.FloatTensor)

    K = K.view(-1, 3, 3)
    R = R.view(-1, 3, 4)

    img_feats = image_enc(imgs.view(-1,3, img_shape[0], img_shape[1] ).type(torch.FloatTensor))
    proj_feats = []
    for j in range(len(img_feats)):
        proj_feats.append(camera.unprojection.unproj_grid(grid_params, img_shape, img_feats[j], K[j], R[j]))
    proj_feats = torch.stack(proj_feats)
    proj_feats = proj_feats.permute(0,2,1)
    proj_feats = proj_feats.view(batch_size, nviews, proj_feats.shape[1], -1)
    proj_feats = proj_feats.view(batch_size, nviews, proj_feats.shape[2], grid_params[0], grid_params[1], grid_params[2])
    layer_output_list, last_state_list = grid_fusion(proj_feats)
    fused_feature_grid = last_state_list[0]
    final_grid = grid_reasoning(fused_feature_grid)

# Test the binary cross entropy loss on random voxel occupancy grid
#loss_func = torch.nn.BCELoss()
#rand_vox = torch.empty(batch_size,1,32,32,32).random_(2).to(device)
#loss = loss_func(final_grid, rand_vox)
#print("test loss = ",loss)
