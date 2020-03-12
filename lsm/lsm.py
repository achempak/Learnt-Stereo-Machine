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

class LSM(nn.Module):

    def __init__(self):

        super(LSM, self).__init__()

        # hyperparameters for Grid Fusion model
        height = width = depth = 32
        in_channels = 32
        hidden_dim = [32, 12] # last hidden dim is output dim
        kernel_size = (3, 3, 3) # kernel size for two stacked hidden layer
        num_layers = 2 # number of stacked hidden layer

        # create 3D feature grids from unprojection step

        # Cuda/gpu setup
        use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if use_gpu:
            dtype = torch.cuda.FloatTensor # computation in GPU
        else:
            dtype = torch.FloatTensor

        # 2d unet
        self.image_enc = models.image_encoder.ImUnet().to(self.device)

        # unprojection
        self.grid_params = [32, 32, 32] #[128,128,128]
        self.img_shape = [224,224] #[137,137]

        # camera.unprojection.unproj_grid(grid_params, img_shape, feats, K, R)

        # Grid Fusion model
        self.grid_fusion = models.grid_fusion.ConvGRU(input_size=(depth, height, width),
                        input_dim=in_channels,
                        hidden_dim=hidden_dim,
                        kernel_size=kernel_size,
                        num_layers=num_layers,
                        dtype=dtype,
                        bias = True,
                        return_all_layers = False).to(self.device)

        #feature_grids = torch.rand(batch_size, sequence_length, in_channels, depth, height, width).type(dtype)  # (b,t,c,d,h,w)
        #print("batch of feature_grids shape = {} ({})".format(feature_grids.shape,'batch size, # of views, # in channels, depth, height, width'))


        # Run grid fusion model
        #layer_output_list, last_state_list = grid_fusion(feature_grids)
        #fused_feature_grid = last_state_list[0]
        #print("fused_feature_grid shape = ",fused_feature_grid.shape)

        # Run grid reasoning model
        self.grid_reasoning = models.grid_reasoning.Modified3DUNet(in_channels=hidden_dim[-1], n_classes=1, base_n_filter = 1).to(self.device)
        #batch_size, in_channels, depth, height, width
        #final_grid = grid_reasoning(fused_feature_grid)
        #print("final_grid shape = ",final_grid.shape)


    def forward(self, imgs, K, R):


        batch_size = imgs.shape[0]
        nviews = imgs.shape[1]
        img_feats = self.image_enc(imgs.view(-1,3, self.img_shape[0], self.img_shape[1] ))
        proj_feats = []
        for j in range(len(img_feats)):
            proj_feats.append(camera.unprojection.unproj_grid(self.grid_params, self.img_shape, img_feats[j], K[j], R[j], self.device))
        proj_feats = torch.stack(proj_feats)
        proj_feats = proj_feats.permute(0,2,1)
        proj_feats = proj_feats.view(batch_size, nviews, proj_feats.shape[1], -1)
        proj_feats = proj_feats.view(batch_size, nviews, proj_feats.shape[2], self.grid_params[0], self.grid_params[1], self.grid_params[2])
        layer_output_list, last_state_list = self.grid_fusion(proj_feats)
        fused_feature_grid = last_state_list[0]
        final_grid = self.grid_reasoning(fused_feature_grid)


        return final_grid
