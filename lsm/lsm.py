import torch
from torch import nn
import torchvision

import numpy as np

import models.grid_fusion
import models.grid_reasoning
import models.image_encoder
import camera.unprojection

class LSM(nn.Module):

    def __init__(self, device):

        super(LSM, self).__init__()

        self.device = device

        # hyperparameters for Grid Fusion model
        height = width = depth = 32
        in_channels = 32
        hidden_dim = [32, 12] # last hidden dim is output dim
        kernel_size = (3, 3, 3) # kernel size for two stacked hidden layer
        num_layers = 2 # number of stacked hidden layer

        self.grid_params = [32, 32, 32] #[128,128,128]
        self.img_shape = [224,224] #[137,137]

        self.image_enc = models.image_encoder.ImUnet().to(self.device)

        self.grid_fusion = models.grid_fusion.ConvGRU(input_size=(depth, height, width),
                        input_dim=in_channels,
                        hidden_dim=hidden_dim,
                        kernel_size=kernel_size,
                        num_layers=num_layers,
                        device=self.device,
                        bias = True,
                        return_all_layers = False).to(self.device)

        self.grid_reasoning = models.grid_reasoning.Modified3DUNet(in_channels=hidden_dim[-1], n_classes=1, base_n_filter = 1).to(self.device)


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
