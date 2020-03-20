import torch
import torch.nn as nn
import torch.nn.functional as F

class FCLayers(nn.Module):
    def __init__(self, in_channels, dim):
        super(FCLayers, self).__init__()
        self.in_channels = in_channels
        self.dim = dim

        self.deconv_1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=1, kernel_size=3)
        self.fc_1 = nn.Linear(in_features=dim**3, out_features=dim**3)
        #self.fc_2 = nn.Linear(in_features=dim**3, out_features=dim**3)

        self.deconv_bn1 = nn.BatchNorm3d(1)
        #self.fc_bn1 = nn.BatchNorm3d(1)

    def forward(self, fused_grid):
        x_deconv_1 = self.deconv_1(fused_grid)
        out = self.fc_1(self.deconv_bn1(F.relu(x_deconv_1)))
        #out = self.fc_2(self.fc_bn1(F.relu(x_fc_1)))
        out = out.view((-1, 1, self.dim, self.dim, self.dim))
        out = torch.sigmoid(out)
        print(out.shape)
        return out