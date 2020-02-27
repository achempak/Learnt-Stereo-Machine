import torch
import models.grid_fusion
import models.grid_reasoning


# Cuda/gpu setup
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if use_gpu:
    dtype = torch.cuda.FloatTensor # computation in GPU

else:
    dtype = torch.FloatTensor

# hyperparameters for Grid Fusion model
height = width = depth = 32
in_channels = 16
hidden_dim = [32, 12] # last hidden dim is output dim
kernel_size = (3, 3, 3) # kernel size for two stacked hidden layer
num_layers = 2 # number of stacked hidden layer

# Grid Fusion model
grid_fusion = models.grid_fusion.ConvGRU(input_size=(depth, height, width),
                input_dim=in_channels,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                num_layers=num_layers,
                dtype=dtype,
                bias = True,
                return_all_layers = False).to(device)

# create 3D feature grids from unprojection step
batch_size = 4
sequence_length = 5
feature_grids = torch.rand(batch_size, sequence_length, in_channels, depth, height, width).type(dtype)  # (b,t,c,d,h,w)
print("batch of feature_grids shape = {} ({})".format(feature_grids.shape,'batch size, # of views, # in channels, depth, height, width'))


# Run grid fusion model
layer_output_list, last_state_list = grid_fusion(feature_grids)
fused_feature_grid = last_state_list[0]
print("fused_feature_grid shape = ",fused_feature_grid.shape)

# Run grid reasoning model
grid_reasoning = models.grid_reasoning.Modified3DUNet(in_channels=hidden_dim[-1], n_classes=1, base_n_filter = 2).to(device)
#batch_size, in_channels, depth, height, width
final_grid = grid_reasoning(fused_feature_grid)
print("final_grid shape = ",final_grid.shape)


# Test the binary cross entropy loss on random voxel occupancy grid
loss_func = torch.nn.BCELoss()
rand_vox = torch.empty(batch_size,1,32,32,32).random_(2).to(device)
loss = loss_func(final_grid, rand_vox)
print("test loss = ",loss)
