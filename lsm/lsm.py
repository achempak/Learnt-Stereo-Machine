import torch
from torch.utils.data import DataLoader
import torchvision

import kaolin
from kaolin.datasets import shapenet

import models.grid_fusion
import models.grid_reasoning
import models.image_encoder
import camera.unprojection


# hyperparameters for Grid Fusion model
height = width = depth = 32
in_channels = 16
hidden_dim = [32, 12] # last hidden dim is output dim
kernel_size = (3, 3, 3) # kernel size for two stacked hidden layer
num_layers = 2 # number of stacked hidden layer

# create 3D feature grids from unprojection step
batch_size = 10
sequence_length = views = 6

# Cuda/gpu setup
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if use_gpu:
    dtype = torch.cuda.FloatTensor # computation in GPU
else:
    dtype = torch.FloatTensor


### DATALOADING
class RandSampler(torch.utils.data.Sampler):

    def __init__(self, ordering):
        self.ordering = ordering

    def __iter__(self):
        return iter(self.ordering)
    def __len__(self):
        return len(self.ordering)

#meshes = shapenet.ShapeNet_Meshes(root=shapenet_dir, categories=['plane'])
#categories=['plane', 'bench', 'cabinet', 'car', 'chair', 'monitor', 'lamp', 'speaker', 'rifle', 'sofa', 'table', 'phone', 'watercraft']

### DIRECTORIES
image_dir = '/data1/shapenet/ShapeNetRendering'
shapenet_dir = '/data1/shapenet/ShapeNetCore.v1'

### CATEGORIES
categories = ['bench']

### VOXELS
voxels = shapenet.ShapeNet_Voxels(root=shapenet_dir, cache_dir='/data1/shapenet/.cache', categories=categories, train=True, split=0.7, resolutions=[128])

### IMAGES
imgs = shapenet.ShapeNet_Images(root=image_dir, categories=categories, train=True, split=0.7, transform=torchvision.transforms.Compose([torchvision.transforms.Resize(128), torchvision.transforms.ToTensor()]), views=views)

### SAMPLER
samp = RandSampler(torch.randperm(len(voxels)).tolist())

train_img_loader = DataLoader(imgs, batch_size=batch_size, shuffle=False, num_workers=8, sampler=samp)
train_vox_loader = DataLoader(voxels, batch_size=batch_size, shuffle=False, num_workers=8, sampler=samp)
#### END DATALOADING

# 2d unet
image_enc = models.image_encoder.ImUnet()

# unprojection
grid_params = [128,128,128]
img_shape = [137,137]
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

for i, (batch_vox, batch_img) in enumerate(zip(train_vox_loader,train_img_loader)):
    K = batch_img['data']['params']['cam_mat']
    R = batch_img['data']['params']['cam_pos']
    img_inputs = batch_img['data']['images'][:,:,0:3,:,:]
    print(img_inputs.shape)
    img_feats = image_enc(img_inputs.view(-1, 3, 128, 128))
    print(img_feats.shape)
    img_feats = img_feats.view(batch_size, views,img_feats.shape[-3], img_feats.shape[-2], img_feats.shape[-1])
    proj_feats = camera.unprojection.unproj_grid(grid_params, img_shape, img_feats, K, R)
    print(proj_feats.shape)


    break

# Test the binary cross entropy loss on random voxel occupancy grid
#loss_func = torch.nn.BCELoss()
#rand_vox = torch.empty(batch_size,1,32,32,32).random_(2).to(device)
#loss = loss_func(final_grid, rand_vox)
#print("test loss = ",loss)
