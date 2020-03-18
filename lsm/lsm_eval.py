import torch
from torch import nn
from vis_tools import *
from shapenet_pytorch import ShapeNetDataset
from IPython.display import display
from IPython.core.display import HTML
from lsm import LSM
from kaolin.metrics.point import chamfer_distance


from config import SHAPENET_IM, SHAPENET_VOX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### TRAINING PARAMETERS
epochs = 10
lr = 0.0001
nvox = 32
batch_size = 10
nviews = 4

### DATALOADING
vox_dir = SHAPENET_VOX[nvox]
im_dir = SHAPENET_IM
split_file = './splits.json'
categories = ['sofa']
test_dataset = ShapeNetDataset(im_dir, vox_dir, nviews, nvox, split_file, train=False, categories=categories)

imgs, vol, K, R = test_dataset[0]
print("vol shape" + str(vol.shape))
imgs_tensor = torch.Tensor(imgs).to(device)
imgs_tensor = imgs_tensor.unsqueeze(0)
K = torch.Tensor(K).to(device)
R = torch.Tensor(R).to(device)
print("imgs shape" + str(imgs_tensor.shape))

lsm = LSM()
lsm.load_state_dict(torch.load('./checkpoints/experiment_1584352545/best_model.pth'))

with torch.no_grad():
    vol_pred = lsm(imgs_tensor, K, R)

vol_pred = vol_pred.squeeze()
vol_pred = vol_pred.cpu().numpy()
print("vox pred shape: "+str(vol_pred.shape))
write_tensor(vol.squeeze(), 'test_true.txt')
write_tensor(vol_pred, 'test_pred.txt')

pts, cols = voxel2pts(vol_pred, type='Chamfer')
pts_true, cols_true = voxel2pts(vol)
display(plot_points(pts, cols, size=0.4, axis=False, title=None, html_out='eval_sofa.html'))
display(plot_points(pts_true, cols_true, size=0.4, axis=False, title=None, html_out='true_sofa.html'))
plot_images(imgs, 'eval_sofa.png')

# Center outputs
HTML("""
<style>
.output {
    align-items: center;
}
</style>
""")