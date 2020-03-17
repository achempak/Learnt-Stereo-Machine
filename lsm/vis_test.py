import torch
from vis_tools import *
from shapenet_pytorch import ShapeNetDataset
from IPython.display import display
from IPython.core.display import HTML

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vox_dir =  os.path.join(sys.path[0], 'shapenet_sample/voxels/modelVoxels64')
im_dir = os.path.join(sys.path[0], 'shapenet_sample/renders')
split_file = os.path.join(sys.path[0], 'shapenet_sample/splits_sample.json')

categories = ['bench']
dataset = ShapeNetDataset(im_dir, vox_dir, 4, 64, split_file, train=False, categories=categories)
imgs, vol, K, R = dataset[0]
print("imgs shape" + str(imgs.shape))
print("imgs type" + str(type(imgs)))
vol = vol.squeeze()
pts, cols = voxel2pts(vol)
print(pts.shape)
display(plot_points(pts, cols, size=0.4, axis=False, title=None, html_out='test.html'))
plot_images(imgs, 'test.png')

# Center outputs
HTML("""
<style>
.output {
    align-items: center;
}
</style>
""")