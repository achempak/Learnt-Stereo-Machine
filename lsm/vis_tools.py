import os
import sys
from pathlib import Path

import numpy as np
import torch
from IPython.display import IFrame
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.ndimage.filters import median_filter
from uuid import uuid4

with open(os.path.join(sys.path[0], 'pyntcloud.js'), 'r') as f:
    TEMPLATE_POINTS = f.read()


def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array, norm=False)[:, :-1]

def plot_points(xyz, colors=None, size=0.1, axis=False, title=None, html_out=None):
    positions = xyz.reshape(-1).tolist()

    Path('./figures/predict/3D/').mkdir(parents=True, exist_ok=True)
    Path('./figures/true/3D/').mkdir(parents=True, exist_ok=True)
    if html_out is None:
        html_out = os.path.join('./figures/', str(uuid4())+'.html')
    else:
        html_out = os.path.join('./figures/', html_out)
    if title is None:
        title = "PointCloud"
    camera_position = xyz.max(0) + abs(xyz.max(0))
    look = xyz.mean(axis=0)

    if colors is None:
        colors = [1, 0.5, 0] * len(positions)
    elif len(colors.shape) > 1:
        colors = colors.reshape(-1).tolist()
    
    if axis:
        axis_size = xyz.ptp() * 1.5
    else:
        axis_size = 0
    with open(html_out, "w") as html:
        html.write(
            TEMPLATE_POINTS.format(
                title=title,
                camera_x=camera_position[0],
                camera_y=camera_position[1],
                camera_z=camera_position[2],
                look_x=look[0],
                look_y=look[1],
                look_z=look[2],
                positions=positions,
                colors=colors,
                points_size=size,
                axis_size=axis_size
            )
        )

    return IFrame(html_out, width=1024, height=768)

def plot_images(imgs, out_file=None):
    Path('./figures/views/').mkdir(parents=True, exist_ok=True)
    a,b,c,d = imgs.shape
    imgs = imgs.reshape((a,c,d,b))
    a,b,c,d = imgs.shape
    num_views = a if a <= 4 else 4
    fig, axes = plt.subplots(2,2)
    for i in range(num_views):
        if(i < 2):
            axes[0][i].imshow(imgs[i,int(b/3)*2:int(b/3)*3,int(b/3)*2:int(b/3)*3,:])
        else:
            axes[1][i-2].imshow(imgs[i,int(b/3)*2:int(b/3)*3,int(b/3)*2:int(b/3)*3,:])
    if out_file is None:
        out_file = os.path.join('./figures/views/',str(uuid4())+'.html')
    else:
        out_file = os.path.join('./figures/views/',out_file)
    plt.savefig(out_file)
#def voxel_grid(voxels, thresh=0.4, cmap='viridis'):



def voxel2pts(voxels, thresh=0.4, cmap='Oranges'):
    if voxels.ndim == 4:
        voxels = voxels.squeeze()
    elif voxels.ndim != 3:
        print('Invalid number of dimensions in voxel grid')
    vox = (voxels > thresh).astype(np.int)
    points = np.argwhere(vox > 0)
    colors = array_to_color(voxels[vox > 0], cmap=cmap)
    return points, colors
