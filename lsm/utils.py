import numpy as np
import torch

def voxel_to_cloud(voxel: torch.Tensor, num_points: int, thresh: float = 0.4):
    one = torch.ones(voxel.shape, requires_grad=True).to(voxel.device)
    zero = torch.zeros(voxel.shape, requires_grad=True).to(voxel.device)
    voxel = torch.where(voxel > thresh, one, zero)
    voxel_positions = (voxel == 1).nonzero().float().requires_grad_(True)
    
    index_list = list(range(voxel_positions.shape[0]))
    select_index = np.random.choice(index_list, size=num_points)
    point_positions = voxel_positions[select_index]

    point_displacement = torch.rand(
        point_positions.shape).to(
        point_positions.device)
    point_positions += point_displacement

    return point_positions

