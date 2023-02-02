import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points
# from option import MyOptions as cfg

def fps(pc, num):
    """farthest point sampling

    Args:
        pc (tensor): 目标点云 [B, N, 3]
        num (int): 需要得到的中心点数量 

    Returns:
        tensor: 经过fps得到的中心点 [B, num, 3]
    """
    _, fps_idx = sample_farthest_points(pc, K=num) 
    # import pdb; pdb.set_trace()
    center = torch.gather(pc.transpose(1, 2).contiguous(), dim=-1, index=fps_idx.unsqueeze(1).repeat(1, 3, 1)).transpose(1,2).contiguous()
    return center