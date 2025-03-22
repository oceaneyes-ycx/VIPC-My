
import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
from decoder.utils.utils import *




def mix_shapes_2(X, Y, img):
    """
    combine 2 shapes arbitrarily in each batch.
    For more details see https://arxiv.org/pdf/2003.12641.pdf
    Input:
        X, Y - shape and corresponding labels
    Return:
        mixed shape, labels and proportion
    """
  

    # uniform sampling of points from each shape
    device = torch.device('cuda:0')
    batch_size, _, num_points = X.shape
    index = torch.randperm(batch_size).to(device)  # random permutation of examples in batch
    img2 = img[index]
    # draw lambda from beta distribution
    a = 0.4
    b = 0.6
    lam = (b - a) * np.random.random_sample() + a #np.random.ra #if opt.mixup_params > 0 else 1.0

    num_pts_a = round(lam * num_points)
    num_pts_b = num_points - num_pts_a
   
    X_p = X.permute(0,2,1)#.contiguous()
    pts_vals_a = farthest_point_sample(X_p, num_pts_a)
    pts_vals_b = farthest_point_sample(X_p[index, :], num_pts_b)
    mixed_X = torch.cat((pts_vals_a, pts_vals_b), 1).permute(0,2,1)  # convex combination
    points_perm = torch.randperm(num_points).to(device)  # draw random permutation of points in the shape
    mixed_X = mixed_X[:, :, points_perm]

    Y_p = Y.permute(0,2,1)#.contiguous()
    pts_vals_a_Y = farthest_point_sample(Y_p, num_pts_a)
    pts_vals_b_Y = farthest_point_sample(Y_p[index, :], num_pts_b)
    mixed_Y = torch.cat((pts_vals_a_Y, pts_vals_b_Y), 1).permute(0,2,1)  # convex combination
    mixed_Y = mixed_Y[:, :, points_perm]

    mixed_img = img * lam + img2 * (1-lam)

    return mixed_X, mixed_Y, mixed_img

class L2_ChamferLoss_weighted(nn.Module):
    def __init__(self):
        super(L2_ChamferLoss_weighted, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self, array1, array2):
        dist1, dist2, _, _ = self.chamfer_dist(array1, array2)
        dist = 0.75 * torch.mean(dist1) + 0.25 * torch.mean(dist2)
        return dist