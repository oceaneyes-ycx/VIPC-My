
import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
from decoder.utils.utils import *

from chamfer_distance import ChamferDistance


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
    


def calc_dcd(x, gt, alpha=40, n_lambda=0.5, return_raw=False, non_reg=False):
    #计算DCD损失
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    loss1 = []
    loss2 = []
    for b in range(batch_size):
        count1 = torch.bincount(idx1[b])
        weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
        weight1 = (weight1 + 1e-6) ** (-1) * frac_21
        loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

        count2 = torch.bincount(idx2[b])
        weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
        weight2 = (weight2 + 1e-6) ** (-1) * frac_12
        loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

    loss1 = torch.stack(loss1)
    loss2 = torch.stack(loss2)
    loss = loss1 * 0.75 + loss2 * 0.25

    res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res


def calc_cd(output, gt, return_raw=False, separate=False):
    # cham_loss = dist_chamfer_3D.chamfer_3DDist()
    cham_loss = ChamferDistance()
    dist1, dist2, idx1, idx2 = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))

    if separate:
        res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
               torch.cat([dist1.mean(1).unsqueeze(0),dist2.mean(1).unsqueeze(0)])]
    else:
        res = [cd_p, cd_t]

    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res