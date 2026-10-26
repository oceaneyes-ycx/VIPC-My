"""Point encoders and attention utilities used by VGTNet.

Derived from the research code and SVDFormer/SymmCompletion; see THIRD_PARTY_NOTICES.md.
"""
import torch
from torch import nn, einsum
from extensions.Pointnet2.pointnet2.pointnet2_utils import (
    furthest_point_sample, grouping_operation, gather_operation,
)

def query_knn_point(k, xyz, new_xyz):
    """
    Input:
        x: point cloud, [B, N, 3]
    Return:
        group_xyz: [B, N, K]
    """
    dist = square_distance(new_xyz, xyz)
    _, group_idx = dist.topk(k, largest=False)
    return group_idx.int()


class Transformer(nn.Module):

    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)
        self.pos_mlp = nn.Sequential(nn.Conv2d(3, pos_hidden_dim, 1), nn.BatchNorm2d(pos_hidden_dim), nn.ReLU(), nn.Conv2d(pos_hidden_dim, dim, 1))
        self.attn_mlp = nn.Sequential(nn.Conv2d(dim, dim * attn_hidden_multiplier, 1), nn.BatchNorm2d(dim * attn_hidden_multiplier), nn.ReLU(), nn.Conv2d(dim * attn_hidden_multiplier, dim, 1))
        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            点云特征
            pos: Tensor of positions, (B, 3, n)
            点的空间坐标

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
            带有attention的点云特征
        """
        identity = x
        x = self.linear_start(x)
        b, dim, n = x.shape
        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn_point(self.n_knn, pos_flipped, pos_flipped)
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)
        key = grouping_operation(key, idx_knn)
        qk_rel = query.reshape((b, -1, n, 1)) - key
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)
        pos_embedding = self.pos_mlp(pos_rel)
        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)
        value = grouping_operation(value, idx_knn) + pos_embedding
        agg = einsum('b c i j, b c i j -> b c i', attention, value)
        y = self.linear_end(agg)
        return y + identity


class Conv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), if_bn=True, activation_fn=torch.relu):
        super(Conv2d, self).__init__()
        self.af = nn.ReLU()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, input):
        out = self.conv(input)
        if self.if_bn:
            out = self.bn(out)
        if self.activation_fn is not None:
            out = self.af(out)
        return out


class MLP_CONV(nn.Module):

    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        self.af = nn.ReLU()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(self.af)
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Args:
        xyz: Tensor, (B, 3, nsample)
        points: Tensor, (B, f, nsample)
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, 1)
        new_points: Tensor, (B, f|f+3|3, 1, nsample)
        idx: Tensor, (B, 1, nsample)
        grouped_xyz: Tensor, (B, 3, 1, nsample)
    """
    b, _, nsample = xyz.shape
    device = xyz.device
    new_xyz = torch.zeros((1, 3, 1), dtype=torch.float, device=device).repeat(b, 1, 1)
    grouped_xyz = xyz.reshape((b, 3, 1, nsample))
    idx = torch.arange(nsample, device=device).reshape(1, 1, nsample).repeat(b, 1, 1)
    if points is not None:
        if use_xyz:
            new_points = torch.cat([xyz, points], 1)
        else:
            new_points = points
        new_points = new_points.unsqueeze(2)
    else:
        new_points = grouped_xyz
    return (new_xyz, new_points, idx, grouped_xyz)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def sample_and_group_knn(xyz, points, npoint, k, use_xyz=True, idx=None):
    """
    Args:
        xyz: Tensor, (B, 3, N)
        points: Tensor, (B, f, N)
        npoint: int
        nsample: int
        radius: float
        use_xyz: boolean

    Returns:
        new_xyz: Tensor, (B, 3, npoint)
        new_points: Tensor, (B, 3 | f+3 | f, npoint, nsample)
        idx_local: Tensor, (B, npoint, nsample)
        grouped_xyz: Tensor, (B, 3, npoint, nsample)

    """
    xyz_flipped = xyz.permute(0, 2, 1).contiguous()
    new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint))
    if idx is None:
        idx = query_knn_point(k, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())
    grouped_xyz = grouping_operation(xyz, idx)
    grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, k)
    if points is not None:
        grouped_points = grouping_operation(points, idx)
        if use_xyz:
            new_points = torch.cat([grouped_xyz, grouped_points], 1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return (new_xyz, new_points, idx, grouped_xyz)


class PointNet_SA_Module_KNN(nn.Module):

    def __init__(self, npoint, nsample, in_channel, mlp, if_bn=True, group_all=False, use_xyz=True, if_idx=False):
        """
        Args:
            npoint: int, number of points to sample    需要下采样的点数（输出点数）
            nsample: int, number of points in each local region  每个局部区域内的点数（邻居点数）
            radius: float  
            in_channel: int, input channel of features(points)
            mlp: list of int,
        """
        super(PointNet_SA_Module_KNN, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = mlp
        self.group_all = group_all
        self.use_xyz = use_xyz
        self.if_idx = if_idx
        if use_xyz:
            in_channel += 3
        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp[:-1]:
            self.mlp_conv.append(Conv2d(last_channel, out_channel, if_bn=if_bn))
            last_channel = out_channel
        self.mlp_conv.append(Conv2d(last_channel, mlp[-1], if_bn=False, activation_fn=None))
        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz, points, idx=None):
        """
        Args:
            xyz: Tensor, (B, 3, N)  输入点云的坐标
            points: Tensor, (B, f, N) 输入点云的特征

        Returns:
            new_xyz: Tensor, (B, 3, npoint)   下采样后的点云坐标
            new_points: Tensor, (B, mlp[-1], npoint)  #下采样后的点云特征
        """
        if self.group_all:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_knn(xyz, points, self.npoint, self.nsample, self.use_xyz, idx=idx)
        new_points = self.mlp_conv(new_points)
        new_points = torch.max(new_points, 3)[0]
        if self.if_idx:
            return (new_xyz, new_points, idx)
        else:
            return (new_xyz, new_points)


def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd


class ImageSelfAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(ImageSelfAttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout), nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(dropout))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, N, C)
               - B: batch size
               - N: number of tokens (e.g., 14x14 = 196 for 14x14 patch grid)
               - C: embedding dimension
        Returns:
            out: Tensor of shape (B, N, C)
        """
        x_res = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x_res + attn_out
        x_res = x
        x = self.norm2(x)
        x = x_res + self.mlp(x)
        return x
