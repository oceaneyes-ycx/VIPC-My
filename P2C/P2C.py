import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.transformer import Group
from extensions.chamfer_dist import ChamferDistanceL1
from extensions.pointops.functions import pointops
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from timm.models.layers import trunc_normal_
from utils.logger import *


class Encoder(nn.Module):
    def __init__(self, feat_dim):
        #基于PCN的编码器
        """
        PCN based encoder
        """
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        #第一层卷积，输入通道数为3，输出通道数为128，卷积核大小为1
        #3*2048->128*2048->256*2048
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, feat_dim, 1)
        )
        #第二层卷积，输入通道数为512，输出通道数为512，卷积核大小为1
        #512*2048->512*2048->1024*2048

    def forward(self, x):
        #前向传播
        #x: B N 3(batch_size, num_points, 3)
        bs, n, _ = x.shape
        feature = self.first_conv(x.transpose(2, 1))  # B 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        # 全局特征： B 256 1 (dim=2表示对第二维度求最大值)
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  
        # B 512 n 中间特征，特征拼接（局部特征+全局特征）
        feature = self.second_conv(feature)  
        # B 1024 n 二次卷积
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  
        # B 1024 最终全局特征
        return feature_global


class Decoder(nn.Module):
    def __init__(self, latent_dim=1024, num_output=2048):
        #latent_dim :   潜在空间的维度
        #num_output :   输出点云的点数
        super().__init__()
        self.latent_dim = latent_dim
        self.num_output = num_output

        self.mlp1 = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 3 * self.num_output)
        )#定义的多层MLP，输入为latent_dim，输出为3*num_output

    def forward(self, z):
        #z： B latent_dim
        bs = z.size(0)

        pcd = self.mlp1(z).reshape(bs, -1, 3)  #  B M C(3)
        # z -> MLP: B 3*num_output, reshape: B num_output 3
        # 此即为生成的点云

        return pcd


class ManifoldnessConstraint(nn.Module):
    """
    The Normal Consistency Constraint
    法向一致性约束
    """
    def __init__(self, support=8, neighborhood_size=32):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=3, eps=1e-6)
        #余弦相似度计算器，计算法向量之间的余弦相似度
        self.support = support
        #支持点数
        self.neighborhood_size = neighborhood_size
        #邻域大小

    def forward(self, xyz):
        #xyz: B N 3  输入点云

        normals = estimate_pointcloud_normals(xyz, neighborhood_size=self.neighborhood_size)
        #估计点云法向量

        idx = pointops.knn(xyz, xyz, self.support)[0]
        #找到每个点的最近邻点
        neighborhood = pointops.index_points(normals, idx)
        #根据最近邻点的索引找到对应的法向量

        cos_similarity = self.cos(neighborhood[:, :, 0, :].unsqueeze(2), neighborhood)
        #计算法向量之间的余弦相似度
        penalty = 1 - cos_similarity
        #余弦相似度越大，法向量越一致，惩罚越小
        penalty = penalty.std(-1)#标准差
        penalty = penalty.mean(-1)#平均值
        return penalty#返回法向一致性约束的惩罚值（作为损失函数的一部分）


class P2C(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__() 
        # define parameters
        self.config = config
        self.num_group = config.P2Cnet.num_group#分组数
        self.group_size = config.P2Cnet.group_size#每组点数
        self.mask_ratio = config.P2Cnet.mask_ratio#分组比例
        self.feat_dim = config.P2Cnet.feat_dim#特征维度
        self.n_points = config.P2Cnet.n_points#输出点数
        self.nbr_ratio = config.P2Cnet.nbr_ratio#邻域比例

        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        #分组器
        self.encoder = Encoder(self.feat_dim)
        #编码器
        self.generator = Decoder(latent_dim=self.feat_dim, num_output=self.n_points)
        #解码器

        # init weights
        self.apply(self._init_weights)#初始化权重
        # init loss
        self._get_lossfnc_and_weights(config)#初始化损失函数

    def _init_weights(self, m):
        #权重初始化方法
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_lossfnc_and_weights(self, config):
        # define loss functions
        self.shape_criterion = ChamferDistanceL1()#cdL1损失
        self.latent_criterion = nn.SmoothL1Loss(reduction='mean')
        #潜在空间重建损失
        self.manifold_constraint = ManifoldnessConstraint(support=config.P2Cnet.support, neighborhood_size=config.P2Cnet.neighborhood_size)
        #法向一致性约束
        self.shape_matching_weight = config.P2Cnet.shape_matching_weight
        #形状匹配权重
        self.shape_recon_weight = config.P2Cnet.shape_recon_weight
        #形状重建权重
        self.latent_weight = config.P2Cnet.latent_weight
        #潜在空间权重
        self.manifold_weight = config.P2Cnet.manifold_weight
        #法向一致性权重

    def _group_points(self, nbrs, center, B, G):
        #点云分组方法
        #点云分组，生成领域点和中心点的分组
        nbr_groups = []
        center_groups = []
        perm = torch.randperm(G)
        acc = 0
        for i in range(3):
            mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
            mask[:, perm[acc:acc+self.mask_ratio[i]]] = True
            nbr_groups.append(nbrs[mask].view(B, self.mask_ratio[i], self.group_size, -1))
            center_groups.append(center[mask].view(B, self.mask_ratio[i], -1))
            acc += self.mask_ratio[i]
        return nbr_groups, center_groups

    def get_loss(self, pts):
        #具体的计算损失的方法
        # group points
        nbrs , center = self.group_divider(pts)  # neighborhood, center
        #nbrs: B G N 3 代表邻域点
        #center: B G 3 代表中心点
        B, G, _ = center.shape
        #B: batch_size
        #G: group_size
        nbr_groups, center_groups = self._group_points(nbrs, center, B, G)
        #点云分组
        #nbr_groups: 邻域点分组,cneter_groups: 中心点分组

        # pre-encoding -- partition 1
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        #重建点云 = 邻域点 + 中心点 (这个点云是一个假GT点云)
        feat  = self.encoder(rebuild_points.view(B, -1, 3))
        #特征提取

        # complete shape generation
        pred = self.generator(feat).contiguous()
        #根据特征解码器生成点云

        #接下来计算假GT点云和生成点云之间的损失
        # shape reconstruction loss
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        #假GT点云
        idx = pointops.knn(center_groups[0], pred,  int(self.nbr_ratio * self.group_size))[0]
        #找到生成点云中每个点的最近邻点
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        #根据最近邻点的索引找到对应的点
        shape_recon_loss = self.shape_recon_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()
        #计算形状重建损失（cdL1损失）
        # shape completion loss
        rebuild_points = nbr_groups[1] + center_groups[1].unsqueeze(-2)
        idx = pointops.knn(center_groups[1], pred,  int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_matching_loss = self.shape_matching_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()
        # latent reconstruction loss
        idx = pointops.knn(center_groups[2], pred, self.group_size)[0]
        nbrs_pred = pointops.index_points(pred, idx)
        feat_recon = self.encoder(nbrs_pred.view(B, -1, 3).detach())
        latent_recon_loss = self.latent_weight * self.latent_criterion(feat, feat_recon)
        # normal consistency constraint
        #NCC损失
        manifold_penalty = self.manifold_weight * self.manifold_constraint(pred).mean()

        total_loss = shape_recon_loss + shape_matching_loss + latent_recon_loss + manifold_penalty

        return total_loss, shape_recon_loss, shape_matching_loss, latent_recon_loss, manifold_penalty


    def forward(self, partial, n_points=None, record=False):
        # group points
        #INPUT: partial: B N 3
        #OUTPUT: pred: B n_points 3
        B, _, _ = partial.shape
        feat = self.encoder(partial)
        pred = self.generator(feat).contiguous()
        return pred
