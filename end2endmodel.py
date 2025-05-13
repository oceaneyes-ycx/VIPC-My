'''
端到端训练模型,不再使用预训练的对比学习模型,而是直接训练一个端到端的模型
没有什么变化，仅仅是解除了编码器的参数冻结。
'''


import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
from datetime import datetime, timedelta
from decoder.dec_net import Decoder_Network
from encoder_dgcnn.dgcnn import DGCNN, DGCNN2, DGCNN3

import torchvision.transforms as transforms
from models.resnet import ResNet, ResNet2
from torchvision.models import resnet50, resnet18, resnet34

from config_xmf import params

from lightly.loss.ntx_ent_loss import NTXentLoss
from utils.average_meter import AverageMeter, AverageMeter2

from datasets import data_utils as d_utils

from encoder_image.resnet_xmf import ResNet as ResNet_xmf

from extensions.chamfer_dist import ChamferDistanceL1
from extensions.pointops.functions import pointops
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from timm.models.layers import trunc_normal_
from decoder.dec_net import MappingNet, AXform

from decoder.utils.utils import *
from models.transformer import Group

from P2C.P2C import Encoder, Decoder,P2C

from UTIL import mix_shapes_2,calc_cd,calc_dcd

from propagation import Propagation



trans_pc_1 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),#将点云转化为Tensor
                d_utils.PointcloudNormalize(),#归一化
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),#缩放
                d_utils.PointcloudRotate(),#随机旋转，增强旋转不变性
                d_utils.PointcloudTranslate(0.5, p=1),#平移，增强空间偏移鲁棒性
                d_utils.PointcloudJitter(p=1),#添加噪声
                d_utils.PointcloudRandomInputDropout(p=1),#随机丢弃一些点，应对输入数据的缺失可能
            ])
    
trans_pc_2 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])

trans_img_1 = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace = True)])

trans_img_2 = transforms.Compose([transforms.Grayscale(1),
                                transforms.Resize((224, 224)),
                                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(0.485, 0.229, inplace = True)])



p = params()

class End2endNetwork(nn.Module):
    #2025/01/20：模型复刻了XMFNet的全部功能，经测试没有问题
    #需要注意输入的点云维度应该为(B,N,3)，而不是(B,3,N)，我在这里进行了一定的修改
    def __init__(self):
        super(End2endNetwork, self).__init__()

        self.img_model = ResNet2(resnet18(pretrained=False), feat_dim = 512)
        self.point_model = DGCNN2()
        #特征融合模块
        #使用注意力层融合两种模态的信息

        self.cross_attn1 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        #使用多头注意力机制，交叉注意力机制
        self.layer_norm1 = nn.LayerNorm(p.d_attn)
        #层归一化

        self.self_attn1 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        #使用多头注意力机制，自注意力机制
        self.layer_norm2 = nn.LayerNorm(p.d_attn)
        #层归一化

        self.cross_attn2 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        #使用多头注意力机制，交叉注意力机制
        self.layer_norm3 = nn.LayerNorm(p.d_attn)

        self.self_attn2 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm4 = nn.LayerNorm(p.d_attn)

        self.cross_attn3 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm5 = nn.LayerNorm(p.d_attn)


        #解码器模块，根据融合之后的特征重建点云
        self.decoder = Decoder_Network()


    



    def forward(self, x_part, view):
        #x_part:部分点云数据   view:图像数据
        #x_part:[16, 3, 2048]  view:[16, 3, 224, 224]

        #特征提取模块

        x_part = x_part.cuda()
        view = view.cuda()
        #提取图像和点云的特征(注意点云跟对比学习训练用的全局特征不一样，这里要用空间特征)
        pc_feat1, pc_feat2  = self.point_model(x_part.transpose(1, 2).contiguous())  # 提取点云特征
        
        im_feat1, im_feat2  = self.img_model(view)  # 提取图像特征
        #fea1：空间特征, 用于点云补全  fea2：空间特征，用于对比学习


        im_feat1 = im_feat1.permute(0, 2, 1)  # 调整图像特征维度   [16, 196, 256]
        pc_feat1 = pc_feat1.permute(0, 2, 1)  # 调整点云特征维度  [16, 128, 256]

        #特征融合模块
        x, _ = self.cross_attn1(pc_feat1, im_feat1, im_feat1)
        pc_feat1 = self.layer_norm1(x + pc_feat1)
        x, _ = self.self_attn1(pc_feat1, pc_feat1, pc_feat1)   #x:[16, 128, 256]
        pc_feat1 = self.layer_norm2(x + pc_feat1)   #pc_feat:[16, 128, 256]

        pc_skip = pc_feat1  #保存pc_feat，用于后续的融合   [16, 128, 256]

        x, _ = self.cross_attn2(pc_feat1, im_feat1, im_feat1)
        pc_feat1 = self.layer_norm3(x + pc_feat1)
        x, _ = self.self_attn2(pc_feat1, pc_feat1, pc_feat1)  #x:[16, 128, 256]
        pc_feat1 = self.layer_norm4(x + pc_feat1)   #pc_feat:[16, 128, 256]

        x, _ = self.cross_attn3(pc_feat1, pc_skip, pc_skip)  #x:[16, 128, 256]
        pc_feat1 = self.layer_norm5(x + pc_feat1)   #pc_feat:[16, 128, 256]
        #x_part = x_part.permute(0, 2, 1)  # B x 3 x N ----> B x N x 3  [16, 2048, 3]

        final = self.decoder(pc_feat1, x_part)  #重建的点云
        


        return final



class FusionDecoder(nn.Module):
    #从end2endNetwork中单独拆分出来的解码器模块，输入是点云和图像特征以及原始点云
    #为P2C2定制。
    def __init__(self):
        super(FusionDecoder, self).__init__()

        self.cross_attn1 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(p.d_attn)

        self.self_attn1 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(p.d_attn)

        self.cross_attn2 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm3 = nn.LayerNorm(p.d_attn)

        self.self_attn2 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm4 = nn.LayerNorm(p.d_attn)

        self.cross_attn3 = nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm5 = nn.LayerNorm(p.d_attn)

        self.decoder = Decoder_Network()

    def forward(self, pc_feat1, im_feat1, x_part):
        """
        pc_feat1: 点云空间特征 [B, N_pc, d_attn]   B,256,80
        im_feat1: 图像空间特征 [B, N_img, d_attn]  B,256,196
        x_part: 输入的原始部分点云 [B, 3, N]
        """
        # 融合阶段
        x, _ = self.cross_attn1(pc_feat1, im_feat1, im_feat1) #B,80,256
        pc_feat1 = self.layer_norm1(x + pc_feat1) #B,80,256

        x, _ = self.self_attn1(pc_feat1, pc_feat1, pc_feat1) #B,80,256
        pc_feat1 = self.layer_norm2(x + pc_feat1)#B,80,256

        pc_skip = pc_feat1#B,80,256

        x, _ = self.cross_attn2(pc_feat1, im_feat1, im_feat1)#B,80,256
        pc_feat1 = self.layer_norm3(x + pc_feat1)#B,80,256

        x, _ = self.self_attn2(pc_feat1, pc_feat1, pc_feat1)#B,80,256
        pc_feat1 = self.layer_norm4(x + pc_feat1)#B,80,256

        x, _ = self.cross_attn3(pc_feat1, pc_skip, pc_skip)#B,80,256
        pc_feat1 = self.layer_norm5(x + pc_feat1)#B,80,256

        # 解码阶段
        final = self.decoder(pc_feat1, x_part)

        return final



class XMFnet(nn.Module):

    def __init__(self):
        super(XMFnet, self).__init__()

        # Encoders for images and Point clouds
        #图像和点云的编码器
        self.pc_encoder = DGCNN()#使用DGCNN编码器
        self.im_encoder = ResNet_xmf()#使用ResNet编码器


        # 特征融合模块
        # Attention layers to fuse the information from the two modalities
        #使用注意力层融合两种模态的信息

        self.cross_attn1 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        #使用多头注意力机制，交叉注意力机制
        self.layer_norm1 = nn.LayerNorm(p.d_attn)
        #层归一化

        self.self_attn1 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        #使用多头注意力机制，自注意力机制
        self.layer_norm2 = nn.LayerNorm(p.d_attn)
        #层归一化

        self.cross_attn2 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        #使用多头注意力机制，交叉注意力机制
        self.layer_norm3 = nn.LayerNorm(p.d_attn)

        self.self_attn2 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm4 = nn.LayerNorm(p.d_attn)

        self.cross_attn3 = nn.MultiheadAttention(
            p.d_attn, p.num_heads, batch_first=True)
        self.layer_norm5 = nn.LayerNorm(p.d_attn)



        #解码器模块，根据融合之后的特征重建点云
        # Decoder Network to reconstruct the point cloud
        self.decoder = Decoder_Network()

    def forward(self, x_part, view):
        '''
        输入：部分点云x_part和视图view
        特征提取：
        点云特征提取：使用DGCNN编码器提取点云特征pc_feat
        图像特征提取：使用ResNet编码器提取图像特征im_feat

        特征融合：
        第一次交叉注意力机制(cross_attn1)：融合图像特征im_feat和点云特征pc_feat,得到融合后的特征x
        第一次层归一化layer_norm1：将融合后的特征x与原始点云特征pc_feat相加并进行层归一化，得到新的点云特征pc_feat
        第一次自注意力机制self_attn1：对新的点云特征pc_feat进行自注意力机制处理，得到融合点云特征x
        第二次层归一化layer_norm2：将融合后的特征x与自注意力机制之前的点云特征pc_feat相加并进行层归一化，得到新的点云特征pc_feat
        
        保存当前点云特征pc_feat为pc_skip，用于后续的融合

        第二次交叉注意力机制cross_attn2：融合图像特征im_feat和上一步得到的点云特征pc_feat，得到融合后的特征x
        第三次层归一化layer_norm3：将融合后的特征x与原始点云特征pc_feat相加并进行层归一化，得到新的点云特征pc_feat
        第二次自注意力机制self_attn2：对新的点云特征pc_feat进行自注意力机制处理，得到融合点云特征x
        第四次层归一化layer_norm4：将融合后的特征x与自注意力机制之前的点云特征pc_feat相加并进行层归一化，得到新的点云特征pc_feat

        第三次交叉注意力机制cross_attn3：融合上一步得到的点云特征pc_feat和pc_skip，得到融合后的特征x
        第五次层归一化layer_norm5：将融合后的特征x与原始点云特征pc_feat相加并进行层归一化，得到新的点云特征pc_feat

        解码器模块：AXform，根据最终的点云特征pc_feat和部分点云x_part重建点云

        
        '''
       
        pc_feat, _  = self.pc_encoder(x_part)  #B x F x N  提取的点云特征，[16,256,128]
        im_feat = self.im_encoder(view)  #B x F x N   提取的图像特征，[16,256,196]

        im_feat = im_feat.permute(0, 2, 1) #调整维度，[16,196,256]
        pc_feat  = pc_feat.permute(0, 2, 1) #调整维度，[16,128,256]

        x, _ = self.cross_attn1(pc_feat, im_feat, im_feat)
        #使用交叉注意力机制，融合图像特征和点云特征，
        #x为融合后的特征[16,128,256]，_为注意力权重
        #在nn.MultiheadAttention(p.d_attn, p.num_heads, batch_first=True)
        #p.d_attn: 表示注意力机制中每个输入特征的维度（特征维度）。
        #p.num_heads: 表示多头注意力的头数，将输入特征维度分割为多个头进行并行计算。
        #batch_first=True: 指定输入的张量格式为 (batch_size, seq_length, feature_dim)。


        #在该调用中，pc_feat和im_feat都是输入特征，即Q、K和V。
        #点云的特征表示作为 Q,用于指导模型从图像特征中提取与之相关的信息
        #图像的特征表示作为K 和 V提供查询和价值信息
        pc_feat = self.layer_norm1(x + pc_feat) # B x N x F
        #层归一化，将融合后的特征与原始点云特征相加,稳定训练，防止梯度爆炸，消除模态间的分布差异
        #pc_feat为融合后的特征，[16,128,256]
        x, _ = self.self_attn1(pc_feat, pc_feat, pc_feat)
        #z使用自注意力机制，融合点云特征
        #Q、K和V都是pc_feat，即点云特征
        #x为融合后的特征[16,128,256]，_为注意力权重
        pc_feat = self.layer_norm2(x + pc_feat)
        #层归一化，将融合后的特征与原始点云特征相加
        #pc_feat为融合后的特征，[16,128,256]

        pc_skip = pc_feat  #保存pc_feat，用于后续的融合
        
        x, _ = self.cross_attn2(pc_feat, im_feat, im_feat)
        #x为融合后的特征[16,128,256]，_为注意力权重
        pc_feat = self.layer_norm3(x + pc_feat)
        #[16,128,256]
        x, _ = self.self_attn2(pc_feat, pc_feat, pc_feat)
        #x为融合后的特征[16,128,256]，_为注意力权重
        pc_feat = self.layer_norm4(x + pc_feat)
        #[16,128,256]

        x, _ = self.cross_attn3(pc_feat, pc_skip, pc_skip)
        #x:[16,128,256]
        pc_feat = self.layer_norm5(x + pc_feat)
        #[16,128,256]
        x_part = x_part.permute(0, 2, 1)  # B x 3 x N ----> B x N x 3
        #[16,3,2048]---->[16,2048,3]        

        final = self.decoder(pc_feat, x_part)
        #[16,2048,3] 
        return final
        

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



class P2CDecoder_O(nn.Module):
    #P2C原版的Decoder
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

class P2C2(nn.Module):
    #需要注意输入的点云维度应该为(B,N,3)，而不是(B,3,N)，我在这里进行了一定的修改
    # 2025/04/22 添加P2C的功能
    def __init__(self, config):
        super(P2C2, self).__init__()

        self.img_model = ResNet2(resnet18(pretrained=False), feat_dim = 512)#图像编码器
        self.point_model = DGCNN3()   #点云编码器

        self.fusion_decoder = FusionDecoder()   #多模态解码器


        # define parameters (用于P2C计算损失中的参数)
        self.config = config
        self.num_group = config.P2Cnet.num_group#分组数
        self.group_size = config.P2Cnet.group_size#每组点数
        self.mask_ratio = config.P2Cnet.mask_ratio#分组比例
        self.feat_dim = config.P2Cnet.feat_dim#特征维度
        self.n_points = config.P2Cnet.n_points#输出点数
        self.nbr_ratio = config.P2Cnet.nbr_ratio#邻域比例        

        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        #分组器
        #self.generator = P2CDecoder_O(latent_dim=self.feat_dim, num_output=self.n_points)
        #P2C原版的解码器，很简单的网络结构

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
        self.shape_criterion = ChamferDistanceL1()#cdL1损失，在这里是RCD损失
        self.latent_criterion = nn.SmoothL1Loss(reduction='mean')
        #潜在空间重建损失
        self.manifold_constraint = ManifoldnessConstraint(support=config.P2Cnet.support, neighborhood_size=config.P2Cnet.neighborhood_size)
        #法向一致性约束
        self.shape_matching_weight = config.P2Cnet.shape_matching_weight
        #形状匹配权重， CD损失的权重
        self.shape_recon_weight = config.P2Cnet.shape_recon_weight
        #形状重建权重， CD损失的权重
        self.latent_weight = config.P2Cnet.latent_weight
        #潜在空间权重, latent损失的权重
        self.manifold_weight = config.P2Cnet.manifold_weight
        #法向一致性权重, 法向一致性约束的权重, 默认是0（不使用）

    def _group_points(self, nbrs, center, B, G):
        #点云分组方法   G: 64
        #点云分组，生成领域点和中心点的分组
        #将输入的点云数据划分为多个子组，并生成对应的邻域点和中心点分组
        #nbrs: B, 64, 32, 3  邻域点
        #center: B, 64, 3  中心点
        nbr_groups = [] #存储邻域点分组结果
        center_groups = []#存储中心点分组结果
        perm = torch.randperm(G)  #生成一个长度为G的随机排列，确保分组的随机性
        acc = 0 #累加器，用于记录当前分组的起始位置
        #mask_ratio: [10, 50, 4]  分组比例，在配置文件中定义，代表每个分组的点数
        for i in range(3):
            mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
            mask[:, perm[acc:acc+self.mask_ratio[i]]] = True
            nbr_groups.append(nbrs[mask].view(B, self.mask_ratio[i], self.group_size, -1))
            center_groups.append(center[mask].view(B, self.mask_ratio[i], -1))
            acc += self.mask_ratio[i]
        return nbr_groups, center_groups

    def get_loss(self, pts, view):
        #具体的计算损失的方法
        #2025/05/09 添加一下mix的方法，尝试是否有效
        
        # group points  
        # 首先，将输入的部分点云划分为多个patch，并生成对应的邻域点和中心点分组
        nbrs , center = self.group_divider(pts)  # neighborhood, center
        #nbrs: B G N 3 代表patch的邻域点   B, 64, 32, 3  (64个group中心点，每个中心点取32个邻域点)
        #center: B G 3 代表patch的中心点   B, 64, 3 （64个group中心点）
        B, G, _ = center.shape
        #B: batch_size
        #G: group_size     64

        #第二步，根据设定好的分组的大小，再将点云进行二次分组操作
        nbr_groups, center_groups = self._group_points(nbrs, center, B, G)
        #点云分组  [10,50,4], 10用于计算重建损失，50用于计算补全损失，4用于计算潜在空间重建损失
        #nbr_groups: 邻域点分组,cneter_groups: 中心点分组
        # nbr_groups: 0: B,10,32,3  1: B,50,32,3, 2: B,4,32,3
        # center_groups: 0: B,10,3  1: B,50,3, 2: B,4,3
        #我们将点云分为三组，10，50，4为每组的中心点的数量，然后，每个中心点对应32个邻域点


        # pre-encoding -- partition 1
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        #重建点云 = 邻域点 + 中心点 (这个点云是一个假GT点云) B,10,32,3
        
        img_feat , _  = self.img_model(view) #使用图像编码器提取图像特征
        #img_feat: B, 256, 196      用于解码的局部特征
        # 第二个输入栏为统一的全局特征  B,256

        #10个中心点，每个中心点对应32个邻域点
        point_feat , feat_global  = self.point_model(rebuild_points.view(B, 3, -1))
        #view: B, 3, 32*10      feat: B, 256, 80  feat_global: B, 256
        #320个点作为假GT点云
        #使用点云编码器提取假GT点云的特征
        point_feat0 = point_feat.permute(0, 2, 1)  # [B, N_pc, C] B, 80, 256
        img_feat = img_feat.permute(0, 2, 1)  # [B, N_img, C] B, 196, 256
        # complete shape generation
        pred = self.fusion_decoder(point_feat0, img_feat, pts)
        #根据联合特征解码器根据提取的特征生成点云    B，2048,3

        #接下来计算假GT点云和生成点云之间的损失


        # shape reconstruction loss （使用第一组点云） 形状重建损失
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)   #和上面一样的重建点云
        #假GT点云  B，10,32,3
        idx = pointops.knn(center_groups[0], pred,  int(self.nbr_ratio * self.group_size))[0]
        # B,10,64   center_groups[0]为采样后的点，pred为生成的点云,使用knn方法找到采样点在pred生成点云中的最近邻点，取k个，这里设置的ratio为2,所以采用了64个
        #找到生成点云中每个点的最近邻点
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        # B，640，3  从生成点云pred中提取对应的邻居点，在生成点云中找到第一个分组对应的生成点们
        #根据最近邻点的索引找到对应的点
        shape_recon_loss = self.shape_recon_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()
        #计算形状重建损失（cdL1损失）

        # shape completion loss  （使用第二组点云） 形状补全损失
        rebuild_points = nbr_groups[1] + center_groups[1].unsqueeze(-2)
        idx = pointops.knn(center_groups[1], pred,  int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_matching_loss = self.shape_matching_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()
        
        # latent reconstruction loss  （使用第三组点云） 潜在空间重建损失
        idx = pointops.knn(center_groups[2], pred, self.group_size)[0]
        nbrs_pred = pointops.index_points(pred, idx)
        _ , feat_recon= self.point_model(nbrs_pred.view(B, 3, -1).detach())
        latent_recon_loss = self.latent_weight * self.latent_criterion(feat_global, feat_recon)
        # normal consistency constraint
        #NCC损失
        manifold_penalty = self.manifold_weight * self.manifold_constraint(pred).mean()

        total_loss = shape_recon_loss + shape_matching_loss + latent_recon_loss + manifold_penalty

        return total_loss, shape_recon_loss, shape_matching_loss, latent_recon_loss, manifold_penalty


    def forward(self, x_part, view):
        x_part = x_part.cuda()
        view = view.cuda()

        pc_feat1, pc_feat2 = self.point_model(x_part.transpose(1, 2).contiguous())
        #pc_feat1: [16,256,512]  pc_feat2: [16,256]
        im_feat1, im_feat2 = self.img_model(view)
        #img_feat1: [16,256,196]  img_feat2: [16,256]
        im_feat1 = im_feat1.permute(0, 2, 1)  # [B, N_img, C] B, 196, 256
        pc_feat1 = pc_feat1.permute(0, 2, 1)  # [B, N_pc, C] B, 512, 256

        final = self.fusion_decoder(pc_feat1, im_feat1, x_part)
        return final

class NET(nn.Module):
    #需要注意输入的点云维度应该为(B,N,3)，而不是(B,3,N)，我在这里进行了一定的修改
    # 2025/04/22 添加P2C的功能
    # 2025/05/13 采用全新的网络架构
    def __init__(self, config):
        super(NET, self).__init__()

        self.img_model = resnet34(pretrained=False)#图像编码器,换用更强的resnet34
        self.point_model = DGCNN3()   #点云编码器

        #self.fusion_decoder = FusionDecoder()   #多模态解码器

        self.layer1 = nn.Sequential(*list(self.img_model.children())[:5])
        #resnet34的前五层
        self.layer2 = nn.Sequential(*list(self.img_model.children())[5])
        #resnet34的第六层
        self.layer3 = nn.Sequential(*list(self.img_model.children())[6])
        #resnet34的第七层
        self.layer4 = nn.Sequential(*list(self.img_model.children())[7])
        #resnet34的第八层

        self.deconv1_1 = nn.ConvTranspose2d(512,256,5,stride=2,padding=2,output_padding=1)
        #512通道的特征图上进行反卷积，输出256通道的特征图
        self.deconv1_2 = nn.ConvTranspose2d(256,128,5,stride=2,padding=2,output_padding=1)
        #256通道的特征图上进行反卷积，输出128通道的特征图
        self.deconv1_3 = nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1)
        #128通道的特征图上进行反卷积，输出64通道的特征图
        self.deconv1_4 = nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1)
        #64通道的特征图上进行反卷积，输出32通道的特征图

        self.conv6_1 = nn.Conv2d(64,32,3,stride=1,padding=1)
        #32通道的特征图上进行卷积，输出32通道的特征图
        self.conv6_2 = nn.Conv2d(32,3,3,stride=1,padding=1)
        #32通道的特征图上进行卷积，输出3通道的特征图
        
        #AXForm解码器的参数
        self.num_branch = p.num_branch#分支数  8
        self.K1 = p.K1
        self.K2 = p.K2
        self.N = p.N


        self.fc0_up = nn.Linear(3, 256)
        self.fc0_down = nn.Linear(256, 3)
        self.cross_attn0_0 = nn.MultiheadAttention(
            256, 4, batch_first=True)
        self.layer_norm0_0 = nn.LayerNorm(256)
        self.cross_attn0_1 = nn.MultiheadAttention(
            256, 4, batch_first=True)
        self.layer_norm0_1 = nn.LayerNorm(256)

        self.fc1_up = nn.Linear(3, 128)
        self.fc1_down = nn.Linear(128, 3)
        self.cross_attn1_0 = nn.MultiheadAttention(
            128, 4, batch_first=True)
        self.layer_norm1_0 = nn.LayerNorm(128)
        self.cross_attn1_1 = nn.MultiheadAttention(
            128, 4, batch_first=True)
        self.layer_norm1_1 = nn.LayerNorm(128)

        self.fc2_up = nn.Linear(3, 64)
        self.fc2_down = nn.Linear(64, 3)
        self.cross_attn2_0 = nn.MultiheadAttention(
            64, 4, batch_first=True)
        self.layer_norm2_0 = nn.LayerNorm(64)
        self.cross_attn2_1 = nn.MultiheadAttention(
            64, 4, batch_first=True)
        self.layer_norm2_1 = nn.LayerNorm(64)

        self.featmap = nn.ModuleList([MappingNet(self.K1) for i in range(self.num_branch)])
        self.pointgen = nn.ModuleList([AXform(self.K1, self.K2, self.N) for i in range(self.num_branch)])

        self.propagation = Propagation()#点云细化模块

        # define parameters (用于P2C计算损失中的参数)
        self.config = config
        self.num_group = config.P2Cnet.num_group#分组数
        self.group_size = config.P2Cnet.group_size#每组点数
        self.mask_ratio = config.P2Cnet.mask_ratio#分组比例
        self.feat_dim = config.P2Cnet.feat_dim#特征维度
        self.n_points = config.P2Cnet.n_points#输出点数
        self.nbr_ratio = config.P2Cnet.nbr_ratio#邻域比例        

        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        #分组器
        #self.generator = P2CDecoder_O(latent_dim=self.feat_dim, num_output=self.n_points)
        #P2C原版的解码器，很简单的网络结构

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
        self.shape_criterion = ChamferDistanceL1()#cdL1损失，在这里是RCD损失
        self.latent_criterion = nn.SmoothL1Loss(reduction='mean')
        #潜在空间重建损失
        self.manifold_constraint = ManifoldnessConstraint(support=config.P2Cnet.support, neighborhood_size=config.P2Cnet.neighborhood_size)
        #法向一致性约束
        self.shape_matching_weight = config.P2Cnet.shape_matching_weight
        #形状匹配权重， CD损失的权重
        self.shape_recon_weight = config.P2Cnet.shape_recon_weight
        #形状重建权重， CD损失的权重
        self.latent_weight = config.P2Cnet.latent_weight
        #潜在空间权重, latent损失的权重
        self.manifold_weight = config.P2Cnet.manifold_weight
        #法向一致性权重, 法向一致性约束的权重, 默认是0（不使用）

    def _group_points(self, nbrs, center, B, G):
        #点云分组方法   G: 64
        #点云分组，生成领域点和中心点的分组
        #将输入的点云数据划分为多个子组，并生成对应的邻域点和中心点分组
        #nbrs: B, 64, 32, 3  邻域点
        #center: B, 64, 3  中心点
        nbr_groups = [] #存储邻域点分组结果
        center_groups = []#存储中心点分组结果
        perm = torch.randperm(G)  #生成一个长度为G的随机排列，确保分组的随机性
        acc = 0 #累加器，用于记录当前分组的起始位置
        #mask_ratio: [10, 50, 4]  分组比例，在配置文件中定义，代表每个分组的点数
        for i in range(3):
            mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
            mask[:, perm[acc:acc+self.mask_ratio[i]]] = True
            nbr_groups.append(nbrs[mask].view(B, self.mask_ratio[i], self.group_size, -1))
            center_groups.append(center[mask].view(B, self.mask_ratio[i], -1))
            acc += self.mask_ratio[i]
        return nbr_groups, center_groups

    def get_loss(self, pts, view):
        #具体的计算损失的方法
        
        # group points  
        # 首先，将输入的部分点云划分为多个patch，并生成对应的邻域点和中心点分组
        nbrs , center = self.group_divider(pts)  # neighborhood, center
        #nbrs: B G N 3 代表patch的邻域点   B, 64, 32, 3  (64个group中心点，每个中心点取32个邻域点)
        #center: B G 3 代表patch的中心点   B, 64, 3 （64个group中心点）
        B, G, _ = center.shape
        #B: batch_size
        #G: group_size     64

        #第二步，根据设定好的分组的大小，再将点云进行二次分组操作
        nbr_groups, center_groups = self._group_points(nbrs, center, B, G)
        #点云分组  [10,50,4], 10用于输入编码器进行编码，50用于生成假GT点云用于计算损失，4用于计算潜在空间重建损失
        #nbr_groups: 邻域点分组,cneter_groups: 中心点分组
        # nbr_groups: 0: B,10,32,3  1: B,50,32,3, 2: B,4,32,3
        # center_groups: 0: B,10,3  1: B,50,3, 2: B,4,3
        #我们将点云分为三组，10，50，4为每组的中心点的数量，然后，每个中心点对应32个邻域点


        # pre-encoding -- partition 1
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)
        #重建点云 = 邻域点 + 中心点 表示一个输入点云的大致形状，用于输入编码器进行编码
        
        #img_feat  = self.img_model(view) #使用图像编码器提取图像特征
        #img_feat: B, 256, 196      用于解码的局部特征

        #10个中心点，每个中心点对应32个邻域点
        point_feat , feat_global  = self.point_model(rebuild_points.view(B, 3, -1))
        #view: B, 3, 32*10      feat: B, 256, 80  feat_global: B, 256
        #320个点作为假GT点云
        #使用点云编码器提取假GT点云的特征
        #point_feat0 = point_feat.permute(0, 2, 1)  # [B, N_pc, C] B, 80, 256
        #img_feat = img_feat.permute(0, 2, 1)  # [B, N_img, C] B, 196, 256
        # complete shape generation
        pred , _ , pc_gen , _  = self.forward(rebuild_points.view(B, -1, 3), view)

        #pred = self.fusion_decoder(point_feat0, img_feat, pts)
        #根据联合特征解码器根据提取的特征生成点云    B，2048,3

        #接下来计算假GT点云和生成点云之间的损失


        # shape reconstruction loss （使用第一组点云） 形状重建损失
        rebuild_points = nbr_groups[0] + center_groups[0].unsqueeze(-2)   #和上面一样的重建点云
        #假GT点云  B，10,32,3
        idx = pointops.knn(center_groups[0], pred,  int(self.nbr_ratio * self.group_size))[0]
        # B,10,64   center_groups[0]为采样后的点，pred为生成的点云,使用knn方法找到采样点在pred生成点云中的最近邻点，取k个，这里设置的ratio为2,所以采用了64个
        #找到生成点云中每个点的最近邻点
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        # B，640，3  从生成点云pred中提取对应的邻居点，在生成点云中找到第一个分组对应的生成点们
        #根据最近邻点的索引找到对应的点
        shape_recon_loss = self.shape_recon_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()
        #计算形状重建损失（cdL1损失）

        # shape completion loss  （使用第二组点云） 形状补全损失
        rebuild_points = nbr_groups[1] + center_groups[1].unsqueeze(-2)
        idx = pointops.knn(center_groups[1], pred,  int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pred, idx).reshape(B, -1, 3)
        shape_matching_loss = self.shape_matching_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()
        
        # latent reconstruction loss  （使用第三组点云） 潜在空间重建损失
        idx = pointops.knn(center_groups[2], pred, self.group_size)[0]
        nbrs_pred = pointops.index_points(pred, idx)
        _ , feat_recon= self.point_model(nbrs_pred.view(B, 3, -1).detach())
        latent_recon_loss = self.latent_weight * self.latent_criterion(feat_global, feat_recon)
        

        #2d重建损失（使用2d重建点云和第二组点云计算损失
        rebuild_points = nbr_groups[1] + center_groups[1].unsqueeze(-2)
        idx = pointops.knn(center_groups[1], pc_gen,  int(self.nbr_ratio * self.group_size))[0]
        nbrs_pred = pointops.index_points(pc_gen, idx).reshape(B, -1, 3)
        gener_loss = self.shape_matching_weight * self.shape_criterion(rebuild_points.reshape(B, -1, 3), nbrs_pred).mean()

        total_loss = shape_recon_loss + shape_matching_loss + latent_recon_loss + gener_loss

        return total_loss, shape_recon_loss, shape_matching_loss, latent_recon_loss, gener_loss


    def forward(self, x_part, view):
        x_part = x_part.cuda()  #B,2048,3
        view = view.cuda() #B,3,224,224

        pc_feat1, pc_feat2 = self.point_model(x_part.transpose(1, 2).contiguous())
        #pc_feat1: [16,256,512]  pc_feat2: [16,512]
        #im_feat1 = self.img_model(view)
        #img_feat1: [16,256,196]  img_feat2: [16,256]
        #im_feat1 = im_feat1.permute(0, 2, 1)  # [B, N_img, C] B, 196, 256
        pc_feat1 = pc_feat1.permute(0, 2, 1)  # [B, N_pc, C] B, 512, 256

        batch_size = x_part.shape[0]

        view = self.layer1(view) #B,3,224,224 -> B,64,56,56
        view = self.layer2(view) #B,64,56,56 -> B,128,28,28
        view = self.layer3(view) #B,128,28,28 -> B,256,14,14
        view = self.layer4(view) #B,256,14,14 -> B,512,7,7

        x_feat = pc_feat1 #B, 512, 256
        x_1 = torch.empty(size=(x_part.shape[0], 0, 3)).to(x_part.device)

        #基于AXForm的点云生成（仅生成缺失部分的点云）
        for i in range(self.num_branch):
            _x_1 = self.pointgen[i](self.featmap[i](x_feat)) 
            x_1 = torch.cat((x_1, _x_1), dim=1)
        #8*128=1024   x_1: B, 1024, 3

        x_partial = x_part.contiguous()  #保存原始部分点云  B,2048,3
        x_partial = farthest_point_sample(x_partial, 1024)  #从输入的部分点云中采样1024个点 B, 1024, 3
        x_1_1024 = farthest_point_sample(x_1, 1024)#从生成的点云中采样1024个点 B, 1024, 3

        #接下来对图像进行处理  view:B,512,7,7
        de1_view = self.deconv1_1(view).view(batch_size, 256, -1).permute(0, 2, 1)
        #对图像进行反卷积操作，输出256通道的特征图  B,196,256
        x_1_256 = self.fc0_up(x_1_1024)
        #对1024个生成点（缺失部分）进行线性变换，输出256维的特征 B, 1024, 256
        x, _ = self.cross_attn0_0(de1_view, x_1_256, x_1_256)
        #对256维的特征进行交叉注意力操作  x: B，196，256
        de1_view = self.layer_norm0_0(x + de1_view)
        #对交叉注意力操作后的特征进行层归一化  B，196, 256
        x, _ = self.cross_attn0_1(x_1_256, de1_view, de1_view)
        #对256维的特征进行交叉注意力操作  x: B，1024，256
        x_1_256 = self.layer_norm0_1(x + x_1_256)
        #对交叉注意力操作后的特征进行层归一化 B, 1024, 256
        x_1_256 = self.fc0_down(x_1_256)
        #对256维的特征进行线性变换，输出3维的特征   B, 1024, 3
        
        de1_view = de1_view.reshape(batch_size, 256, 14, 14)
        #将256维的特征reshape成14*14的特征图  B,196,256 -> B, 256, 14, 14
        de2_view = self.deconv1_2(de1_view).view(batch_size, 128, -1).permute(0, 2, 1)
        #对14*14的特征图进行反卷积操作，输出128通道的特征图  B,784,128
        x_1_512 = self.fc1_up(x_1_1024)
        #对1024个点（缺失部分）进行线性变换，输出128维的特征  B, 1024, 128
        x, _ = self.cross_attn1_0(de2_view, x_1_512, x_1_512)
        #对128维的特征进行交叉注意力操作  x: B，784，128
        de2_view = self.layer_norm1_0(x + de2_view)
        #对交叉注意力操作后的特征进行层归一化 B，784，128
        x, _ = self.cross_attn1_1(x_1_512, de2_view, de2_view)
        #对128维的特征进行交叉注意力操作  x: B，1024，128
        x_1_512 = self.layer_norm1_1(x + x_1_512)
        #对交叉注意力操作后的特征进行层归一化  B， 1024, 128
        x_1_512 = self.fc1_down(x_1_512)
        #对128维的特征进行线性变换，输出3维的特征  B, 1024, 3

        de2_view = de2_view.reshape(batch_size, 128, 28, 28)
        #将128维的特征reshape成28*28的特征图  B，128, 28, 28
        de3_view = self.deconv1_3(de2_view).view(batch_size, 64, -1).permute(0, 2, 1)
        #对28*28的特征图进行反卷积操作，输出64通道的特征图
        x_1_1024 = self.fc2_up(x_1_1024)
        #对1024个点（缺失部分）进行线性变换，输出64维的特征
        x, _ = self.cross_attn2_0(de3_view, x_1_1024, x_1_1024)
        #对64维的特征进行交叉注意力操作
        de3_view = self.layer_norm2_0(x + de3_view)
        #对交叉注意力操作后的特征进行层归一化
        x, _ = self.cross_attn2_1(x_1_1024, de3_view, de3_view)
        #对64维的特征进行交叉注意力操作
        x_1_1024 = self.layer_norm2_1(x + x_1_1024)
        #对交叉注意力操作后的特征进行层归一化
        x_1_1024 = self.fc2_down(x_1_1024)
        #对64维的特征进行线性变换，输出3维的特征   B， 1024, 3

        x_1 = torch.cat((x_1_256, x_1_512, x_1_1024), dim=1)
        #将多层特征拼接在一起   B，3072，3
        x_1 = farthest_point_sample(x_1, 1024)
        #采样1024个点

        final = torch.cat((x_1, x_partial), dim=1)
        #缺失点云和输入部分点云拼接在一起（重建点云2048个点）

        de3_view = de3_view.reshape(batch_size, 64, 56, 56)
        #将64维的特征reshape成56*56的特征图   B，64, 56, 56
        de3_view = self.conv6_1(de3_view)
        #对56*56的特征图进行卷积操作，输出32通道的特征图  B，32, 56, 56
        de3_view = self.conv6_2(de3_view)
        #对32通道的特征图进行卷积操作，输出3通道的特征图 B,3, 56, 56
        de3_view = de3_view.view(batch_size,56*56,3)
        #将3通道的特征图reshape成56*56*3的特征图  B, 3136, 3
        pc_generate = farthest_point_sample(de3_view, 2048)
        #从56*56*3的特征图中采样2048个点    B，2048，3
        #单视图重建得到的点云（2048个点）

        pred_pcds = self.propagation(final.detach(), final.detach())[0]
        #对重建点云进行上采样细化操作  B，4096，3
        complete = farthest_point_sample(pred_pcds.contiguous() , 2048)
        #对细化后的点云进行采样，得到2048个点，最终完成结果


        return complete, final, pc_generate, pred_pcds





if __name__ == '__main__':
    import config_vipc
    cfg = config_vipc.cfg
   
    x_part = torch.randn(16, 2048,3).cuda()
    view = torch.randn(16, 3, 224, 224).cuda()
    model = NET(cfg).cuda()
    complete, final, pc_generate, pred_pcds = model(x_part, view)

    loss= model.get_loss(x_part, view)


