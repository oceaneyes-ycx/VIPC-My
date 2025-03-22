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
from torchvision.models import resnet50, resnet18, resnet101

from config_xmf import params

from lightly.loss.ntx_ent_loss import NTXentLoss
from utils.average_meter import AverageMeter, AverageMeter2

from datasets import data_utils as d_utils

from encoder_image.resnet_xmf import ResNet as ResNet_xmf

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
        


if __name__ == '__main__':

   
    x_part = torch.randn(16, 2048,3).cuda()
    view = torch.randn(16, 3, 224, 224).cuda()
    model = End2endNetwork().cuda()
    out = model(x_part, view)
    print(out.shape) 
