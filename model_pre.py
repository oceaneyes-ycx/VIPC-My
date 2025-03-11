'''
完整模型代码,对比学习进行VIPC任务
点云编码器和图像编码器使用预训练的编码器(参数冻结)
特征融合模块和解码器模块和XMFNet保持一致
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
from encoder_dgcnn.dgcnn import DGCNN, DGCNN2

import torchvision.transforms as transforms
from models.resnet import ResNet, ResNet2
from torchvision.models import resnet50, resnet18, resnet101

from config_xmf import params

p = params()

class Network(nn.Module):
    def __init__(self, pretrained_model_path="./weights/best_model.pth"):
        super(Network, self).__init__()

        #点云和图像的编码器
        #加载预训练的编码器
        checkpoint = torch.load(pretrained_model_path)
        self.img_model = ResNet2(resnet18(), feat_dim = 512)
        self.point_model = DGCNN2()

        self.point_model.load_state_dict(checkpoint['point_model_state_dict'])
        self.img_model.load_state_dict(checkpoint['img_model_state_dict'])

        # 将模型设置为评估模式
        self.img_model.eval()
        self.point_model.eval()

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
        with torch.no_grad():
            #将输入数据转移到GPU
            x_part = x_part.cuda()
            view = view.cuda()
            #提取图像和点云的特征(注意点云跟对比学习训练用的全局特征不一样，这里要用空间特征)
            pc_feat, _  = self.point_model(x_part.transpose(1, 2).contiguous())  # 提取点云特征
            im_feat, _  = self.img_model(view)  # 提取图像特征
            #pc_feat:[16, 256, 128]  im_feat:[16, 256, 196]


        im_feat = im_feat.permute(0, 2, 1)  # 调整图像特征维度   [16, 196, 256]
        pc_feat = pc_feat.permute(0, 2, 1)  # 调整点云特征维度  [16, 128, 256]

        #特征融合模块
        x, _ = self.cross_attn1(pc_feat, im_feat, im_feat)
        pc_feat = self.layer_norm1(x + pc_feat)
        x, _ = self.self_attn1(pc_feat, pc_feat, pc_feat)   #x:[16, 128, 256]
        pc_feat = self.layer_norm2(x + pc_feat)   #pc_feat:[16, 128, 256]

        pc_skip = pc_feat  #保存pc_feat，用于后续的融合   [16, 128, 256]

        x, _ = self.cross_attn2(pc_feat, im_feat, im_feat)
        pc_feat = self.layer_norm3(x + pc_feat)
        x, _ = self.self_attn2(pc_feat, pc_feat, pc_feat)  #x:[16, 128, 256]
        pc_feat = self.layer_norm4(x + pc_feat)   #pc_feat:[16, 128, 256]

        x, _ = self.cross_attn3(pc_feat, pc_skip, pc_skip)  #x:[16, 128, 256]
        pc_feat = self.layer_norm5(x + pc_feat)   #pc_feat:[16, 128, 256]
        #x_part = x_part.permute(0, 2, 1)  # B x 3 x N ----> B x N x 3  [16, 2048, 3]

        final = self.decoder(pc_feat, x_part)

        return final
        
        


if __name__ == '__main__':

   
    x_part = torch.randn(16, 2048,3).cuda()
    view = torch.randn(16, 3, 224, 224).cuda()
    model = Network(pretrained_model_path="./weights/best_model.pth").cuda()
    out = model(x_part, view)
    print(out.shape) 
