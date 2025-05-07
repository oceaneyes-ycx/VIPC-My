import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
from datetime import datetime, timedelta


from decoder.utils.utils import *
import os
import sys
import inspect

#import config from the parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from config_xmf import params

opt = params()



def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
   # idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


def batched_index_select(x, dim, index):
    for i in range(1, len(x.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(x.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(x, dim, index)


class EdgePoolingLayer(nn.Module):
    """ Dynamic Edge Pooling layer - Relued Scores before TopK"""

    def __init__(self, in_channels, k, ratio=0.5, scoring_fun="tanh", num_points=-1):
        super().__init__()
        self.in_channels = in_channels
        self.k = k
        self.ratio = ratio
        self.score_layer = nn.Conv2d(
            in_channels * 2, 1, kernel_size=1, bias=True)
        self.scoring_fun = scoring_fun
        self.num_points = num_points

    def __str__(self):
        return 'EdgePoolingLayer(in_channels=' + str(self.in_channels) + ', k=' + str(self.k) + ', ratio=' + str(
            self.ratio) + ', scoring_fun=' + str(self.scoring_fun) + ', num_points=' + str(self.num_points) + ')'

    def forward(self, feat, idx=None):
        batch_size, dim, in_points = feat.size()  # (batch_size, α, in_points)
        assert dim == self.in_channels

        # Dynamic Edge Conv
        # Re-computing graph before pooling
        # (batch_size, α * 2, in_points, self.k)
        x = get_graph_feature(feat, k=self.k, idx=idx)
        x = self.score_layer(x)  # (batch_size, 1, in_points, self.k)

        # scores = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1, in_points)
        # ReLU applied to Scores
        # (batch_size, 1, in_points)
        scores = F.relu(x.max(dim=-1, keepdim=False)[0])

        if self.num_points < 0:
            # default - 'num_points' not specified, use 'ratio'
            num_keypoints = math.floor(in_points * self.ratio)
        else:
            # do not use ratio but sample a fixed number of points 'num_points'
            assert self.num_points < in_points, \
                "Pooling more points (%d) than input ones (%d) !" % (
                    self.num_points, in_points)
            num_keypoints = self.num_points

        top, top_idxs = torch.topk(scores.squeeze(), k=num_keypoints, dim=1)
        new_feat = batched_index_select(feat.permute(
            0, 2, 1), 1, top_idxs)  # (batch, num_keypoints, α)
        top = top.unsqueeze(2)

        # Apply scoring function!
        if self.scoring_fun == 'tanh':
            new_feat = new_feat * torch.tanh(top)
        elif self.scoring_fun == 'softmax':
            new_feat = new_feat * F.softmax(top, dim=1)
        elif self.scoring_fun == 'leaky-relu':
            new_feat = new_feat * F.leaky_relu(top, negative_slope=0.001)

        return new_feat




class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, dim=2)
        return x


class MappingNet(nn.Module):
    def __init__(self, K1):
        super(MappingNet, self).__init__()
        self.K1 = K1     #输出特征的维度

        #四个全连接层
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, self.K1)

        #每个全连接层后面都跟一个batchnorm，用于加速训练
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(self.K1)

    def forward(self, x):
        '''
        输入特征x依次经过四个全连接层，每个全连接层后面都跟一个batchnorm
        每次经过全连接层和batchnorm之后后都使用relu激活函数
        '''


        #激活函数relu
        x = F.relu(self.bn1(self.fc1(x).permute(0,2,1)))
       
       #调整维度
        x = x.permute(0,2,1)
        x = F.relu(self.bn2(self.fc2(x).permute(0,2,1)))
      
        x = x.permute(0,2,1)
        x = F.relu(self.bn3(self.fc3(x).permute(0,2,1)))
        x = x.permute(0,2,1)
        x = F.relu(self.bn4(self.fc4(x).permute(0,2,1)))
        
        x = x.permute(0,2,1)
       
        return x


class AXform(nn.Module):
    def __init__(self, K1, K2, N):
        super(AXform, self).__init__()
        self.K1 = K1#输入点云特征的维度
        self.K2 = K2#输出点云特征的维度
        self.N = N  #点云的数量，即点的个数

       
        #定义三个卷积层，kernel_size=1,逐点卷积
        self.conv1 = nn.Conv1d(K1, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)  

        #定义两个batchnorm层，用于加速训练
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        
        #softmax层,用于计算权重(注意力机制)
        self.softmax = nn.Softmax(dim=2)

        #使用卷积层将特征映射到K2维度，即生成点云
        self.conv4 = nn.Conv1d(K2, 3, 1)

    def forward(self, x):
        x_base = x #保存输入特征副本
        x = x.transpose(1, 2).contiguous()#调整维度
        x = F.relu(self.bn1(self.conv1(x)))#经过第一个卷积层
        x = F.relu(self.bn2(self.conv2(x)))#经过第二个卷积层
        x = self.conv3(x)#经过第三个卷积层
       
        x_weights = self.softmax(x)#计算注意力权重
        
        x = torch.bmm(x_weights, x_base) 
        #使用注意力调整特征，即将注意力权重和输入特征相乘
        #torch.bmm()是批量矩阵乘法，即对每个batch进行矩阵乘法，来生成新的特征矩阵 
        
        x = x.transpose(1, 2).contiguous() #调整维度
        x = self.conv4(x) #生成点云
        x = x.transpose(1, 2).contiguous()

    
        return x






class Decoder_Network(nn.Module):
    #解码器网络(基于AXFORM)
    def __init__(self):
        super(Decoder_Network, self).__init__()
        
        #解码器网络的参数
        self.num_branch = opt.num_branch   #8  AXFORM的分支数
        self.K1 = opt.K1   #64  中间特征维度，表示经过 MappingNet 映射后的特征维度
        self.K2 = opt.K2   #64  在生成点之前的中间特征维度（AXform内部使用）
        self.N = opt.N    #128 每个分支生成的点数
        #128 * 8 = 1024
        #1024+1024 = 2048

        #定义两个modulelist       
        self.featmap = nn.ModuleList([MappingNet(self.K1) for i in range(self.num_branch)])
        #mappingnet用于将输入的特征映射到K1维度的特征（将点云和图像特征对齐）        
        self.pointgen = nn.ModuleList([AXform(self.K1, self.K2, self.N) for i in range(self.num_branch)])
        #pointgen用于生成点云
       

    def forward(self, x, x_part):
        #联合特征 x:  B,512,256   B，80，256     原始的部分点云 x_part: B,2048,3
        x_partial = x_part.contiguous() # .contiguous() to make fps work
        x_partial = farthest_point_sample(x_partial, 1024)
        #对原始的部分点云进行采样，得到1024个点   B,1024,3

        x_feat = x  #B,512,256

        x_1 = torch.empty(size=(x_part.shape[0], 0, 3)).to(x_part.device)
        #x_1用于存储每个分支生成的点云，初始化为空    B,0,3        

        x_branch = [] #空列表，用于存储每个分支生成的点云

        for i in range(self.num_branch):
            _x_1 = self.pointgen[i](self.featmap[i](x_feat)) 
            #使用AXForm 生成点云 B,128,3 
            x_1 = torch.cat((x_1, _x_1), dim=1) 
            #将生成的点拼接到x_1上， 每一个batch会增加128个点
            x_branch.append(_x_1)

        x_coarse = torch.cat((x_1, x_partial), dim=1)  #1024+1024=2048 

        
        
        
        
        return x_coarse


if __name__ == '__main__':

    x_part = torch.randn(16, 2048, 3).cuda()
    x = torch.randn(16, 128, 256).cuda()
    model = Decoder_Network().cuda()
    out = model(x, x_part)
    
    