
import math

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os

import os
import sys
import inspect

#import config from the parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from config_vipc import cfg
from utils.loss_utils import farthest_point_sample

def knn(x, k):
    #Knn算法，用于计算输入点云数据的k个最近邻点
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
   # idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx#返回k个最近邻点的索引(点云索引)


def get_graph_feature(x, k=20, idx=None):
    #获取图特征
    #x: (batch_size, num_dims, num_points)点云数据
    #k: int, number of neighbors
    #idx: (batch_size, num_points, k)点云索引
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)#(batch_size, num_dims, num_points)
    if idx is None:#未提供点云索引的情况下，使用knn算法计算点云索引
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1)*num_points#(batch_size, 1, 1),基础索引

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature#返回图特征


def batched_index_select(x, dim, index):
    #接受输入张量x，维度dim和索引index，返回一个新的张量，其形状与索引相同
    #用于从输入张量x的指定维度dim中选择索引index的元素
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
    '''动态边缘池化层，目的是在图神经网路中对特征进行池化操作'''

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
    #pointnet网络
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
    #接受K1和N作为输入，返回一个映射网络
    #经过全连接层和批量归一化层，返回处理后的特征x
    def __init__(self, K1,N=256):
        super(MappingNet, self).__init__()
        self.K1 = K1
        self.N=N

        self.fc1 = nn.Linear(self.N, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, self.K1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(self.K1)

    def forward(self, x):
        
        x = F.relu(self.bn1(self.fc1(x).permute(0,2,1)))
       
        x = x.permute(0,2,1)
        x = F.relu(self.bn2(self.fc2(x).permute(0,2,1)))
      
        x = x.permute(0,2,1)
        x = F.relu(self.bn3(self.fc3(x).permute(0,2,1)))
        x = x.permute(0,2,1)
        x = F.relu(self.bn4(self.fc4(x).permute(0,2,1)))
        
        x = x.permute(0,2,1)
       
        return x

class AXform(nn.Module):
    #通过AXform网络，一系列的卷积和归一化操作，返回处理后的特征x
    def __init__(self, K1, K2, N):
        super(AXform, self).__init__()
        self.K1 = K1
        self.K2 = K2
        self.N = N  
        self.conv1 = nn.Conv1d(K1, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.softmax = nn.Softmax(dim=2)
        self.conv4 = nn.Conv1d(K2, 3, 1)

    def forward(self, x):
        x_base = x
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
       
        x_weights = self.softmax(x)
        
        x = torch.bmm(x_weights, x_base)  
        
        x = x.transpose(1, 2).contiguous()
        x = self.conv4(x)
        x = x.transpose(1, 2).contiguous()

        return x
    
class Decoder_Network(nn.Module):
    def __init__(self, K1, K2, N):
        #K1: 输入特征的维度，K1定义了MappingNet的输出维度以及AXform的输入维度
        #K1本质上代表中间特征表示的维度，控制特征的丰富程度
        #K2: 用于控制生成点云的数量，再AXform中定义了输出维度
        #N: 在MappingNet中定义了输入维度，需要与输入特征的最后一个维度匹配
        '''
        充分处理输入特征(通过N维度)
        提取有效的中间表示(通过K1维度)
        生成适量的输出点云(通过K2维度)
        '''
        super(Decoder_Network, self).__init__()
        
        self.num_branch = 8#分支数量
        self.K1 = K1
        self.K2 = K2
        self.N = N
        
        self.featmap = nn.ModuleList([MappingNet(self.K1,self.N) for i in range(self.num_branch)])
        self.pointgen = nn.ModuleList([AXform(self.K1, self.K2, self.N) for i in range(self.num_branch)])
       
    def forward(self, x, x_part):
       
        x_partial = x_part.contiguous() # .contiguous() to make fps work
        x_partial = farthest_point_sample(x_partial, 1024)
        x_feat = x
        x_1 = torch.empty(size=(x_part.shape[0], 0, 3)).to(x_part.device)
        
        x_branch = []
        for i in range(self.num_branch):
            _x_1 = self.pointgen[i](self.featmap[i](x_feat)) 
            x_1 = torch.cat((x_1, _x_1), dim=1) 
            x_branch.append(_x_1)

        x_coarse = torch.cat((x_1, x_partial), dim=1)

        return x_coarse

if __name__ == '__main__':

    x_part = torch.randn(16, 2048, 3).cuda()#部分点云数据,[16,2048,3]
    x = torch.randn(16, 128, 256).cuda()#特征数据,[16,128,256]
    model = Decoder_Network(128,128,256).cuda()
    out = model(x, x_part)
    #print(model)
    print(f"Output shape: {out.shape}")
    #summary(model, input_size=[(128, 256), (2048, 3)])



    
    