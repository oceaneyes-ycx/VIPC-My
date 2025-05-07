import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import sys
import inspect

#import config from the parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from config_xmf import params

args = params()


def knn(x, k):
    #K近邻算法，x为输入的点云，k为近邻数
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    #计算点云中点对的内积
    xx = torch.sum(x**2, dim=1, keepdim=True)
    #计算每个点的平方和
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    #计算所有点对之间的成对距离
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
   # idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx#返回k近邻的索引


def get_graph_feature(x, k=20, idx=None):
    #获取图特征，x为输入的点云，k为近邻数，idx为k近邻的索引
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        #在没有提供idx，使用KNN算法计算每个点最近邻的索引
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
    
    #开始提取图特征
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    #返回形状为（B，2C, N, k）的图特征
    #B为batch_size，C为每个点的特征维度（3），N为点云中的点数，k为近邻数(设定为20)
    return feature
    

def batched_index_select(x, dim, index):
    #处理多维张量，并在指定维度上根据索引选择元素。
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
    #动态边池化层-在TopK之前的Relued分数，对点云进行池化操作,压缩特征

    def __init__(self, in_channels, k, ratio=0.5, scoring_fun="tanh", num_points=-1):
        super().__init__()
        self.in_channels = in_channels
        self.k = k
        self.ratio = ratio
        self.score_layer = nn.Conv2d(in_channels * 2, 1, kernel_size=1, bias=True)
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
        x = get_graph_feature(feat, k=self.k, idx=idx)  # (batch_size, α * 2, in_points, self.k)
        x = self.score_layer(x)  # (batch_size, 1, in_points, self.k)

      
        scores = F.relu(x.max(dim=-1, keepdim=False)[0])  # (batch_size, 1, in_points)

        if self.num_points < 0:
            # default - 'num_points' not specified, use 'ratio'
            num_keypoints = math.floor(in_points * self.ratio)
        else:
            # do not use ratio but sample a fixed number of points 'num_points'
            assert self.num_points < in_points, \
                "Pooling more points (%d) than input ones (%d) !" % (self.num_points, in_points)
            num_keypoints = self.num_points
        #print(scores.shape)

        top, top_idxs = torch.topk(scores.squeeze(dim=1), k=num_keypoints, dim=1)
        new_feat = batched_index_select(feat.permute(0, 2, 1), 1, top_idxs) 
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
    def __init__(self, output_channels=40):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    #DGCNN网络用于特征提取
    #xmnet原版DGCNN

    def __init__(self, output_channels=512):
        super(DGCNN, self).__init__()

        self.k = args.k
        self.output_channels = args.emb_dims
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.pool1 = EdgePoolingLayer(in_channels=64, k=args.k_pool1,
                                    ratio=0.5, scoring_fun=args.scoring_fun,
                                    num_points = args.pool1_points)

        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.pool2 = EdgePoolingLayer(in_channels=128, k=args.k_pool2,
                                    ratio=0.5, scoring_fun=args.scoring_fun,
                                    num_points = args.pool2_points)

        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(256, self.output_channels, kernel_size=1, bias=False), #input dimension prima era 512
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        

    def forward(self, x): #x --> (B X F X N)
        #x为输入的点云：torch.size([B,3,2048])
        x = get_graph_feature(x, k=self.k)
        #x: torch.size([B,6,2048,20]) 为获取到的图特征
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        #x1: torch.size([B,64,2048]) 为第一次卷积后的特征
        x = get_graph_feature(x1, k=self.k)
        #x: torch.size([B,128,2048,20]) 为第一次卷积后进而获取的图特征
        x = self.conv2(x)
        #x: torch.size([B,64,2048]) 为第二次卷积后的特征        
        x2 = x.max(dim=-1, keepdim=False)[0]
        #x2: torch.size([B,64,2048])最后一个维度上进行池化操作
        

        #SAG Pool N ---> N/2
        x = self.pool1(x2).permute(0, 2, 1)
        #x: torch.size([B,64,512]) 为第一次池化后的特征
        x = get_graph_feature(x, k=self.k)
        #x: torch.size([B,128,512,20]) 为第一次池化后进而获取的图特征
        x = self.conv3(x)#第三次卷积，
        #x: torch.size([B,128,512,20]) 为第三次卷积后的特征
        x3 = x.max(dim=-1, keepdim=False)[0]
        #x3: torch.size([B,128,512])最后一个维度上进行池化操作
        #SAG Pool N/2 ---> N/4
        x = self.pool2(x3).permute(0, 2, 1)
        #x: torch.size([B,128,128]) 为第二次池化后的特征
        x = get_graph_feature(x, k=self.k)
        #x: torch.size([B,256,128,20]) 为第二次池化后进而获取的图特征
        x = self.conv4(x)
        #x: torch.size([B,256,128,20]) 为第四次卷积后的特征
        x4 = x.max(dim=-1, keepdim=False)[0]
        #x4: torch.size([B,256,128])最后一个维度上进行池化操作
        x = self.conv5(x4)
        #x: torch.size([B,256,128]),最后一次卷积后的特征,也是最终的特征  
        # #256:特征维度，每个点经过网络编码后被表示为一个256维的向量
        # #128:点云数量，原始点云经过两次池化后，点云数量减半，即2048->512->128   
        
        # 原始输出
        point_features = x  # torch.Size([B, 256, 128])

        # 新增的全局池化操作，得到对比学习的特征
        global_features = torch.max(point_features, dim=-1)[0]  # Global Max Pooling
        # global_features = torch.mean(point_features, dim=-1) # 如果需要 GAP，可以替换为此行



        return point_features, global_features


class DGCNN2(nn.Module):
    #修改之后的DGCNN网络，针对对比学习任务
    def __init__(self,  output_channels=512):
        super(DGCNN2, self).__init__()

        self.k = args.k
        self.output_channels = args.emb_dims
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.pool1 = EdgePoolingLayer(in_channels=64, k=args.k_pool1,
                                    ratio=0.5, scoring_fun=args.scoring_fun,
                                    num_points = args.pool1_points)

        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.pool2 = EdgePoolingLayer(in_channels=128, k=args.k_pool2,
                                    ratio=0.5, scoring_fun=args.scoring_fun,
                                    num_points = args.pool2_points)

        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(256, self.output_channels, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.inv_head = nn.Sequential(
                            nn.Linear(args.emb_dims * 2, args.emb_dims),
                            nn.BatchNorm1d(args.emb_dims),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.emb_dims, 256)
                            )

    def forward(self, x): #x --> (B X F X N)

        #x为输入的点云：torch.size([B,3,2048])
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  #图特征：torch.size([B,6,2048,20])  k=20
        x = self.conv1(x) #卷积， B,64,2048,20
        x1 = x.max(dim=-1, keepdim=False)[0]  #最大池化，x1: torch.size([B,64,2048]) 为第一次卷积后的特征

        x = get_graph_feature(x1, k=self.k) #二次提取图特征  B,128,2048,20
        x = self.conv2(x) #二次卷积，B，64，2048,20
        x2 = x.max(dim=-1, keepdim=False)[0] #最大池化，x2: torch.size([B,64,2048])为第二次卷积后的特征

        # SAG Pool N ---> N/2
        x = self.pool1(x2).permute(0, 2, 1) #B，64，512

        x = get_graph_feature(x, k=self.k) # B,128,512,20
        ############# 这里的x是经过池化后的特征，进行卷积操作
        x = self.conv3(x) #B，128，512，20
        x3 = x.max(dim=-1, keepdim=False)[0] # B，128，512

        # SAG Pool N/2 ---> N/4
        x = self.pool2(x3).permute(0, 2, 1) #B，128，128

        x = get_graph_feature(x, k=self.k) # B,256,128,20
        x = self.conv4(x) #B，256，128，20
        x4 = x.max(dim=-1, keepdim=False)[0]# B，256，128

        x = self.conv5(x4)   #2,256,128

        x0 = x # torch.Size([B, 256, 128])，最后一次卷积后的特征,也是最终的局部特征

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # B，256 
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1) # B，256
        x = torch.cat((x1, x2), 1) #B，512 

        inv_feat = self.inv_head(x)  #使用反向头生成对比学习特征，弃用全局池化， B，256 全局特征
        return  x0, inv_feat  #返回： 局部特征，全局特征

class DGCNN3(nn.Module):
    #P2C任务使用，修改返回的全局特征，不再使用对比学习的特征头了
    def __init__(self,  output_channels=512):
        super(DGCNN3, self).__init__()

        self.k = args.k
        self.output_channels = args.emb_dims
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.pool1 = EdgePoolingLayer(in_channels=64, k=args.k_pool1,
                                    ratio=0.5, scoring_fun=args.scoring_fun,
                                    )

        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.pool2 = EdgePoolingLayer(in_channels=128, k=args.k_pool2,
                                    ratio=0.5, scoring_fun=args.scoring_fun,
                                    )

        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(256, self.output_channels, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.inv_head = nn.Sequential(
                            nn.Linear(args.emb_dims * 2, args.emb_dims),
                            nn.BatchNorm1d(args.emb_dims),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.emb_dims, 256)
                            )  #特征变换将一个特征向量，映射为一个维度为256的输出向量。

    def forward(self, x): #x --> (B X F X N)

        #x为输入的点云：torch.size([B,3,2048])    B,3,320
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  #图特征：torch.size([B,6,2048,20])  k=20   B,6,320,20
        x = self.conv1(x) #卷积， B,64,2048,20  B,64,320,20
        x1 = x.max(dim=-1, keepdim=False)[0]  #最大池化，x1: torch.size([B,64,2048]) 为第一次卷积后的特征  B,64,320

        x = get_graph_feature(x1, k=self.k) #二次提取图特征  B,128,2048,20  B,128,320,20 
        x = self.conv2(x) #二次卷积，B，64，2048,20   B,64,320,20
        x2 = x.max(dim=-1, keepdim=False)[0] #最大池化，x2: torch.size([B,64,2048])   B,64,320为第二次卷积后的特征

        # SAG Pool N ---> N/2
        x = self.pool1(x2).permute(0, 2, 1) #B，64，512  B,64,160

        x = get_graph_feature(x, k=self.k) # B,128,512,20  B,128,160,20
        ############# 这里的x是经过池化后的特征，进行卷积操作
        x = self.conv3(x) #B，128，512，20 B,128,160,20 
        x3 = x.max(dim=-1, keepdim=False)[0] # B，128，512   B,128,160

        # SAG Pool N/2 ---> N/4
        x = self.pool2(x3).permute(0, 2, 1) #B，128，128  B,128,80

        x = get_graph_feature(x, k=self.k) # B,256,128,20   B,256,80,20
        x = self.conv4(x) #B，256，128，20  B,256,80,20
        x4 = x.max(dim=-1, keepdim=False)[0]# B，256，128 B,256,80

        x = self.conv5(x4)   #  B,256,128  B,256,80  

        x0 = x # torch.Size([B, 256, 128])， B,256,80最后一次卷积后的特征,也是最终的局部特征

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # 对每个样本的特征在最后一个维度上进行自适应最大池化 B，256 B, 256 
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1) # 对每个样本的特征在最后一个维度上进行自适应平均池化 B，256 B，256
        x = torch.cat((x1, x2), 1) #B，512  B，512全局特征由最大池化输出和平均池化输出组合而成

        #inv_feat = self.inv_head(x)  #使用反向头生成对比学习特征，弃用全局池化， B，256  B,256全局特征
        return  x0, x  #返回： 局部特征，全局特征












if __name__ == '__main__':


    x = torch.randn(2, 3, 2048).cuda()
    model = DGCNN2().cuda()
    out1,out2 = model(x)
    #out:torch.size([1,256,128])

    #DGCNN1: 2,256,128    2,256
    #DGCNN2: 2,256,128   2,256
    print('dgcnnout:', out1.shape) 
                    
   