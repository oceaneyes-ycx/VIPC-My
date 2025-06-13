import torch
import torch.nn as nn
from extensions.chamfer_dist import ChamferDistanceL1
from models.model_utils import MLP_CONV, Transformer, PointNet_SA_Module_KNN,ImageSelfAttentionBlock
from decoder.utils.utils import *
from torchvision.models import resnet18
from torch import einsum
from models.layers.group import KNNGroup, QueryAndGroup, get_aggregation_feautres
from models.layers.conv import  create_convblock2d,create_linearblock
from models.layers.local_aggregation import CHANNEL_MAP
from models.layers.norm import create_norm
from itertools import repeat
from models.layers.subsample import furthest_point_sample, random_sample
from config_vipc import cfg
import collections.abc
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):

        B, N, C = x.shape
        _, NK, _ = y.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  #  B, H, N, C
        k = self.k(y).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  #  B, H, NK, C
        v = self.v(y).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  #  B, H, NK, C

        attn = (q @ k.transpose(-2, -1)) * self.scale #  B, H, N, NK
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    


class CrossFormer(nn.Module):
    #交叉注意力transformer模块

    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.1):
        super().__init__()
        self.bn1 = nn.LayerNorm(dim)#对输入x进行归一化
        self.bn2 = nn.LayerNorm(dim)#对输入y进行归一化
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        #多头注意力机制，输入为(batch_size, seq_length, embedding_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.bn3 = nn.LayerNorm(dim)
        #前馈再归一化层
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim, out_dim),
        ) #前馈网络，输出维度为out_dim

    def forward(self, x, y):

        #x：点云特征
        #y：图像特征
        short_cut = x#残差连接

        #首先对输入x和y进行归一化处理，提升训练稳定性
        x = self.bn1(x)
        y = self.bn2(y)

        #跨注意力机制，使用x去关注y的特征，实现跨模态/跨特征融合
        x = self.attn(query=x, key=y, value=y)[0]

        x = short_cut + self.drop_path(x)
        #残差连接，注意力输出和原始x相加，保留原始信息
        x = x + self.drop_path(self.ffn(self.bn3(x)))
        return x
    

class LSTNet(nn.Module):
    #LSTNET: Local Symmetric Transformer Network
    def __init__(self, out_dim=512):
        super(LSTNet, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        #使用KNN修改之后的pointnet++，用于点云的采样和特征提取
        self.transformer_1 = Transformer(128, dim=64)
        self.expanding = MLP_CONV(in_channel=128, layer_dims=[256, out_dim])     
        
        self.mlp = nn.Sequential(
                nn.Linear(512*2, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 9+3)
        )

    def forward(self, point_cloud):
        #input: point_cloud (B, 3, N)  部分点云
        b = point_cloud.shape[0]   #batch size
        l0_xyz = point_cloud   
        l0_points = point_cloud

        # get the key points and its features
        keypoints, keyfeatures, _ = self.sa_module_1(l0_xyz, l0_points)  
        # (B, 3, 512), (B, 128, 512)  通过pointnetknn采样512个关键点和对应的特征   
        keyfeatures = self.transformer_1(keyfeatures, keypoints) 
        # B,128,512  使用Transformer对关键点特征进行处理,获得注意力特征

        feat = self.expanding(keyfeatures)
        #特征维度从128扩展到512
        feat = feat.transpose(2, 1).contiguous()
        # B,512,512  转置为(B,512,512)的形式
        gf_feat = feat.max(dim=1, keepdim=True)[0]
        # B,1,512  获取全局特征，取512个特征的最大值
        feat = torch.cat([feat, gf_feat.repeat(1, feat.size(2), 1)], dim=-1) 
        # B,640,512 将全局特征与局部特征拼接在一起

        ret = self.mlp(feat)   

        #预测每个点的变换参数（旋转矩阵R和位移向量T）
        R = ret[:, :, :9].view(b, 512, 3, 3)  #B,512,3,3  3x3旋转矩阵
        T = ret[:, :, 9:]  #B,512,3  3维位移向量
        symmetry_points = torch.matmul(keypoints.transpose(2, 1).contiguous().unsqueeze(2), R).view(b, 512, 3)
        # B,512,3  将关键点云与旋转矩阵相乘，得到对称点云
        symmetry_points = symmetry_points + T  #应用变换生成对称点云
        symmetry_points = symmetry_points.transpose(2, 1).contiguous()
        # B,3,512  转置为(B,3,512)的形式
        coarse = torch.cat([symmetry_points, keypoints], dim=-1) 
        # B, 1024, 3  将对称点云和关键点云拼接在一起，形成粗糙的点云
        return coarse, symmetry_points, keyfeatures
        #返回： 粗糙点云，堆成点云，关键点特征



def _ntuple(n):
    '''私有工厂函数，输入一个整数n，
    返回一个函数，这个函数接受一个参数x，
    这个函数是
    '''
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            # 如果x是一个可迭代对象，但不是字符串
            return tuple(x)#返回x的元组
        return tuple(repeat(x, n))
        #x不是可迭代对象或者是字符串时
        #返回一个元组，元组中有n个x
    return parse#返回parse函数,用于将输入的参数转换为一个元组

to_1tuple = _ntuple(1)
#接受一个参数x，x是一个可迭代对象且不是字符串时
#返回一个元组，仅包含一个x（就是把单x转换成了一个元组形式）
#否则，返回一个元组，元组中有一个x
to_2tuple = _ntuple(2)
#接受一个参数x，x是一个可迭代对象且不是字符串时
#返回一个元组，包含两个x
#否则，返回一个元组，元组中有两个x
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple#别名


def gram_matrix(tensor):
    #计算输入张量的格拉姆矩阵
    gram = einsum('b i n, b n j  -> b i j', tensor.transpose(1,2),tensor)
    return gram


class img_tokenizer(nn.Module):
    '''2d image tokenzier
    这个类的设计灵感来自于视觉 
    Transformer（ViT）模型中的图像分块和嵌入过程。
    将输入图像分割成固定大小的补丁，
    并将这些补丁嵌入到一个高维空间中。
    '''
    def __init__(self, img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=768,
                 norm_layer=None, 
                 flatten=True, 
                 bias=True):
        '''初始化函数，继承nn.Module'''
        '''
        图像大小：img_size，224*224
        补丁大小：patch_size，16*16
        输入通道数：in_chans，3
        嵌入维度：embed_dim，768
        归一化层：norm_layer，None
        是否展平：flatten，True
        偏置：bias，True
        '''
        super().__init__()#继承父类的初始化函数
        img_size = to_2tuple(img_size)
        #将img_size转换为一个元组，元组中有两个img_size
        patch_size = to_2tuple(patch_size)
        #将patch_size转换为一个元组，元组中有两个patch_size
        self.img_size = img_size#赋值
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        #计算图像被分割成补丁之后的网格大小
        #图像大小（高度）//(整数除法)高度方向补丁大小，得到高度方向补丁数量
        #图像大小（宽度）//(整数除法)宽度方向补丁大小，得到宽度方向补丁数量
        #得到一个元组，元组中有两个元素，分别是高度方向补丁数量和宽度方向补丁数量
        #也就是补丁网格的大小
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        #计算补丁的数量，即网格大小的乘积
        self.flatten = flatten#赋值

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=bias)
        #卷积层，输入通道数为in_chans，输出通道数为embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        #归一化层，输入通道数为embed_dim

    def forward(self, x):
        #前向传播函数
        x = self.proj(x) #卷积层
        _, _, H, W = x.shape #获取x的形状(宽度和高度)
        if self.flatten:#如果flatten为True，需要展平
            x = x.flatten(2).transpose(1, 2)  
            # BCHW -> BNC
            #B：batch_size，C：通道数，H：高度，W：宽度
            #N：补丁数量
        x = self.norm(x)#归一化
        return x, H, W#返回处理后的补丁，高度，宽度

class pc_tokenizer(nn.Module):
    def __init__(self,
                 sample_ratio=0.0625,
                 scale=4,
                 group_size=32,
                 in_channels=3,
                 layers=4,
                 embed_dim=384,
                 subsample='fps',  # random, FPS
                 group='ballquery',
                 normalize_dp=False,
                 radius=0.1,
                 feature_type='dp_df',
                 relative_xyz=True,
                 norm_args={'norm': 'in2d'},
                 act_args={'act': 'relu'},
                 conv_args={'order': 'conv-norm-act'},
                 reduction='max',
                 return_group_points=False,
                 **kwargs
                 ):
        #初始化函数
        super().__init__()
        self.sample_ratio = sample_ratio
        #采样比例，默认为0.0625
        self.group_size = group_size
        #分组大小，默认为32
        self.scale=scale
        #缩放比例，默认为4
        self.feature_type = feature_type
        #特征类型，默认为dp_df
        # subsample layer and group layer

        #选择采样方法，FPS或者随机采样
        if subsample.lower() == 'fps':
            self.sample_fn = furthest_point_sample
        elif 'random' in subsample.lower():
            self.sample_fn = random_sample


        #选择分组方法，ballquery或者knn
        self.group = group.lower()
        if 'ball' in self.group or 'query' in self.group:
            self.grouper = QueryAndGroup(nsample=self.group_size,
                                         relative_xyz=relative_xyz, normalize_dp=normalize_dp,
                                         radius=radius)
        elif 'knn' in self.group.lower():
            self.grouper = KNNGroup(self.group_size, relative_xyz=relative_xyz, normalize_dp=normalize_dp)
        else:
            raise NotImplementedError(f'{self.group.lower()} is not implemented. Only support ballquery, knn')

        # stages
        #构建多层卷积网络

        #首先，计算需要的卷积阶段数
        '''log(1/sample_ratio, scale) 计算的是在给定采样率 sample_ratio 和缩放因子 scale 的情况下，需要经过多少个卷积阶段。这个公式的含义如下：
        sample_ratio：表示输入数据在经过卷积操作后的采样率。例如，如果采样率是0.5，那么经过一次卷积操作后，输入数据的分辨率会减半。
        scale：表示每次卷积操作后，输入数据分辨率减小的比例。例如，如果 scale 是2，那么每次卷积操作后，输入数据的分辨率会减半。 
        通过计算 log(1/sample_ratio, scale)，我们可以得到需要经过多少次卷积操作才能将输入数据的分辨率降低到1。例如，如果 sample_ratio 是0.5，scale 是2，
        那么 log(1/0.5, 2) 的结果是2，这意味着需要经过2次卷积操作才能将输入数据的分辨率降低到1。'''
        stages = int(math.log(1/sample_ratio, scale))
        #计算每个阶段的嵌入维度
        embed_dim = int(embed_dim // 2 ** (stages-1))


        #开始构建卷积层
        #self.convs是一个ModuleList，用于存储多个卷积层
        self.convs = nn.ModuleList()
        #channel_list用于存储每个阶段的输入通道数
        self.channel_list = [in_channels]
        #循环构建卷积块
        #循环stages次
        for _ in range(int(stages)):
            # convolutions

            #根据映射表CHANNEL_MAP，根据feature_type确定初始通道数
            #默认feature_type为dp_df,'dp_df': lambda x: x + 3,则in_channels+3
            #embed_dim是当前阶段的嵌入维度，layers是卷积层的数量。
            channels = [CHANNEL_MAP[feature_type](in_channels)] + [embed_dim] * (layers // 2) + [embed_dim * 2] * (
                    layers // 2 - 1) + [embed_dim]
            
            #构建第一部分卷积层，包含layers//2个卷积块
            conv1 = []
            for i in range(layers // 2):
                #构建卷积块，输入通道数为channels[i]，输出通道数为channels[i+1]
                #create_convblock2d是一个函数，用于构建卷积块
                #norm_args和act_args是归一化层和激活函数的参数,设定为仅在最后一层为none，避免出现重复的归一化和激活函数
                conv1.append(create_convblock2d(channels[i], channels[i + 1],
                                                norm_args=norm_args if i!=(layers//2-1) else None,
                                                act_args=act_args if i!=(layers//2-1) else None,
                                                **conv_args))
            conv1 = nn.Sequential(*conv1)#将多个卷积块组合成一个序列



            #构建第二部分卷积层，包含layers//2个卷积块
            channels[layers // 2] *= 2#中间层的通道数乘以2
            conv2 = []
            for i in range(layers // 2, layers):
                conv2.append(create_convblock2d(channels[i], channels[i + 1],
                                                norm_args=norm_args,
                                                act_args=act_args,
                                                **conv_args
                                                ))
            conv2 = nn.Sequential(*conv2)

            #将两部分卷积层组合成一个ModuleList
            self.convs.append(nn.ModuleList([conv1, conv2]))
            #更新channel_list，添加当前阶段的嵌入维度
            self.channel_list.append(embed_dim)

            #更新输入通道数和嵌入维度，为下一阶段做准备
            in_channels = embed_dim
            embed_dim *= 2

        # reduction layer 池化操作，用于特征降维
        #根据reduction确定池化操作
        #如果reduction是以下的几种情况,使用平均池化
        if reduction in ['mean', 'avg', 'meanpool', 'avgpool']:
            self.pool = lambda x: torch.mean(x, dim=-1, keepdim=True)
        else:#不然使用最大池化
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=True)[0]
        #输出通道数，channel列表的最后一个元素。
        self.out_channels = channels[-1]
        self.re_gr_p=return_group_points#是否返回group_points

    def forward(self, p, f=None):
        B, N, _ = p.shape[:3]#获取batch_size，点数，特征数
        out_p, out_f = [p], [f]#初始化输出点和特征
        #循环处理每个卷积层
        for convs in self.convs:
            # Progressive downsampling
            #进行逐步下采样
            cur_p, cur_f = out_p[-1], out_f[-1]
            idx = self.sample_fn(cur_p, int(N //self.scale)).long()
            N = N // self.scale#更新邻居数
            center_p = torch.gather(cur_p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            center_f = torch.gather(cur_f, 2, idx.unsqueeze(1).expand(-1, cur_f.shape[1], -1))

            # query neighbors.
            #获取邻居
            dp, fj = self.grouper(center_p, cur_p, cur_f)
            fj = get_aggregation_feautres(center_p, dp, center_f, fj, self.feature_type)

            # graph convolutions
            #图卷积
            fj = convs[0](fj)
            fj = torch.cat(
                [self.pool(fj).expand(-1, -1, -1, self.group_size),
                fj],
                dim=1)

            # output
            out_f.append(self.pool(convs[1](fj)).squeeze(-1))
            out_p.append(center_p)
        if self.re_gr_p:#如果需要返回group_points
            return out_p, out_f,dp
        else:#不需要返回group_points
            return out_p, out_f#返回输出点和特征

class transfer_loss_shared_encoder_sym(nn.Module):
    #多模态编码器
    def __init__(self,
                 embed_dim=1024, 
                 block_head=16, 
                 depth=3,
                 img_patch_size=14,
                 sample_ratio=0.25, 
                 scale=2,
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 act_args={'act': 'gelu'},
                 pc_h_hidden_dim=384,
                 fuse_layer_num=1,
                 **kwargs
                 ):
        '''
        输入参数：
        embed_dim：嵌入维度，默认为192
        block_head：块头，默认为12
        depth：深度，默认为3，transformer中的block数量
        img_patch_size：图像补丁大小，默认为14
        sample_ratio：采样比例，默认为0.125
        scale：缩放，默认为2
        '''
        super().__init__()
        from timm.models.vision_transformer import Block#transformer搭载模块
        self.im_to_token=img_tokenizer(patch_size=img_patch_size,embed_dim=embed_dim)
        self.pc_to_token=pc_tokenizer(sample_ratio=sample_ratio,scale=scale,embed_dim=pc_h_hidden_dim)
        self.embed_dim=embed_dim#赋值
        self.fuse_layer_num=fuse_layer_num
        
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=block_head,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(depth)])
        '''
        dim：嵌入维度，默认为embed_dim
        num_heads：多头自注意力机制的头数，默认为block_head
        mlp_ratio：MLP的放大倍数，默认为4
        qkv_bias：在计算query、key和value时是否使用偏置，默认为True
        norm_layer：归一化层，默认为nn.LayerNorm
        act_layer：激活函数，默认为nn.GELU
        '''
        
        #跨模态注意力机制
        self.cross_layer=nn.MultiheadAttention(embed_dim, block_head, batch_first=True)
        #pytorch中的多头注意力机制
        #与block中的自注意力不同，这里可以被设计为跨模态注意力机制
        self.cross_norm=nn.LayerNorm(embed_dim)

        self.depth=depth
        self.pc_norm = create_norm(norm_args, self.embed_dim)
        #点云归一化层

        self.im_norm = create_norm(norm_args, self.embed_dim)
        #图像归一化层
        self.proj = nn.Linear(self.pc_to_token.out_channels, self.embed_dim)
        #线性层，输入通道数为pc_to_token的输出通道数，输出通道数为embed_dim
        #用于将点云数据的特征映射到与图像数据相同的嵌入维度

        self.pc_pos_embed = nn.Sequential(
            create_linearblock(3, 128, norm_args=None, act_args=act_args),
            nn.Linear(128, self.embed_dim)
        )
        #点云位置嵌入层，用于将点云数据的位置信息嵌入到嵌入空间中
        #对点云数据的位置信息进行编码

        self.mlp = nn.Sequential(
                nn.Linear(512*2, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 9+3)
        )#用于预测对称点的MLP网络


    def forward(self,pc,im):
        #前向传播函数，接收参数：pc，点云数据（张量）；im，图像数据（张量）
        b=pc.size(0)
        p_list,x_list=self.pc_to_token(pc,pc.transpose(1,2).contiguous())
        #点云数据token化，返回点云数据和特征  
        # p_list:B,2048,3 B,1024,3 B,512,3 B,256,3  代表不断下采样得到的点云数据
        # x_list:B,3,2048 B,512,1024 B,1024,512 B,2048,256  代表不断下采样得到的点云特征
        cent_p,pc_f=p_list[-1],self.proj(x_list[-1].transpose(1, 2))
        #获取最后一层的点云数据（作为keypoints）和点云特征（维度调整）
        #cent_p:B,512,3  pc_f:B,512,192  代表最后一层的点云数据和特征
        #
        im_f,_,_=self.im_to_token(im)
        #im_f:B,256,1024
        #图像数据token化

        pc_pos_emd = self.pc_pos_embed(cent_p)
        
        #计算点云数据的位置嵌入 B,256,1024


        #存储每个深度层的点云数据和图像数据
        pc_f_list=[]
        im_f_list=[]

        for i in range(self.depth):
            #循环处理每个深度层
            pc_f = self.blocks[i](pc_f + pc_pos_emd)
            im_f = self.blocks[i](im_f)
            pc_f_list.append(pc_f)
            im_f_list.append(im_f)
        
        
        pc_f=self.pc_norm(pc_f)#点云特征归一化
        
        im_f=self.im_norm(im_f)#图像特征归一化
        
        x, _ = self.cross_layer(pc_f, im_f, pc_f)#跨模态注意力机制
        #x为点云特征和图像特征的融合结果   B，256,192
        pc_f=self.cross_norm(pc_f+x)#跨模态注意力机制归一化
        #B，256,192
        keypoints=cent_p.permute(0, 2, 1).contiguous()
        #获取最后一层的点云,并进行维度调整

        ret=self.mlp(pc_f)

        R = ret[:, :, :9].view(b, 256, 3, 3)  #B,256,3,3  3x3旋转矩阵

        T = ret[:, :, 9:]  #B,256,3  3维位移向量

        symmetry_points = torch.matmul(keypoints.transpose(2, 1).contiguous().unsqueeze(2), R).view(b, 256, 3)

        symmetry_points = symmetry_points + T  #应用变换生成对称点云
        #B,256,3  将关键点云与旋转矩阵相乘
        symmetry_points = symmetry_points.transpose(2, 1).contiguous()
            # B,3,256  转置为(B,3,256)的形式
        coarse = torch.cat([symmetry_points, keypoints], dim=-1)
        # B, 3, 512  将对称点云和关键点云拼接在一起，形成粗糙的点云

        #计算风格迁移损失
        if self.fuse_layer_num==0:  #不进行风格迁移
            im_target_gram = gram_matrix(im_f_list[-1])
            pc_target_gram = gram_matrix(pc_f_list[-1])
            style_transfer_loss = torch.mean((im_target_gram - pc_target_gram) ** 2) / (self.embed_dim*256)
        else:
            im_target_gram = gram_matrix(im_f_list[-1])#计算图像数据的格拉姆矩阵
            im_style_gram = gram_matrix(im_f_list[-(self.fuse_layer_num+1)])#计算图像数据的格拉姆矩阵
            pc_target_gram = gram_matrix(pc_f_list[-1])#计算点云数据的格拉姆矩阵
            pc_style_gram = gram_matrix(pc_f_list[-(self.fuse_layer_num+1)])#计算点云数据的格拉姆矩阵
            im_style_loss = torch.mean((im_target_gram - pc_style_gram) ** 2)#计算图像数据的风格损失
            pc_style_loss = torch.mean((pc_target_gram - im_style_gram) ** 2)#计算点云数据的风格损失
            style_loss = (im_style_loss+pc_style_loss) / (self.embed_dim*256)#info，256是格拉姆矩阵的维度
            pc_content_loss=torch.mean((pc_f_list[-1]-pc_f_list[-(self.fuse_layer_num+1)])**2)#str，计算点云数据的内容损失
            style_transfer_loss=pc_content_loss+style_loss#计算风格迁移损失
        return coarse,cent_p,pc_f,im_f,style_transfer_loss#返回点云特征，图像特征，中心点，风格迁移损失
           




# ResNet-based image encoder
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        backbone = resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # remove avgpool & fc
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):  # x: [B, 3, H, W]
        x = self.feature_extractor(x)  # B, 512, H', W'
        x = self.pool(x).squeeze(-1).squeeze(-1)  # B, 512
        x = self.fc(x)  # B, out_dim
        return x


class ImageEncoder2(nn.Module):
    #一个简单的resnet18图像编码器，可以返回全局特征和局部特征
    def __init__(self, out_dim=128, num_local_tokens=512):
        super().__init__()
        backbone = resnet18(pretrained=False)

        # 提取卷积层（去掉 avgpool 和 fc）
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # remove avgpool & fc
        
        # 自适应池化用于全局特征
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 映射全局特征到 out_dim
        self.fc = nn.Linear(512, out_dim)

        # 局部特征映射头（可选）：将 512 降维或保持不变
        self.local_proj = nn.Conv2d(512, out_dim, kernel_size=1)
        # 新增上采样模块，使局部特征变为 B, out_dim, H, W where H*W = 512
        self.local_upsample = nn.Upsample(size=(16, 32), mode='bilinear', align_corners=False)
        self.num_local_tokens = num_local_tokens

    def forward(self, x):  # x: [B, 3, H, W]
        feat_map = self.feature_extractor(x)  # B, 512, H', W'
        global_feat = self.pool(feat_map).squeeze(-1).squeeze(-1)  # B, 512
        global_feat = self.fc(global_feat)  # B, out_dim

        local_feat = self.local_proj(feat_map)            # B, out_dim, 7, 7
        local_feat = self.local_upsample(local_feat)      # B, out_dim, 16, 32
        local_feat = local_feat.flatten(2)                # B, out_dim, 512
        return global_feat, local_feat
        #return: 全局特征,局部特征


class CrossLSTNet(nn.Module):
    #LSTNET: Local Symmetric Transformer Network
    #2.0版本，添加图片作为补充输入信息
    def __init__(self, out_dim=512,img_feat_dim=128):
        from timm.models.vision_transformer import Block
        super(CrossLSTNet, self).__init__()
        self.img_encoder = ImageEncoder(out_dim=img_feat_dim)  #图像编码器
        
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        #使用KNN修改之后的pointnet++，用于点云的采样和特征提取
        
        self.transformer_1 = Transformer(128, dim=64)
        
        self.cross_attn = CrossFormer(dim=128, out_dim=128)
        
        self.expanding = MLP_CONV(in_channel=128, layer_dims=[256, out_dim])     
        
        self.mlp = nn.Sequential(
                nn.Linear(512*2, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 9+3)
        )
        self.blocks= nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=block_head,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(3)])

    def forward(self, point_cloud,image):
        #input: point_cloud (B, 3, N)  部分点云
        b = point_cloud.shape[0]   #batch size
        l0_xyz = point_cloud   #部分点云
        l0_points = point_cloud #部分点云

        #提取图像全局特征
        img_feat = self.img_encoder(image)  # B, img_feat_dim
        img_feat0 = img_feat
        #B,256

        #==========下采样模块开始=========
        #=========首先使用SA（Set Abstraction）也就是knn修改之后的PointNet++对点云进行下采样和特征提取=========
        #=========得到下采样得到的关键点云（keypoints)和

        # get the key points and its features
        #点云特征提取
        keypoints, keyfeatures, _ = self.sa_module_1(l0_xyz, l0_points)  
        # (B, 3, 512), (B, 128, 512)  通过pointnetknn采样512个关键点和对应的特征   
        #keypoints: 关键点云，keyfeatures: 关键点特征

        #==========使用Transformer对关键点特征进行处理,获得其自注意力特征=========
        keyfeatures = self.transformer_1(keyfeatures, keypoints) 
        # B,128,512  使用Transformer对关键点特征进行处理,获得其自注意力特征
        
        # 图像特征扩展到每个关键点
        img_feat = img_feat.unsqueeze(1).repeat(1, keyfeatures.size(-1), 1)  # B, 512, img_feat_dim

        # 转换为 (B, N, C) 供 cross-attn 使用
        keyfeatures = keyfeatures.transpose(1, 2).contiguous()  # B, 512, 256

        #将关键点特征和图像特征进行交叉注意力
        keyfeatures = self.cross_attn(keyfeatures, img_feat)    # B, 512, 128
        keyfeatures = keyfeatures.transpose(1, 2).contiguous()  # B, 128, 512

        feat = self.expanding(keyfeatures)
        #特征维度从128扩展到512    B, 512, 512
        feat = feat.transpose(2, 1).contiguous()
        # B,512,512  转置为(B,512,512)的形式
        gf_feat = feat.max(dim=1, keepdim=True)[0]
        # B,1,512  获取全局特征，取512个特征的最大值
        # ==========下采样模块结束   

        feat = torch.cat([feat, gf_feat.repeat(1, feat.size(2), 1)], dim=-1) 
        # B,512,1024 将全局特征与局部特征拼接在一起

        ret = self.mlp(feat)   #使用一个MLP来预测旋转矩阵和位移向量

        #预测每个点的变换参数（旋转矩阵R和位移向量T）
        R = ret[:, :, :9].view(b, 512, 3, 3)  #B,512,3,3  3x3旋转矩阵
        T = ret[:, :, 9:]  #B,512,3  3维位移向量
        symmetry_points = torch.matmul(keypoints.transpose(2, 1).contiguous().unsqueeze(2), R).view(b, 512, 3)
        # B,512,3  将关键点云与旋转矩阵相乘，得到对称点云
        symmetry_points = symmetry_points + T  #应用变换生成对称点云
        symmetry_points = symmetry_points.transpose(2, 1).contiguous()
        # B,3,512  转置为(B,3,512)的形式
        coarse = torch.cat([symmetry_points, keypoints], dim=-1) 
        # B, 1024, 3  将对称点云和关键点云拼接在一起，形成粗糙的点云
        return coarse, symmetry_points, keyfeatures,img_feat
        #返回： 粗糙点云，对称点云，关键点特征, 图像特征


class CSTNet(nn.Module):
    #Cross Symmetric Transformation Network
    #使用图像特征和点云特征预测结合对称先验对称点云
    def __init__(self, out_dim=512,img_feat_dim=128):
        from timm.models.vision_transformer import Block
        super(CSTNet, self).__init__()
        self.img_encoder = ImageEncoder2(out_dim=img_feat_dim)  #图像编码器
        
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        #使用KNN修改之后的pointnet++，用于点云的采样和特征提取
        
        self.transformer_1 = Transformer(128, dim=64)
        self.transformer_2 = ImageSelfAttentionBlock(128)
        
        self.cross_attn = Fusion(in_channel=128)
        
        self.expanding = MLP_CONV(in_channel=128, layer_dims=[256, out_dim])     
        
        self.mlp = nn.Sequential(
                nn.Linear(512*2, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 9+3)
        )



    def forward(self, point_cloud,image):
        #input: point_cloud (B, 3, N)  部分点云
        #image: 图像输入
        b = point_cloud.shape[0]   #batch size
        l0_xyz = point_cloud   #部分点云
        l0_points = point_cloud #部分点云

        #提取图像全局特征
        img_feat_G, img_feat_L = self.img_encoder(image)  # B, img_feat_dim
        
        #B,128    B,128,512

        #==========下采样模块开始=========
        #=========首先使用SA（Set Abstraction）也就是knn修改之后的PointNet++对点云进行下采样和特征提取=========
        #=========得到下采样得到的关键点云（keypoints)和关键点云的特征（Nk x C)=========

        # get the key points and its features
        #点云特征提取
        keypoints, keyfeatures, _ = self.sa_module_1(l0_xyz, l0_points)  
        # (B, 3, 512), (B, 128, 512)  通过pointnetknn采样512个关键点和对应的局部特征   
        #keypoints: 关键点云，keyfeatures: 关键点特征


        #==========使用Transformer对关键点特征进行处理,获得其自注意力特征=========
        keyfeatures = self.transformer_1(keyfeatures, keypoints) 
        # B,128,512  使用Transformer对关键点特征进行处理,获得其自注意力特征
        
        # 图像特征扩展到每个关键点
        #img_feat = img_feat_G.unsqueeze(1).repeat(1, keyfeatures.size(-1), 1)  # B, 512, img_feat_dim
        img_feat = img_feat_L.transpose(1, 2).contiguous()  # B, 512, 128
        img_feat = self.transformer_2(img_feat)  # B, 512, 128
        #使用图像自注意力模块对图像特征进行处理

        # B, 512, 128  将图像全局特征扩展到每个关键点
        # 转换为 (B, N, C) 供 cross-attn 使用
        keyfeatures = keyfeatures.transpose(1, 2).contiguous()  # B, 512, 128

        #将关键点特征和图像特征进行交叉注意力融合
        keyfeatures = self.cross_attn(keyfeatures, img_feat)    # B, 512, 128
        #嵌入维度为128的关键点特征和图像特征进行交叉注意力融合


        #计算style loss
        im_target_gram = gram_matrix(img_feat)
        pc_target_gram = gram_matrix(keyfeatures)
        style_transfer_loss = torch.mean((im_target_gram - pc_target_gram) ** 2) / (128*512)
        
        #作为keypoints最终的特征
        keyfeatures = keyfeatures.transpose(1, 2).contiguous()  # B, 128, 512
        img_feat= img_feat.transpose(1, 2).contiguous()  # B, 128, 512


        feat = self.expanding(keyfeatures)
        #特征维度从128扩展到512    B, 512, 512
        feat = feat.transpose(2, 1).contiguous()
        # B,512,512  转置为(B,512,512)的形式
        gf_feat = feat.max(dim=1, keepdim=True)[0]
        # B,1,512  获取全局特征，取512个特征的最大值
        
        # ==========下采样模块结束
        # downsampling 输出： keypoints:关键点云Pk,   keyfeatures关键点云局部特征（128xC)，全局特征gf_feat（1xC）


        feat = torch.cat([feat, gf_feat.repeat(1, feat.size(2), 1)], dim=-1) 
        # B,512,1024 将全局特征与局部特征拼接在一起
      
                
        ret = self.mlp(feat)   #使用一个MLP来预测旋转矩阵和位移向量



        #预测每个点的变换参数（旋转矩阵R和位移向量T）
        R = ret[:, :, :9].view(b, 512, 3, 3)  #B,512,3,3  3x3旋转仿射矩阵
        T = ret[:, :, 9:]  #B,512,3  3维位移向量
        symmetry_points = torch.matmul(keypoints.transpose(2, 1).contiguous().unsqueeze(2), R).view(b, 512, 3)
        # B,512,3  将关键点云与旋转矩阵相乘，得到对称点云
        symmetry_points = symmetry_points + T  #应用变换生成对称点云
        symmetry_points = symmetry_points.transpose(2, 1).contiguous()
        # B,3,512  转置为(B,3,512)的形式
        coarse = torch.cat([symmetry_points, keypoints], dim=-1) 
        # B, 1024, 3  将对称点云和关键点云拼接在一起，形成粗糙的点云
        return coarse, symmetry_points, keyfeatures,img_feat,style_transfer_loss
        #返回： 粗糙点云，对称点云，关键点特征, 图像特征



class Fusion(nn.Module):
    def __init__(self, in_channel=512):
        super(Fusion, self).__init__()
        
        self.corssformer_1 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)
        self.corssformer_2 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)
    
    def forward(self, feat_x, feat_y):
        #feat_x：Q特征，feat_y：K特征

        #首先进行交叉注意力融合Q特征和K特征
        #然后对融合后的特征进行自注意力机制
        # cross attention
        feat = self.corssformer_1(feat_x, feat_y)
        
        # self attention
        feat = self.corssformer_2(feat, feat)
        return feat
    
class SGFormer(nn.Module):
    #点云上采样和细化模块
    #接收粗糙点云（对称拼接之后的key点云、对称点云特征和关键点点云特征作为输入
    def __init__(self, gf_dim=512, up_factor=2):
        super(SGFormer, self).__init__()
        self.up_factor = up_factor

        #卷积层用于维度变换
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_gf = MLP_CONV(in_channel=gf_dim, layer_dims=[256, 128])
        self.mlp_2 = MLP_CONV(in_channel=256, layer_dims=[256, 128])
        
        #transformer用于几何特征增强
        self.transformer = Transformer(in_channel=128, dim=64)
        
        self.expand_dim_1 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_2 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_3 = MLP_CONV(in_channel=128, layer_dims=[128, 256])

        self.fusion_1 = Fusion(in_channel=256)
        self.fusion_2 = Fusion(in_channel=256)
        
        self.mlp_fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512)
        )
        self.fusion_3 = Fusion(in_channel=512)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3 * self.up_factor)
        )

    def forward(self, coarse, symmetry_feat, partial_feat):
        # coarse: B, 3, N 粗糙点云  
        # symmetry_feat:  对称点云特征
        # partial_feat:  key点云特征
        b, _, n = coarse.shape
        feat = self.mlp_1(coarse)  #从粗糙点云中提取特征 B，128,1024
        feat_max = feat.max(dim=-1, keepdim=True)[0]#最大池化获取全局特征 B，128，1
        feat= torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
        #合并局部特征和全局特征 B，256,1024
        feat = self.mlp_2(feat)   #输出128维特征 B，128,1024

        feat = self.transformer(feat, coarse)
        # B, 128, N  使用Transformer对特征进行处理,增强特征表示

        #将当前点云特征，对称点云特征和key点云特征扩展到相同维度
        feat = self.expand_dim_1(feat)#B, 256, 1024
        partial_feat = self.expand_dim_2(partial_feat) #B,256,512
        symmetry_feat = self.expand_dim_3(symmetry_feat)#B,256,512

        #转置特征以适应后续操作
        feat = feat.transpose(2, 1).contiguous()#B,1024,256
        partial_feat = partial_feat.transpose(2, 1).contiguous()#b,512,256
        symmetry_feat = symmetry_feat.transpose(2, 1).contiguous()#b,512,256


        # partial part awareness
        feat_p = self.fusion_1(feat, partial_feat)
        #部分感知：粗糙点云特征和key点云特征融合
        #首先是用partial_feat作为key特征，粗糙点云特征作为query特征进行交叉注意力
        #然后对融合后的特征进行自注意力机制
        # (key点云特征和粗糙点云特征的交叉注意力)


        # symmetric part awareness
        feat_s = self.fusion_2(feat, symmetry_feat)
        #对称感知：粗糙点云特征和对称点云特征融合 
        #首先是用symmetry_feat作为key特征，粗糙点云特征作为query特征进行交叉注意力
        #然后对融合后的特征进行自注意力机制
        #粗糙点云特征和对称点云特征的交叉注意力

        # fusion feature
        feat = torch.cat([feat_p, feat_s], dim=-1)
        #将部分感知特征和对称感知特征拼接在一起，得到融合特征   2Nk x C
        feat = self.mlp_fusion(feat)

        # self attention for upsampling
        feat = self.fusion_3(feat, feat)
        #自注意力机制对融合特征进行处理，增强特征表示
        #这里是实际上使用了两次自注意力机制来处理融合特征

        offset = self.fc(feat).view(b, -1, 3) # B, N * up_ratio, 3
        #解码器，预测每一个点的偏移量

        pcd_up = coarse.transpose(2, 1).contiguous().unsqueeze(dim=2).repeat(1, 1, self.up_factor, 1).view(b, -1, 3) + offset
        #得到上采样之后的高分辨率点云
        return pcd_up


class SGFormer2(nn.Module):
    #点云上采样和细化模块
    #接收粗糙点云（对称拼接之后的key点云、对称点云特征和关键点点云特征作为输入
    #2.0版本，添加了图像特征作为输入,使用图片的交叉注意力来指导上采样
    def __init__(self, gf_dim=512,img_feat_dim=128, up_factor=2,attn_dim=256):
        super(SGFormer2, self).__init__()
        self.up_factor = up_factor

        #卷积层用于维度变换
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_gf = MLP_CONV(in_channel=gf_dim, layer_dims=[256, 128])
        self.mlp_2 = MLP_CONV(in_channel=256, layer_dims=[256, 128])
        
        #transformer用于几何特征增强
        self.transformer = Transformer(in_channel=128, dim=64)
        
        self.expand_dim_1 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_2 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_3 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_4 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        
        # 图像 - 点云 Cross Attention 模块
        self.cross_attn_partial = CrossFormer(dim=img_feat_dim, out_dim=attn_dim)
        self.cross_attn_symmetry = CrossFormer(dim=img_feat_dim, out_dim=attn_dim)


        self.fusion_1 = Fusion(in_channel=256)
        self.fusion_2 = Fusion(in_channel=256)
        self.fusion_4 = Fusion(in_channel=256)
        self.fusion_5 = Fusion(in_channel=256)
        
        self.mlp_fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512)
        )
        self.fusion_3 = Fusion(in_channel=512)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3 * self.up_factor)
        )
        self.expand_img_feat = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2)
)

    def forward(self, coarse, symmetry_feat, partial_feat, img_feat):
        # coarse: B, 3, N 粗糙点云  
        # symmetry_feat:  对称点云特征   B，128,512
        # partial_feat:  key点云特征    B，128，512
        # img_feat: 经过自注意力机制加强的图像特征   B，128，512
        b, _, n = coarse.shape
        feat = self.mlp_1(coarse)  #从粗糙点云中提取特征 B,128,1024
        feat_max = feat.max(dim=-1, keepdim=True)[0]#最大池化获取粗糙点云全局特征  B,128,1
        feat= torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
        #合并局部特征和全局特征，作为粗糙点云的特征  B,256,1024
        feat = self.mlp_2(feat)   #维度转换输出128维特征  B,128,1024

        feat = self.transformer(feat, coarse)
        # B, 128, N  使用Transformer对粗糙点云特征进行处理,增强特征表示

        #将当前点云特征，对称点云特征和key点云特征扩展到相同维度
        feat = self.expand_dim_1(feat) #B, 256, 1024
        partial_feat = self.expand_dim_2(partial_feat) #B,256,512   关键点点云特征
        symmetry_feat = self.expand_dim_3(symmetry_feat)#B,256,512  #对称点云（预测点云）特征

        #转置特征以适应后续操作
        feat = feat.transpose(2, 1).contiguous()#B,1024,256
        partial_feat = partial_feat.transpose(2, 1).contiguous()#b,512,256
        symmetry_feat = symmetry_feat.transpose(2, 1).contiguous()#b,512,256

        # ============ Cross Attention from Image ============
        img_feat_exp = self.expand_dim_4(img_feat)  
        img_feat_exp = img_feat_exp.transpose(2, 1).contiguous()  # B, 512, 256
        
        feat_partial_attn = self.fusion_1(partial_feat,img_feat_exp)
        #图像特征和关键点点云特征的交叉注意力

        feat_symmetry_attn = self.fusion_4(symmetry_feat,img_feat_exp)  
        #图像特征和对称点云特征的交叉注意力

        #feat=self.fusion_1(feat, img_feat_exp)      

        # 融合注意力增强特征
        feat_p = self.fusion_5(feat, feat_partial_attn)
        #部分感知：粗糙点云特征和key点云特征融合
        # (key点云特征和粗糙点云特征的交叉注意力)


        # symmetric part awareness
        feat_s = self.fusion_2(feat, feat_symmetry_attn)
        #对称感知：粗糙点云特征和对称点云特征融合 
        #粗糙点云特征和对称点云特征的交叉注意力

        # fusion feature
        feat = torch.cat([feat_p, feat_s], dim=-1)
        #将部分感知特征和对称感知特征拼接在一起，得到融合特征
        feat = self.mlp_fusion(feat)

        # self attention for upsampling
        feat = self.fusion_3(feat, feat)
        #自注意力机制对融合特征进行处理，增强特征表示
        offset = self.fc(feat).view(b, -1, 3) # B, N * up_ratio, 3
        #预测每一个点的偏移量
        pcd_up = coarse.transpose(2, 1).contiguous().unsqueeze(dim=2).repeat(1, 1, self.up_factor, 1).view(b, -1, 3) + offset
        #得到上采样之后的高分辨率点云
        return pcd_up





class local_encoder(nn.Module):
    def __init__(self,out_channel=128):
        super(local_encoder, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2, layer_dims=[256, out_channel])
        self.transformer = Transformer(out_channel, dim=64)

    def forward(self,input):
        feat = self.mlp_1(input)
        feat = torch.cat([feat,torch.max(feat, 2, keepdim=True)[0].repeat((1, 1, feat.size(2)))], 1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat,input)

        return feat

class SymmCompletion(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.up_factors = [int(i) for i in config.SymmCompletion.up_factors.split(',')]
        self.lstnet = LSTNet(out_dim=512)   
        #对称transformer网络
        #输入： 部分点云   输出：粗糙点云，对称点云，关键点特征
        self.local_encoder = local_encoder(out_channel=128)
        #局部编码器，用于提取对称点云的局部特征，并通过transformer进行增强
        #输入：对称点云  输出：对称点云特征
        self.sgformer_1 = SGFormer(gf_dim=512, up_factor=self.up_factors[0])
        #上采样细化模块
        #输入：粗糙点云，对称点云特征，部分点云特征  输出：上采样后的点云
        self.sgformer_2 = SGFormer(gf_dim=512, up_factor=self.up_factors[1])
        self.include_input = config.SymmCompletion.include_input
        #是否将输入点云包含在输出中
        self.loss_func = ChamferDistanceL1()
        
    def get_loss(self, rets, gt):
        #损失函数计算框架
        loss_list = []
        loss_total = 0
        for pcd in rets:
            #对于粗糙点云，一次上采样点云，二次上采样点云，都计算损失
            loss = self.loss_func(pcd, gt)
            loss_list.append(loss)
            loss_total += loss
        return loss_total, loss_list[0], loss_list[-1]

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        coarse, symmetry_points, keyfeatures = self.lstnet(point_cloud.transpose(2,1).contiguous())
        #LSTNet处理输入部分点云数据，得到粗糙点云（包括原始关键点和对称生成的点）
        #coarse: B,3,1024, symmetry_points: B,3,512, keyfeatures: B,128,512
        feat_symmetry = self.local_encoder(symmetry_points) 
        # B,128,512  提取对称点云的几何特征
        feat_partial = keyfeatures # B,128,512  关键点特征作为部分点云特征
        fine1 = self.sgformer_1(coarse, feat_symmetry, feat_partial)
        #第一次上采样细化   B,2048,3
        fine2 = self.sgformer_2(fine1.transpose(2,1).contiguous(), feat_symmetry, feat_partial)
        #第二次上采样细化   B,6144,3

        if self.include_input: 
            #如果包含输入点云，则将输入点云与细化后的点云拼接
            #获取更为完整的点云表示
            fine2 = torch.cat([fine2, point_cloud],dim=1).contiguous()
        #B,8192,3
        rets = [coarse.transpose(2,1).contiguous(), fine1, fine2]
        #results包括粗糙点云、第一次上采样细化后的点云和第二次上采样细化后的点云
        self.pred_dense_point = rets[-1]
        #保存第二次上采样细化后的点云作为预测结果
        complete=farthest_point_sample(rets[-1], 2048)

        return rets,complete


class SymmCompletion2(nn.Module):
    #多模态对称点云补全网络
    def __init__(self, config, **kwargs):
        super().__init__()
        self.up_factors = [int(i) for i in config.SymmCompletion.up_factors.split(',')]
        self.lstnet = CrossLSTNet(out_dim=512,img_feat_dim=128)
        #对称transformer网络
        #输入： 部分点云   输出：粗糙点云，对称点云，关键点特征
        self.local_encoder = local_encoder(out_channel=128)
        #局部编码器，用于提取对称点云的局部特征，并通过transformer进行增强
        #输入：对称点云  输出：对称点云特征
        self.sgformer_1 = SGFormer2(gf_dim=512, up_factor=4)
        #上采样细化模块   up_factor: 2
        #输入：粗糙点云，对称点云特征，部分点云特征  输出：上采样后的点云
        #self.sgformer_2 = SGFormer2(gf_dim=512, up_factor=self.up_factors[1])
        #up_factor: 3
        self.include_input = config.SymmCompletion.include_input
        #是否将输入点云包含在输出中
        self.loss_func = ChamferDistanceL1()
        
    def get_loss(self, rets, gt, view):
        #损失函数计算框架
        loss_list = []
        loss_total = 0
        for pcd in rets:
            #对于粗糙点云，一次上采样点云，二次上采样点云，都计算损失
            loss = self.loss_func(pcd, gt)
            loss_list.append(loss)
            loss_total += loss

        return loss_total

    def forward(self, point_cloud, view):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        coarse, symmetry_points, keyfeatures,img_feat = self.lstnet(point_cloud.transpose(2,1).contiguous(),view)
        #LSTNet处理输入部分点云数据，得到粗糙点云（包括原始关键点和对称生成的点）
        #coarse: B,3,1024, symmetry_points: B,3,512, keyfeatures: B,128,512
        #粗糙点云为原始key点和对称生成的点拼接而成的，即
        #symmetry_points可以理解为网络预测的缺失部分的点云，是预测的key点的对称点云
        #coarse=keypoints + symmetry_points
        # symmetry_points为预测的对称点云，接下来要对这部分的对称点云进行细化
        #keyfeatures为网络下采样的关键点提取的特征
        feat_symmetry = self.local_encoder(symmetry_points) 
        # B,128,512  提取对称点云的几何特征
        feat_partial = keyfeatures # B,128,512  关键点特征作为部分点云特征（因为其本身就是输入的部分点云的一部分）
        
        #对coarse点云进行细化，使用key点云特征和对称点云特征的交叉注意力机制
        fine1 = self.sgformer_1(coarse, feat_symmetry, feat_partial, img_feat)
        #第一次上采样细化   B,2048,3
        #fine2 = self.sgformer_2(fine1.transpose(2,1).contiguous(), feat_symmetry, feat_partial)
        #第二次上采样细化   B,6144,3


            #如果包含输入点云（输入的部分点云），则将输入点云与细化后的点云拼接
            #获取更为完整的点云表示
        fine2 = torch.cat([fine1, point_cloud],dim=1).contiguous()
        #B,8192,3
        rets = [coarse.transpose(2,1).contiguous(), fine1, fine2]
        #results包括粗糙点云、第一次上采样细化后的点云和第二次上采样细化后的点云
        self.pred_dense_point = rets[-1]
        #保存第二次上采样细化后的点云作为预测结果
        complete=farthest_point_sample(rets[-1], 2048)

        return rets,complete


class SymmCompletion3(nn.Module):
    #多模态对称点云补全网络
    def __init__(self,config, **kwargs):
        super().__init__()
        self.up_factors = [int(i) for i in config.SymmCompletion.up_factors.split(',')]
        self.lstnet = CSTNet(out_dim=512)
        #对称transformer网络
        #输入： 部分点云   输出：粗糙点云，对称点云，关键点特征
        self.local_encoder = local_encoder(out_channel=128)
        #局部编码器，用于提取对称点云的局部特征，并通过transformer进行增强
        #输入：对称点云  输出：对称点云特征
        self.sgformer_1 = SGFormer2(gf_dim=512, up_factor=4)
        #上采样细化模块   up_factor: 2
        #输入：粗糙点云，对称点云特征，部分点云特征  输出：上采样后的点云
        #self.sgformer_2 = SGFormer2(gf_dim=512, up_factor=self.up_factors[1])
        #up_factor: 3
        self.include_input = config.SymmCompletion.include_input
        #是否将输入点云包含在输出中
        self.loss_func = ChamferDistanceL1()
        
    def get_loss(self, rets, gt, view):
        #损失函数计算框架
        loss_list = []
        loss_total = 0
        for pcd in rets:
            #对于粗糙点云，一次上采样点云，二次上采样点云，都计算损失
            loss = self.loss_func(pcd, gt)
            loss_list.append(loss)
            loss_total += loss

        return loss_total

    def forward(self, point_cloud, view):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        
        coarse, symmetry_points, keyfeatures,img_feat,style_loss = self.lstnet(point_cloud.transpose(2,1).contiguous(),view)    
        #coarse: B,3,1024 ,symmetry_points: B,3,512, keyfeatures: B,128,512, img_feat: B,128,512




        feat_symmetry = self.local_encoder(symmetry_points) 
        # B,128,512  提取对称点云的几何特征
        feat_partial = keyfeatures # B,128,512  关键点特征作为部分点云特征（因为其本身就是输入的部分点云的一部分）
        
        #对coarse点云进行细化，使用key点云特征和对称点云特征的交叉注意力机制
        fine1 = self.sgformer_1(coarse, feat_symmetry, feat_partial, img_feat)
        #第一次上采样细化   B,2048,3
        #fine2 = self.sgformer_2(fine1.transpose(2,1).contiguous(), feat_symmetry, feat_partial)

        rets = [coarse.transpose(2,1).contiguous(), fine1]
        #results包括粗糙点云、第一次上采样细 化后的点云和第二次上采样细化后的点云
        self.pred_dense_point = rets[-1]
        #保存第二次上采样细化后的点云作为预测结果
        complete=farthest_point_sample(rets[-1], 2048)
        rets.append(complete)

        return rets,complete,style_loss


if __name__ == '__main__':
    model = SymmCompletion3(config=cfg)
    model=model.cuda()
    pc = torch.rand([4,2048,3]).cuda()#部分点云数据,[4,3,2048]
    img = torch.rand([4, 3, 224, 224]).cuda()#图像数据,[4,3,224,224]

    rets=model(pc, img)

