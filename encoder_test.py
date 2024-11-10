"""多模态解码器"""

import numpy as np
from torch import einsum
from models.layers.subsample import furthest_point_sample, random_sample
from models.layers.group import KNNGroup, QueryAndGroup, get_aggregation_feautres
from models.layers.conv import  create_convblock2d,create_linearblock
from models.layers.local_aggregation import CHANNEL_MAP
from models.layers.norm import create_norm
import math
import torch
import torch.nn as nn
import collections.abc
from itertools import repeat
# From PyTorch internals
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
            cur_p, cur_f = out_p[-1], out_f[-1]#获取当前点和特征
            idx = self.sample_fn(cur_p, int(N //self.scale)).long()#得到采样索引
            N = N // self.scale#更新邻居数

            #根据采样索引获取中心点和特征
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

class transfer_loss_shared_encoder(nn.Module):
    #多模态编码器
    def __init__(self,
                 embed_dim=192, 
                 block_head=12, 
                 depth=3,
                 img_patch_size=14,
                 sample_ratio=0.125, 
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

    def forward(self,pc,im):
        #前向传播函数，接收参数：pc，点云数据（张量）；im，图像数据（张量）
        B=pc.size(0)
        p_list,x_list=self.pc_to_token(pc,pc.transpose(1,2).contiguous())
        #点云数据token化，返回点云数据和特征

        cent_p,pc_f=p_list[-1],self.proj(x_list[-1].transpose(1, 2))
        #计算点云数据的中心点和特征，将特征映射到与图像数据相同的嵌入维度
        #返回点云数据的中心点和特征
        im_f,_,_=self.im_to_token(im)
        #图像数据token化

        pc_pos_emd = self.pc_pos_embed(cent_p)
        #计算点云数据的位置嵌入


        #存储每个深度层的点云数据和图像数据
        pc_f_list=[]
        im_f_list=[]

        for i in range(self.depth):
            #循环处理每个深度层
            pc_f = self.blocks[i](pc_f + pc_pos_emd)
            im_f = self.blocks[i](im_f)
            pc_f_list.append(pc_f)
            im_f_list.append(im_f)
        pc_f=self.pc_norm(pc_f)#点云数据归一化
        im_f=self.im_norm(im_f)#图像数据归一化
        x, _ = self.cross_layer(pc_f, im_f, pc_f)#跨模态注意力机制
        pc_f=self.cross_norm(pc_f+x)#跨模态注意力机制归一化

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
        return pc_f,im_f,cent_p,style_transfer_loss#返回点云特征，图像特征，中心点，风格迁移损失
           
if __name__ == '__main__':
    import time
    pc = torch.rand([4, 2048, 3]).cuda()#4个点云，每个点云有2048个点，每个点有3个特征,4*2048*3
    img = torch.rand([4, 3, 224, 224]).cuda()#4个图像，每个图像有3个通道，224*224,4*3*224*224
    pointmodel=pc_tokenizer().cuda()
    _,pc_f=pointmodel(pc,pc.transpose(1,2).contiguous())
    print(pc_f)
    #model = transfer_loss_shared_encoder().cuda()
    #s = time.time()
    #x1,x2, p,s_l= model(pc,img)
    #x1是点云数据，x2是图像数据，p是点云数据的中心点，s_l是风格迁移损失
    #e = time.time()
    #print(e-s)#打印时间
    #print(x1.shape)#打印点云特征的形状
    #print(x2.shape)#打印图像特征的形状
    #print(s_l)#打印风格迁移损失
    #model_parameters = filter(lambda p: p.requires_grad, model.parameters())#过滤出需要梯度的参数
    #parameters = sum([np.prod(p.size()) for p in model_parameters])#计算参数数量
    #print(f"n parameters:{parameters}")#打印参数数量

