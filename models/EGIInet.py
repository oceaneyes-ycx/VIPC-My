import numpy as np
#from deformable_attention import DeformableAttention1D
from torch import nn

from models.dec_net import Decoder_Network
import torch
from torch.nn import Module
from models.encoders import  transfer_loss_shared_encoder
from config_vipc import cfg

class EGIInet(Module):
    #EGIInet网络
    def __init__(self, 
                 embed_dim=cfg.NETWORK.EGIInet.embed_dim,
                 depth=cfg.NETWORK.EGIInet.depth,
                 img_patch_size=cfg.NETWORK.EGIInet.img_patch_size,
                 pc_sample_rate=cfg.NETWORK.EGIInet.pc_sample_rate,
                 pc_sample_scale=cfg.NETWORK.EGIInet.pc_sample_scale,
                 fuse_layer_num=cfg.NETWORK.EGIInet.fuse_layer_num,
                 ):
        #初始化函数
        super().__init__()#继承
        self.encoder=transfer_loss_shared_encoder(embed_dim=embed_dim,
                                               img_patch_size=img_patch_size,
                                               sample_ratio=pc_sample_rate,
                                               scale=pc_sample_scale,
                                               block_head=cfg.NETWORK.shared_encoder.block_head,
                                               depth=depth,
                                               pc_h_hidden_dim=cfg.NETWORK.shared_encoder.pc_h_hidden_dim,
                                               fuse_layer_num=fuse_layer_num,
                                               )
        self.decoder=Decoder_Network(K1=embed_dim,K2=embed_dim,N=embed_dim)

    def forward(self,pc,img):
        feature, _, _, style_loss = self.encoder(pc=pc, im=img)#编码
        final = self.decoder(feature, pc)#解码
        return final,style_loss#返回结果和风格损失

if __name__ == '__main__':
    import time
    model = EGIInet().cuda()
    pc = torch.rand([4, 2048, 3]).cuda()#部分点云数据,[4,2048,3]
    img = torch.rand([4, 3, 224, 224]).cuda()#图像数据,[4,3,224,224]
    s=time.time()
    fine = model(pc, img)#前向传播
    e=time.time()
    print(e-s)
    #print(fine.shape)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #计算参数数量
    parameters = sum([np.prod(p.size()) for p in model_parameters])
    print(f"n parameters:{parameters}")
