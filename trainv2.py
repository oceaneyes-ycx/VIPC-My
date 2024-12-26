'''
V2版本, 修改为端到端的方式进行训练。
'''

import torch
import os
import time
from model import Network
from decoder.utils.utils import *
import wandb
import argparse
import logging
from torch.utils.data import DataLoader
from utils.ViPCdataloader import ViPCDataLoader, ViPCDataLoader3
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from utils.schedular import GradualWarmupScheduler
from torch.optim.lr_scheduler import *
from tqdm import tqdm
from datetime import datetime
from testv1 import test_net

import open3d as o3d
import os
import numpy as np
from PIL import Image

from end2endmodel import End2endNetwork
from lightly.loss.ntx_ent_loss import NTXentLoss

from utils.average_meter import AverageMeter, AverageMeter2


loss_cd=L1_ChamferLoss()#训练损失函数
loss_cd_eval=L2_ChamferEval()#评估损失函数

def train_net(cfg):

    # 初始化WandB
    wandb.init(project="ViPC-Training-V2", config=cfg, name=f"{cfg.TRAIN.CATE}_{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}")

    #设置cuDNN的benchmark模式为True，以加速卷积操作
    torch.backends.cudnn.benchmark = True

    #创建ViPC训练数据加载器，使用自定义的ViPCDataLoader类
    ViPC_train = ViPCDataLoader3(r'./train_list.txt',
                                 data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='train',
                                 view_align=False, category=cfg.TRAIN.CATE)        

    #创建训练数据加载器，使用pytorch自带的DataLoader类
    train_loader = DataLoader(ViPC_train,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=cfg.CONST.NUM_WORKERS,
                                  shuffle=True,
                                  drop_last=True,
                                  prefetch_factor=cfg.CONST.DATA_perfetch)

    #创建ViPC测试数据加载器
    ViPC_test = ViPCDataLoader3(r'./test_list.txt',
                                data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='test',
                                view_align=False, category=cfg.TRAIN.CATE)
    #创建测试数据加载器
    test_loader = DataLoader(ViPC_test,
                                 batch_size=cfg.TRAIN.BATCH_SIZE,
                                 num_workers=cfg.CONST.NUM_WORKERS,
                                 shuffle=True,
                                 drop_last=True,
                                 prefetch_factor=cfg.CONST.DATA_perfetch)
    

    #创建模型
    model = End2endNetwork().cuda()
    point_model=model.point_model
    img_model=model.img_model

    # Create the optimizers
    # 定义优化器，使用Adam优化算法
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),  # 过滤掉不需要梯度的参数
                                 lr=cfg.TRAIN.LEARNING_RATE,  # 学习率
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,  # 权重衰减
                                 betas=cfg.TRAIN.BETAS)  # Adam优化算法的参数

    # lr scheduler
    scheduler_steplr = MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    #多步学习率调度器
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)
    #渐进式学习率调度器

    init_epoch = 0  #初始化训练epoch为0
    best_metrics = float('inf')#初始化最佳指标为无穷大
    steps = 0#初始化步数为0
    best_epoch = 0

    # 对比学习损失函数
    criterion = NTXentLoss(temperature=0.1).cuda()    

    #开始训练
    for epoch in range(init_epoch+1, cfg.TRAIN.N_EPOCHS + 1):
        #遍历每个epoch
        model.train()

        total_cd_pc=0 #L1 CD loss
        contra_loss = AverageMeter2() #对比学习总损失
        train_intra_loss = AverageMeter2()
        train_cross_loss = AverageMeter2()
        total_loss=0 #总损失

        n_batches = len(train_loader)#获取训练数据加载器的长度
        print('epoch:',epoch,'optimizer: ',optimizer.param_groups[0]['lr'])
        with tqdm(train_loader) as t:
            #视图，GT 点云, 部分点云, 对比学习的1号点云，对比学习的2号点云
            for batch_idex, (view,gt_pc,part_pc,pc1,pc2) in enumerate(t):
                partial=part_pc.cuda()
                gt=gt_pc.cuda()
                png=view.cuda()
                pc_1=pc1.cuda()
                pc_2=pc2.cuda()

                #对部分点云和GT点云进行FPS采样
                partial=farthest_point_sample(partial,2048)
                gt=farthest_point_sample(gt,2048)

                #对比学习的点云进行FPS采样
                pc_1=farthest_point_sample(pc_1,2048)
                pc_2=farthest_point_sample(pc_2,2048)

                complete=model(partial,png)   #输出补全的点云

                #计算CD损失
                cd=chamfer_sqrt(complete,gt) #计算L1 CD loss

                #计算对比学习损失
                batch_size = png.size(0)
                data = torch.cat((pc_1, pc_2))
                _ , point_feats = point_model(data, data.transpose(1, 2).contiguous())

                point_t1_feats = point_feats[:batch_size, :]
                point_t2_feats = point_feats[batch_size:, :]

                loss_intra = criterion(point_t1_feats, point_t2_feats) #模态内损失

                _ ,img_rgb_feat= img_model(png)

                point_feats_combined = torch.stack([point_t1_feats, point_t2_feats]).mean(dim=0)

                loss_cross = criterion(img_rgb_feat, point_feats_combined) #模态间损失

                contra_loss = loss_intra + loss_cross #对比学习总损失

                loss_total = cd + 0.15*contra_loss #总损失


                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()


                cd_pc_item=cd.item()*1e3    #CD损失扩大化，用于更好显示
                total_cd_pc+=cd_pc_item
                contra_item=contra_loss.item()*1e1

                






                





