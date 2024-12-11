#V1版本训练框架
'''
Cross-Model 对比学习用于VIPC任务,使用预训练好的点云和图像编码器（参数冻结）
代码基于XMFNet,设定几乎完全一致
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
from utils.ViPCdataloader import ViPCDataLoader
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from utils.schedular import GradualWarmupScheduler
from torch.optim.lr_scheduler import *


def train_net(cfg):
    #接收一个配置文件作为参数

    #设置cuDNN的benchmark模式为True，以加速卷积操作
    torch.backends.cudnn.benchmark = True

    #创建ViPC训练数据加载器，使用自定义的ViPCDataLoader类
    ViPC_train = ViPCDataLoader(r'train_list.txt',
                                 data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='train',
                                 view_align=False, category=cfg.TRAIN.CATE)
    
    #创建训练数据加载器，使用pytorch自带的DataLoader类
    train_data_loader = DataLoader(ViPC_train,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=cfg.CONST.NUM_WORKERS,
                                  shuffle=True,
                                  drop_last=True,
                                  prefetch_factor=cfg.CONST.DATA_perfetch)
    #创建ViPC测试数据加载器
    ViPC_test = ViPCDataLoader(r'test_list.txt',
                                data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='test',
                                view_align=False, category=cfg.TRAIN.CATE)
    #创建测试数据加载器
    val_data_loader = DataLoader(ViPC_test,
                                 batch_size=cfg.TRAIN.BATCH_SIZE,
                                 num_workers=cfg.CONST.NUM_WORKERS,
                                 shuffle=True,
                                 drop_last=True,
                                 prefetch_factor=cfg.CONST.DATA_perfetch)
    
    #创建模型，加载预训练的编码器
    model = Network(pretrained_model_path=r'weights/best_model.pth')
    model.cuda()
    # 如果GPU可用
    if torch.cuda.is_available():
    # 将模型包装成DataParallel模型，以便在多个GPU上进行并行计算
        model = torch.nn.DataParallel(model).cuda()



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
    BestEpoch = 0




    #开始训练
    for epoch in range(init_epoch+1, cfg.TRAIN.N_EPOCHS+1):
        
        #遍历每个epoch
        model.train()
