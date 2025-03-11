''''
在此处复刻P2C点云自监督学习的训练代码
'''''

import torch
import os
import time
#from model_pre import Network
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
from torchvision import transforms
from datasets import data_utils as d_utils

from P2C.P2C import Encoder, Decoder,P2C

loss_cd=L2_ChamferLoss()#训练损失函数
loss_cd_eval=L1_ChamferEval()#评估损失函数

from utils.misc import *


def train_net(cfg):

    # 初始化WandB
    wandb.init(project="P2C", config=cfg, name=f"{cfg.TRAIN.CATE}_{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}")

    #设置cuDNN的benchmark模式为True，以加速卷积操作
    torch.backends.cudnn.benchmark = True

    ViPC_train = ViPCDataLoader(r'./train_list.txt',
                                 data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='train',
                                 view_align=False, category=cfg.TRAIN.CATE)        
    train_loader = DataLoader(ViPC_train,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=cfg.CONST.NUM_WORKERS,
                                  shuffle=True,
                                  drop_last=True,
                                  prefetch_factor=cfg.CONST.DATA_perfetch)  

    ViPC_test = ViPCDataLoader3(r'./test_list.txt',
                                data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='test',
                                view_align=False, category=cfg.TRAIN.CATE)    

    test_loader = DataLoader(ViPC_test,
                                 batch_size=cfg.TRAIN.BATCH_SIZE,
                                 num_workers=cfg.CONST.NUM_WORKERS,
                                 shuffle=True,
                                 drop_last=True,
                                 prefetch_factor=cfg.CONST.DATA_perfetch)
    
    model=P2C(cfg).cuda()


    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.P2C.lr, weight_decay=cfg.P2C.weight_decay)
    

    scheduler_steplr=MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)

    lr_scheduler=GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)

    init_epoch = 0  #初始化训练epoch为0
    best_metrics = float('inf')#初始化最佳指标为无穷大
    steps = 0#初始化步数为0
    best_epoch = 0

    for epoch in range(init_epoch+1, cfg.P2C.max_epoch + 1):
        model.train()
        total_loss = AverageMeter2()
        ShapeReconLoss = AverageMeter2()
        ShapeCompletionLoss = AverageMeter2()
        LatentLoss = AverageMeter2()
        NCCLoss = AverageMeter2()










if __name__ == "__main__":
    import config_vipc
    cfg = config_vipc.cfg
    train_net(cfg)
