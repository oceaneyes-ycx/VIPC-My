''''
使用P2C模型训练VIPC数据集
自监督学习方法
优化器和学习率调度器等按照EGIINET的设置进行
暂时没有用到图像模态的功能
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
    wandb.init(project="P2C", config=cfg, name=f"{cfg.P2C.CATE}_{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}")

    #设置cuDNN的benchmark模式为True，以加速卷积操作
    torch.backends.cudnn.benchmark = True

    ViPC_train = ViPCDataLoader(r'./train_list.txt',
                                 data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='train',
                                 view_align=False, category=cfg.P2C.CATE)        
    train_loader = DataLoader(ViPC_train,
                                  batch_size=cfg.P2C.BATCH_SIZE,
                                  num_workers=cfg.CONST.NUM_WORKERS,
                                  shuffle=True,
                                  drop_last=True,
                                  prefetch_factor=cfg.CONST.DATA_perfetch)  

    ViPC_test = ViPCDataLoader(r'./test_list.txt',
                                data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='test',
                                view_align=False, category=cfg.P2C.CATE)    

    test_loader = DataLoader(ViPC_test,
                                 batch_size=cfg.TRAIN.BATCH_SIZE,
                                 num_workers=cfg.CONST.NUM_WORKERS,
                                 shuffle=True,
                                 drop_last=True,
                                 prefetch_factor=cfg.CONST.DATA_perfetch)
    
    model=P2C(cfg).cuda()


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),  # 过滤掉不需要梯度的参数
                                 lr=cfg.TRAIN.LEARNING_RATE,  # 学习率
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,  # 权重衰减
                                 betas=cfg.TRAIN.BETAS)  # Adam优化算法的参数

    # lr scheduler
    scheduler_steplr = MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    #多步学习率调度器
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)

    init_epoch = 0  #初始化训练epoch为0
    best_metrics = float('inf')#初始化最佳指标为无穷大
    steps = 0#初始化步数为0
    best_epoch = 0
    #断点恢复

    checkpoint_path = os.path.join(cfg.DIR.CHECKPOINTS, 'P2C-latest.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        init_epoch = checkpoint['epoch']
        best_metrics = checkpoint.get('best_metrics', float('inf'))
        print(f"Resumed from epoch {init_epoch}")


    for epoch in range(init_epoch+1, cfg.P2C.max_epoch + 1):
        model.train()
        losses = AverageMeter(['TotalLoss', 'ShapeReconLoss', 'ShapeCompletionLoss', 'LatentLoss', 'NCCLoss'])

        n_batches = len(train_loader)#获取训练数据加载器的长度
        print('epoch:',epoch,'optimizer: ',optimizer.param_groups[0]['lr'])

        with tqdm(train_loader) as t:
            for batch_idex , (view,gt_pc,part_pc) in enumerate(t):
                view,gt_pc,part_pc = view.cuda(),gt_pc.cuda(),part_pc.cuda()
                part_pc=farthest_point_sample(part_pc,2048)
                
                gt_pc=farthest_point_sample(gt_pc,2048)
                
                #计算损失
                loss, shape_recon_loss, shape_compl_loss, latent_recon_loss, ncc_loss = model.get_loss(part_pc)
                #总损失，形状重建损失（CDL1），形状补全损失(CDL1)，潜在重建损失（smoothL1)，法向一致性约束损失

                
                #更新损失
                losses.update([loss.item(), shape_recon_loss.item(), shape_compl_loss.item(), latent_recon_loss.item(), ncc_loss.item()])


                
                optimizer.zero_grad()
                loss.backward()#总损失反向传播
                optimizer.step()
                #每个batch日志记录
                wandb.log({'total_loss':loss.item(),'ShapeReconLoss':shape_recon_loss.item(),'ShapeComLoss':shape_compl_loss.item(),'LatentLoss':latent_recon_loss.item(),'NCCLoss':ncc_loss.item,
                          'learning_rate': optimizer.param_groups[0]['lr']})
        #每个epoch日志记录
        lr_scheduler.step() #学习率调度器更新

        print(f"Epoch {epoch}, Total_Loss: {losses.avg(0)}, ShapeReconLoss: {losses.avg(1)}, ShapeCompletionLoss: {losses.avg(2)}, LatentLoss: {losses.avg(3)}, NCCLoss: {losses.avg(4)}")

        wandb.log({'Epoch': epoch, 
           'total_loss': losses.avg(0), 
           'ShapeReconLoss': losses.avg(1), 
           'ShapeComLoss': losses.avg(2), 
           'LatentLoss': losses.avg(3), 
           'NCCLoss': losses.avg(4)})


        #开始测试
        cd_eval=test_net(cfg,epoch,test_loader,model=model)

        wandb.log({"Val/CD_Loss": cd_eval, "Epoch": epoch}) #记录到WandB
        # Save checkpoints
        if epoch % cfg.P2C.SAVE_FREQ == 0 or cd_eval < best_metrics:
            if cd_eval < best_metrics:
                best_metrics = cd_eval
                best_epoch = epoch
                file_name = 'P2C-best.pth'

            else:
                file_name = 'P2C-epoch-%03d.pth' % epoch
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, output_path)

            logging.info('Saved checkpoint to %s ...' % output_path)

    logging.info(f"Training complete. Best performance at epoch {best_epoch} with CD loss {best_metrics:.4f}")
    wandb.finish()     #结束WandB        









if __name__ == "__main__":
    import config_vipc
    cfg = config_vipc.cfg
    train_net(cfg)
