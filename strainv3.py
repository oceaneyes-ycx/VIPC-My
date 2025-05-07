'''
自监督训练代码
采用XMFNET的自监督设计

使用P2C的模型进行试验
2d视图使用resnet提取特征并使用AXFORM作为生成器

2025/04/30
暂时砍掉render LOSS
P2C暂时不使用NCC loss
'''
 
import torch
import os
import time
#from model_pre import Network
#from decoder.utils.utils import *
import wandb
import argparse
import logging
from torch.utils.data import DataLoader
from utils.ViPCdataloader import ViPCDataLoader, ViPCDataLoader3,ViPCDataLoader6,ViPCDataLoader_ft
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

from end2endmodel import End2endNetwork,XMFnet,P2C2
from lightly.loss.ntx_ent_loss import NTXentLoss

from utils.average_meter import AverageMeter, AverageMeter2
from torchvision import transforms
from datasets import data_utils as d_utils
from UTIL import mix_shapes_2,calc_cd,calc_dcd

from renderer import Renderer

import matplotlib.pyplot as plt
import torch.nn.functional as F

from P2C.P2C import Encoder, Decoder,P2C

#loss_cd=L1_ChamferLoss()#训练损失函数
#loss_cd_eval=L2_ChamferEval()#评估损失函数



def train_net(cfg):

    # 初始化WandB
    wandb.init(project="ViPC-Training-self-v3-2", config=cfg, name=f"{cfg.TRAIN.CATE}_{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}")

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
    

    #创建模型
    #model = P2C2().apply(weights_init_normal) #创建模型
    model=P2C2(cfg).cuda()
    #point_model=model.point_model
    #img_model=model.img_model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #渲染器相关
    #colors=colors = torch.ones(size=(cfg.TRAIN.BATCH_SIZE, 2048, 1)).to(device)
    #colors[:, :, 0] = colors[:, :, 0]*0.8
    #render = Renderer(colors).to(device)

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
    #criterion = NTXentLoss(temperature=0.1).cuda()    

    #开始训练
    for epoch in range(init_epoch+1, cfg.P2C.max_epoch + 1):
        #遍历每个epoch
        model.train()
        losses = AverageMeter(['TotalLoss', 'ShapeReconLoss', 'ShapeCompletionLoss', 'LatentLoss', 'NCCLoss'])

        #total_cd_pc=0 #L1 CD loss
        #total_contra = 0 #对比学习损失
        #total_loss = 0 #总损失
        #contra_loss = AverageMeter2() #对比学习总损失
        #train_intra_loss = AverageMeter2()
        #train_cross_loss = AverageMeter2()
        #total_render_loss=0 #渲染损失
        #total_p2c_loss=0 #P2C损失
        

        n_batches = len(train_loader)#获取训练数据加载器的长度
        print('epoch:',epoch,'optimizer: ',optimizer.param_groups[0]['lr'])
        with tqdm(train_loader) as t:
            #2d视图，完整点云，部分点云，相机坐标，图像掩码，部分点云的部分点云
            for batch_idex, (view,gt_pc,part_pc) in enumerate(t):
                gt=gt_pc.cuda()#GT点云

                partial=part_pc.cuda() #部分点云
                #partpart=pc_partpart.cuda()#部分点云的部分点云
                png=view.cuda()#2D图像
                #eye=eye.cuda()#相机坐标
                #mask_gt=mask.cuda()#图像掩码



                #对部分点云和GT点云进行FPS采样
                partial=farthest_point_sample(partial,2048)#(B,2048,3)
                #gt=farthest_point_sample(gt,2048)#(4,2048,3)
                #partpart=farthest_point_sample(partpart,2048)#(B,2048,3)
                #partpart=partpart.permute(0,2,1) #(B,3,2048)
                #partial=partial.permute(0,2,1) #(B,3,2048)

                #gt_pc=farthest_point_sample(gt_pc,2048)

                gt_pc=gt_pc.permute(0,2,1) #(B,3,2048)


                loss, shape_recon_loss, shape_compl_loss, latent_recon_loss, ncc_loss = model.get_loss(partial,png) #计算损失

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
        if epoch % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            if cd_eval < best_metrics:
                best_metrics = cd_eval
                best_epoch = epoch
                file_name = 'ckpt-best.pth'

            else:
                file_name = 'ckpt-epoch-%03d.pth' % epoch
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, output_path)

            logging.info('Saved checkpoint to %s ...' % output_path)

    logging.info(f"Training complete. Best performance at epoch {best_epoch} with CD loss {best_metrics:.4f}")
    wandb.finish()     #结束WandB

                
def weights_init_normal(m):
    """ Weights initialization with normal distribution.. Xavier """
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



if __name__ == "__main__":
    import config_vipc
    cfg = config_vipc.cfg
    train_net(cfg)

                





