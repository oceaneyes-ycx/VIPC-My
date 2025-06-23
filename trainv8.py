#复刻SymmCompletion
#代码微调，加入style_loss
'''

'''

import torch
import os
import time
from model_pre import Network
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
from tqdm import tqdm
from datetime import datetime
from testv1 import test_net

import open3d as o3d
import os
import numpy as np
from PIL import Image

from Symm.symm import SymmCompletion, SymmCompletion2, SymmCompletion3


loss_cd=L1_ChamferLoss()#训练损失函数
loss_cd_eval=L2_ChamferEval()#评估损失函数

def train_net(cfg):
    #接收一个配置文件作为参数

    # Initialize wandb
    wandb.init(project="ViPC-Training-symm-v2-test", config=cfg, name=f"{cfg.TRAIN.CATE}_{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}")

    #设置cuDNN的benchmark模式为True，以加速卷积操作
    torch.backends.cudnn.benchmark = True

    #创建ViPC训练数据加载器，使用自定义的ViPCDataLoader类
    ViPC_train = ViPCDataLoader(r'./train_list_clean_plane2.txt',
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
    ViPC_test = ViPCDataLoader(r'./test_reduced.txt',
                                data_path=cfg.DATASETS.SHAPENET.VIPC_PATH, status='test',
                                view_align=False, category=cfg.TRAIN.CATE)
    #创建测试数据加载器
    test_loader = DataLoader(ViPC_test,
                                 batch_size=cfg.TRAIN.BATCH_SIZE,
                                 num_workers=cfg.CONST.NUM_WORKERS,
                                 shuffle=True,
                                 drop_last=True,
                                 prefetch_factor=cfg.CONST.DATA_perfetch)
    
    
    model = SymmCompletion3(cfg)  
    model.cuda()
    
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


    #开始训练
    for epoch in range(init_epoch+1, cfg.TRAIN.N_EPOCHS + 1):
        
        #遍历每个epoch
        model.train()

        total_cd_pc= 0 #L1 CD loss
        total_style = 0

        total_loss = 0 #总损失

        n_batches = len(train_loader)#获取训练数据加载器的长度

        print('epoch:',epoch,'optimizer: ',optimizer.param_groups[0]['lr'])
        with tqdm(train_loader) as t:
            for batch_idex, (view,gt_pc,part_pc) in enumerate(t):

                partial=part_pc.cuda()
                gt=gt_pc.cuda()
                png=view.cuda()

                #对部分点云和GT点云进行FPS采样
                partial=farthest_point_sample(partial,2048)
                gt=farthest_point_sample(gt,2048)

                rets,complete,style_loss,_=model(partial,png)   #输出补全的点云

                #计算损失
                cd=model.get_loss(rets,gt,png)
                #cd2=model.get_loss(coarse,gt,png) #计算L1 CD loss

                loss_total=cd + 30*style_loss*1e-2 

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()


                cd_pc_item=loss_total.item()*1e3
                total_cd_pc+=cd_pc_item

                style_item = style_loss.item() * 1e1 *30 #风格损失
                total_style += style_item

                loss_item = loss_total.item() * 1e3
                total_loss += loss_item
                t.set_description(f"[Epoch {epoch}/{cfg.TRAIN.N_EPOCHS}] Batch {batch_idex+1}/{len(train_loader)}")
                t.set_postfix(cd_loss=loss_total.item())


        # Average CD loss for epoch
        avg_cdc = total_cd_pc / n_batches #平均L1 CD loss
        wandb.log({"Train/CD_Loss": avg_cdc, "Epoch": epoch, "Learning_Rate": optimizer.param_groups[0]['lr']})
        avg_style = total_style / n_batches #平均风格损失
        wandb.log({"Train/Style_Loss": avg_style, "Epoch": epoch, "Learning_Rate": optimizer.param_groups[0]['lr']})

        avg_loss = total_loss / n_batches #平均总损失
        wandb.log({"Train/Total_Loss": avg_loss, "Epoch": epoch, "Learning_Rate": optimizer.param_groups[0]['lr'],
                    "Learning_Rate": optimizer.param_groups[0]['lr'],
                    "Grad_Norm": torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)})

        lr_scheduler.step() #学习率调度器更新

        print('epoch:',epoch,'cd_pc:',avg_cdc, 'style_loss:', avg_style,'loss:', avg_loss)



        #开始测试
        cd_eval=test_net(cfg,epoch,test_loader,model=model)
        wandb.log({"Val/CD_Loss": cd_eval, "Epoch": epoch})


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
    wandb.finish()     





if __name__ == "__main__":
    import config_vipc
    cfg = config_vipc.cfg
    train_net(cfg)

                



