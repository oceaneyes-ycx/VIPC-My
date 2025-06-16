'''
自监督训练代码
采用XMFNET的自监督设计


使用对比学习预训练的编码器进行训练
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

from end2endmodel import End2endNetwork,XMFnet
from lightly.loss.ntx_ent_loss import NTXentLoss

from utils.average_meter import AverageMeter, AverageMeter2
from torchvision import transforms
from datasets import data_utils as d_utils
from UTIL import mix_shapes_2,calc_cd,calc_dcd

from model_pre import Network
from renderer import Renderer

import matplotlib.pyplot as plt
import torch.nn.functional as F

#loss_cd=L1_ChamferLoss()#训练损失函数
#loss_cd_eval=L2_ChamferEval()#评估损失函数



def train_net(cfg):

    # 初始化WandB
    wandb.init(project="ViPC-Training-self-v2", config=cfg, name=f"{cfg.TRAIN.CATE}_{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}")

    #设置cuDNN的benchmark模式为True，以加速卷积操作
    torch.backends.cudnn.benchmark = True

    #创建ViPC训练数据加载器，使用自定义的ViPCDataLoader类
    ViPC_train = ViPCDataLoader_ft(r'./train_list_clean_plane2.txt',
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
    ViPC_test = ViPCDataLoader(r'./test_list.txt',
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
    model = Network(pretrained_model_path=r'best_model.pth') #创建模型
    model=model.cuda()
    #point_model=model.point_model
    #img_model=model.img_model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #渲染器相关
    colors=colors = torch.ones(size=(cfg.TRAIN.BATCH_SIZE, 2048, 1)).to(device)
    colors[:, :, 0] = colors[:, :, 0]*0.8
    render = Renderer(colors).to(device)

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
    for epoch in range(init_epoch+1, cfg.TRAIN.N_EPOCHS + 1):
        #遍历每个epoch
        model.train()

        total_cd_pc=0 #L1 CD loss
        #total_contra = 0 #对比学习损失
        total_loss = 0 #总损失
        #contra_loss = AverageMeter2() #对比学习总损失
        #train_intra_loss = AverageMeter2()
        #train_cross_loss = AverageMeter2()
        total_render_loss=0 #渲染损失
        

        n_batches = len(train_loader)#获取训练数据加载器的长度
        print('epoch:',epoch,'optimizer: ',optimizer.param_groups[0]['lr'])
        with tqdm(train_loader) as t:
            #2d视图，完整点云，部分点云，相机坐标，图像掩码，部分点云的部分点云
            for batch_idex, (view,gt_pc,part_pc,eye,mask,pc_partpart) in enumerate(t):
                gt=gt_pc.cuda()#GT点云

                partial=part_pc.cuda() #部分点云
                partpart=pc_partpart.cuda()#部分点云的部分点云
                png=view.cuda()#2D图像
                eye=eye.cuda()#相机坐标
                mask_gt=mask.cuda()#图像掩码



                #对部分点云和GT点云进行FPS采样
                partial=farthest_point_sample(partial,2048)#(4,2048,3)
                #gt=farthest_point_sample(gt,2048)#(4,2048,3)
                partpart=farthest_point_sample(partpart,2048)#(4,2048,3)
                partpart=partpart.permute(0,2,1) #(4,3,2048)
                partial=partial.permute(0,2,1) #(4,3,2048)

                mixed,mixed_gt,mixed_img=mix_shapes_2(partpart,partial,png) #混合部分点云和部分图像
                #mixed:(4,3,2048)  mixed_gt:(4,3,2048)  mixed_img:(4,3,224,224)

                
                mixed_gt=mixed_gt.permute(0,2,1) #(4,2048,3)

                partial=partial.permute(0,2,1) #(4,2048,3)
                batch_gt = torch.cat((mixed_gt, partial), dim = 0)
                #8,2048,3
                #使用cat函数将混合后的点云的GT和部分点云拼接
                batch_input = torch.cat((mixed, partpart), dim = 0)
                #8,3,2048
                #使用cat函数将混合后的点云和部分点云的部分点云拼接
                batch_view = torch.cat((mixed_img, png), dim = 0) 
                #8,3,224,224 
                #使用cat函数将混合后的点云的2D图像和2D图像拼接
                
                #2025/04/03  调试： 
                #batch_input: tensor: (8,3,2048) batch_view: tensor: (8,3,224,224) batch_gt: tensor: (8,2048,3)

                batch_input=batch_input.permute(0,2,1).contiguous() #调整维度为(8,2048,3)
                complete=model(batch_input, batch_view)   #输出补全的点云   (4,2048,3)
                # complete: tensor:(8,2048,3) 

                proj=render(complete[cfg.TRAIN.BATCH_SIZE:,...], eye)
                #使用补全之后的完整的点云和相机的位置进行渲染
                # #proj:渲染之后的图像  
                proj = proj.permute(0,3,1,2).squeeze(1)
                #调整维度并去掉第一个维度 

                difference=np.abs(proj[0].detach().cpu().numpy() -  mask_gt[0].cpu().numpy())
                #计算渲染图像和GT图像之间的差异

                plt.imsave('difference.png', difference)#存储差异图像
                plt.imsave('zao.png', proj[0].detach().cpu().numpy(), cmap = plt.get_cmap("binary"))
                #存储渲染图像
                plt.imsave('zao_GT.png', mask_gt[0].cpu().numpy(), cmap= plt.get_cmap("binary"))
                #存储GT图像

                H = np.array(np.mat('0.000009501, 0.000056320, 0.000215654, 0.000544067, 0.000929938, 0.001107336, 0.000929938, 0.000544067, 0.000215654, 0.000056320, 0.000009501; 0.000056320, 0.000313537, 0.001107336, 0.002543353, 0.003994979, 0.004589996, 0.003994979, 0.002543353, 0.001107336, 0.000313537, 0.000056320; 0.000215654, 0.001107336, 0.003454844, 0.006607893, 0.008327918, 0.008509345, 0.008327918, 0.006607893, 0.003454844, 0.001107336, 0.000215654; 0.000544067, 0.002543353, 0.006607893, 0.008265356, 0.002299816, -0.002872123, 0.002299816, 0.008265356, 0.006607893, 0.002543353, 0.000544067; 0.000929938, 0.003994979, 0.008327918, 0.002299816, -0.022397153, -0.039158923, -0.022397153, 0.002299816, 0.008327918, 0.003994979, 0.000929938; 0.001107336, 0.004589996, 0.008509345, -0.002872123, -0.039158923, -0.062876027, -0.039158923, -0.002872123, 0.008509345, 0.004589996, 0.001107336; 0.000929938, 0.003994979, 0.008327918, 0.002299816, -0.022397153, -0.039158923, -0.022397153, 0.002299816, 0.008327918, 0.003994979, 0.000929938; 0.000544067, 0.002543353, 0.006607893, 0.008265356, 0.002299816, -0.002872123, 0.002299816, 0.008265356, 0.006607893, 0.002543353, 0.000544067; 0.000215654, 0.001107336, 0.003454844, 0.006607893, 0.008327918, 0.008509345, 0.008327918, 0.006607893, 0.003454844, 0.001107336, 0.000215654; 0.000056320, 0.000313537, 0.001107336, 0.002543353, 0.003994979, 0.004589996, 0.003994979, 0.002543353, 0.001107336, 0.000313537, 0.000056320; 0.000009501, 0.000056320, 0.000215654, 0.000544067, 0.000929938, 0.001107336, 0.000929938, 0.000544067, 0.000215654, 0.000056320, 0.000009501'))
                #一个二维高斯滤波器的卷积核

                H = torch.from_numpy(H).to(device).unsqueeze(0).unsqueeze(0).float() 
                #将numpy数组转换为torch张量，并添加两个维度
                #使得H的形状从(11,11)变为(1,1,11,11)
                #这样做是为了在卷积操作中使用，pytorch的卷积操作需要输入的张量有四个维度
                #.float()将数据类型转换为float32，确保后续计算的精度一致

                edge = F.conv2d(mask_gt.unsqueeze(1), H, padding='same').squeeze(1)  
                #对GT图像进行卷积操作，得到边缘图

                tau = 0.02  #边缘图的阈值，只有edge大于tau的部分才会被保留，即标记为边缘
                edge_map = torch.where(edge>tau, 0.4, 1.0) 
                #加权边缘图，边缘区域（edge>tau）被赋值为0.4，其他区域被赋值为1.0(原始权重)
                #让模型更加关注物体内部区域而不是边缘区域的噪声
                loss_img = torch.mean(((proj-mask_gt)*edge_map)**2)   
                #渲染损失，通过计算渲染图像proj和真实掩码之间的均方误差来衡量
                loss_pc, _ , _ =calc_dcd(complete, batch_gt)
                #计算完成之后的点云和伪GT点云之间的DCD损失
                loss_pc = loss_pc.mean()#损失求平均

                loss_final=loss_pc + 0.10*loss_img #总损失


                #计算CD损失
                #loss_total=chamfer_sqrt(complete,batch_gt) #计算L1 CD loss

                #计算对比学习损失
                #batch_size = png.size(0)
               # data = torch.cat((pc_1, pc_2))
                #_ , point_feats = point_model(data.transpose(1, 2).contiguous())

                #point_t1_feats = point_feats[:batch_size, :]
                #point_t2_feats = point_feats[batch_size:, :]

                #loss_intra = criterion(point_t1_feats, point_t2_feats) #模态内损失

                #_ ,img_rgb_feat= img_model(png)

                #point_feats_combined = torch.stack([point_t1_feats, point_t2_feats]).mean(dim=0)

                #loss_cross = criterion(img_rgb_feat, point_feats_combined) #模态间损失

                #loss_contra = loss_intra + loss_cross #对比学习总损失

                #loss_total = cd + 0.15*loss_contra #总损失


                optimizer.zero_grad()
                loss_final.backward()
                optimizer.step()


                cd_pc_item=loss_pc.item()*1e3    #损失扩大化，用于更好显示
                render_item=0.10*loss_img.item()*1e3
                total_cd_pc+=cd_pc_item   #累加CD损失
                total_render_loss+=render_item

                loss_item = loss_final.item() * 1e3
                #total_loss += loss_item
                #contra_item=0.15*loss_contra.item()*1e3
                #total_contra+=contra_item
                total_loss+=loss_final.item()*1e3


                t.set_description(f"[Epoch {epoch}/{cfg.TRAIN.N_EPOCHS}] Batch {batch_idex+1}/{len(train_loader)}")
                t.set_postfix(cd_pc=loss_pc.item(),render_loss=loss_img.item()) #打印训练信息
        
        #计算平均CD损失
        avg_cdc=total_cd_pc/n_batches #平均L1 CD损失
        #avg_contra=total_contra/n_batches #平均对比学习损失
        avg_render=total_render_loss/n_batches #平均渲染损失
        wandb.log({"Train/CD_Loss":avg_cdc,"Train/Render_Loss":avg_render, "Epoch": epoch, "Learning_Rate": optimizer.param_groups[0]['lr']}) #记录到WandB

        avg_loss = total_loss / n_batches #平均总损失
        wandb.log({"Train/Total_Loss":avg_loss, "Epoch": epoch, "Learning_Rate": optimizer.param_groups[0]['lr'],
                    "Learning_Rate": optimizer.param_groups[0]['lr'],
                    "Grad_Norm": torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)})
        
        lr_scheduler.step() #学习率调度器更新

        print('epoch:',epoch,'cd_pc:',avg_cdc,'render_loss:',avg_render ,'loss:',avg_loss) #打印训练信息



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

                





